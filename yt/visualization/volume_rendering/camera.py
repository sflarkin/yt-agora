"""
Import the components of the volume rendering extension



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import __builtin__
import numpy as np

from yt.funcs import *
from yt.utilities.math_utils import *
from copy import deepcopy

from .grid_partitioner import HomogenizedVolume
from .transfer_functions import ProjectionTransferFunction

from yt.utilities.lib import \
    arr_vec2pix_nest, arr_pix2vec_nest, \
    arr_ang2pix_nest, arr_fisheye_vectors, lines, \
    PartitionedGrid, ProjectionSampler, VolumeRenderSampler, \
    LightSourceRenderSampler, InterpolatedProjectionSampler, \
    arr_vec2pix_nest, arr_pix2vec_nest, arr_ang2pix_nest, \
    pixelize_healpix, arr_fisheye_vectors, rotate_vectors

from yt.utilities.math_utils import get_rotation_matrix
from yt.utilities.orientation import Orientation
from yt.data_objects.api import ImageArray
from yt.visualization.image_writer import write_bitmap, write_image, apply_colormap
from yt.data_objects.data_containers import data_object_registry
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, ProcessorPool, parallel_objects
from yt.utilities.amr_kdtree.api import AMRKDTree
from .blenders import  enhance_rgba
from numpy import pi

def get_corners(le, re):
    return np.array([
      [le[0], le[1], le[2]],
      [re[0], le[1], le[2]],
      [re[0], re[1], le[2]],
      [le[0], re[1], le[2]],
      [le[0], le[1], re[2]],
      [re[0], le[1], re[2]],
      [re[0], re[1], re[2]],
      [le[0], re[1], re[2]],
    ], dtype='float64')

class Camera(ParallelAnalysisInterface):
    r"""A viewpoint into a volume, for volume rendering.

    The camera represents the eye of an observer, which will be used to
    generate ray-cast volume renderings of the domain.

    Parameters
    ----------
    center : array_like
        The current "center" of the view port -- the focal point for the
        camera.
    normal_vector : array_like
        The vector between the camera position and the center.
    width : float or list of floats
        The current width of the image.  If a single float, the volume is
        cubical, but if not, it is left/right, top/bottom, front/back.
    resolution : int or list of ints
        The number of pixels in each direction.
    transfer_function : `yt.visualization.volume_rendering.TransferFunction`
        The transfer function used to map values to colors in an image.  If
        not specified, defaults to a ProjectionTransferFunction.
    north_vector : array_like, optional
        The 'up' direction for the plane of rays.  If not specific, calculated
        automatically.
    steady_north : bool, optional
        Boolean to control whether to normalize the north_vector
        by subtracting off the dot product of it and the normal
        vector.  Makes it easier to do rotations along a single
        axis.  If north_vector is specified, is switched to
        True. Default: False
    volume : `yt.extensions.volume_rendering.HomogenizedVolume`, optional
        The volume to ray cast through.  Can be specified for finer-grained
        control, but otherwise will be automatically generated.
    fields : list of fields, optional
        This is the list of fields we want to volume render; defaults to
        Density.
    log_fields : list of bool, optional
        Whether we should take the log of the fields before supplying them to
        the volume rendering mechanism.
    sub_samples : int, optional
        The number of samples to take inside every cell per ray.
    pf : `~yt.data_objects.api.StaticOutput`
        For now, this is a require parameter!  But in the future it will become
        optional.  This is the parameter file to volume render.
    use_kd: bool, optional
        Specifies whether or not to use a kd-Tree framework for
        the Homogenized Volume and ray-casting.  Default to True.
    l_max: int, optional
        Specifies the maximum level to be rendered.  Also
        specifies the maximum level used in the kd-Tree
        construction.  Defaults to None (all levels), and only
        applies if use_kd=True.
    no_ghost: bool, optional
        Optimization option.  If True, homogenized bricks will
        extrapolate out from grid instead of interpolating from
        ghost zones that have to first be calculated.  This can
        lead to large speed improvements, but at a loss of
        accuracy/smoothness in resulting image.  The effects are
        less notable when the transfer function is smooth and
        broad. Default: True
    tree_type: string, optional
        Specifies the type of kd-Tree to be constructed/cast.
        There are three options, the default being 'domain'. Only
        affects parallel rendering.  'domain' is suggested.

        'domain' - Tree construction/casting is load balanced by
        splitting up the domain into the first N subtrees among N
        processors (N must be a power of 2).  Casting then
        proceeds with each processor rendering their subvolume,
        and final image is composited on the root processor.  The
        kd-Tree is never combined, reducing communication and
        memory overhead. The viewpoint can be changed without
        communication or re-partitioning of the data, making it
        ideal for rotations/spins.

        'breadth' - kd-Tree is first constructed as in 'domain',
        but then combined among all the subtrees.  Rendering is
        then split among N processors (again a power of 2), based
        on the N most expensive branches of the tree.  As in
        'domain', viewpoint can be changed without re-partitioning
        or communication.

        'depth' - kd-Tree is first constructed as in 'domain', but
        then combined among all subtrees.  Rendering is then load
        balanced in a back-to-front manner, splitting up the cost
        as evenly as possible.  If the viewpoint changes,
        additional data might have to be partitioned.  Is also
        prone to longer data IO times.  If all the data can fit in
        memory on each cpu, this can be the fastest option for
        multiple ray casts on the same dataset.
    le: array_like, optional
        Specifies the left edge of the volume to be rendered.
        Currently only works with use_kd=True.
    re: array_like, optional
        Specifies the right edge of the volume to be rendered.
        Currently only works with use_kd=True.

    Examples
    --------

    >>> cam = vr.Camera(c, L, W, (N,N), transfer_function = tf, pf = pf)
    >>> image = cam.snapshot()

    >>> from yt.mods import *
    >>> import yt.visualization.volume_rendering.api as vr
    
    >>> pf = EnzoStaticOutput('DD1701') # Load pf
    >>> c = [0.5]*3 # Center
    >>> L = [1.0,1.0,1.0] # Viewpoint
    >>> W = np.sqrt(3) # Width
    >>> N = 1024 # Pixels (1024^2)

    # Get density min, max
    >>> mi, ma = pf.h.all_data().quantities['Extrema']('Density')[0]
    >>> mi, ma = np.log10(mi), np.log10(ma)

    # Construct transfer function
    >>> tf = vr.ColorTransferFunction((mi-2, ma+2))
    # Sample transfer function with 5 gaussians.  Use new col_bounds keyword.
    >>> tf.add_layers(5,w=0.05, col_bounds = (mi+1,ma), colormap='spectral')
    
    # Create the camera object
    >>> cam = vr.Camera(c, L, W, (N,N), transfer_function=tf, pf=pf) 
    
    # Ray cast, and save the image.
    >>> image = cam.snapshot(fn='my_rendering.png')

    """
    _sampler_object = VolumeRenderSampler
    _pylab = None
    _tf_figure = None
    _render_figure = None
    def __init__(self, center, normal_vector, width,
                 resolution, transfer_function = None,
                 north_vector = None, steady_north=False,
                 volume = None, fields = None,
                 log_fields = None,
                 sub_samples = 5, pf = None,
                 use_kd=True, l_max=None, no_ghost=True,
                 tree_type='domain',
                 le=None, re=None, use_light=False):
        ParallelAnalysisInterface.__init__(self)
        if pf is not None: self.pf = pf
        if not iterable(resolution):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.sub_samples = sub_samples
        self.rotation_vector = north_vector
        if not iterable(width):
            width = (width, width, width) # left/right, top/bottom, front/back 
        self.orienter = Orientation(normal_vector, north_vector=north_vector, steady_north=steady_north)
        if not steady_north:
            self.rotation_vector = self.orienter.unit_vectors[1]
        self._setup_box_properties(width, center, self.orienter.unit_vectors)
        if fields is None: fields = ["Density"]
        self.fields = fields
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function
        self.log_fields = log_fields
        if self.log_fields is None:
            self.log_fields = [self.pf.field_info[f].take_log for f in self.fields]
        self.use_kd = use_kd
        self.l_max = l_max
        self.no_ghost = no_ghost
        self.use_light = use_light
        self.light_dir = None
        self.light_rgba = None
        if self.no_ghost:
            mylog.info('Warning: no_ghost is currently True (default). This may lead to artifacts at grid boundaries.')
        self.tree_type = tree_type
        if le is None: le = self.pf.domain_left_edge
        self.le = np.array(le)
        if re is None: re = self.pf.domain_right_edge
        self.re = np.array(re)
        if volume is None:
            if self.use_kd:
                volume = AMRKDTree(self.pf, l_max=l_max, fields=self.fields, no_ghost=no_ghost,
                                   log_fields = log_fields, le=self.le, re=self.re)
            else:
                volume = HomogenizedVolume(fields, pf = self.pf,
                                           log_fields = log_fields)
        else:
            self.use_kd = isinstance(volume, AMRKDTree)
        self.volume = volume        
        self.center = (self.re + self.le) / 2.0
        self.region = self.pf.h.region(self.center, self.le, self.re)

    def _setup_box_properties(self, width, center, unit_vectors):
        self.width = width
        self.center = center
        self.box_vectors = np.array([unit_vectors[0]*width[0],
                                     unit_vectors[1]*width[1],
                                     unit_vectors[2]*width[2]])
        self.origin = center - 0.5*np.dot(width,unit_vectors)
        self.back_center =  center - 0.5*width[2]*unit_vectors[2]
        self.front_center = center + 0.5*width[2]*unit_vectors[2]         

    def update_view_from_matrix(self, mat):
        pass

    def project_to_plane(self, pos, res=None):
        if res is None: 
            res = self.resolution
        dx = np.dot(pos - self.origin, self.orienter.unit_vectors[1])
        dy = np.dot(pos - self.origin, self.orienter.unit_vectors[0])
        dz = np.dot(pos - self.center, self.orienter.unit_vectors[2])
        # Transpose into image coords.
        py = (res[0]*(dx/self.width[0])).astype('int')
        px = (res[1]*(dy/self.width[1])).astype('int')
        return px, py, dz

    def draw_grids(self, im, alpha=0.3, cmap='algae', min_level=None, 
                   max_level=None):
        r"""Draws Grids on an existing volume rendering.

        By mapping grid level to a color, draws edges of grids on 
        a volume rendering using the camera orientation.

        Parameters
        ----------
        im: Numpy ndarray
            Existing image that has the same resolution as the Camera, 
            which will be painted by grid lines.
        alpha : float, optional
            The alpha value for the grids being drawn.  Used to control
            how bright the grid lines are with respect to the image.
            Default : 0.3
        cmap : string, optional
            Colormap to be used mapping grid levels to colors.
        min_level, max_level : int, optional
            Optional parameters to specify the min and max level grid boxes 
            to overplot on the image.  
        
        Returns
        -------
        None

        Examples
        --------
        >>> im = cam.snapshot() 
        >>> cam.add_grids(im)
        >>> write_bitmap(im, 'render_with_grids.png')

        """
        corners = self.region.grid_corners
        levels = self.region.grid_levels[:,0]

        if max_level is not None:
            subset = levels <= max_level
            levels = levels[subset]
            corners = corners[:,:,subset]
        if min_level is not None:
            subset = levels >= min_level
            levels = levels[subset]
            corners = corners[:,:,subset]
            
        colors = apply_colormap(levels*1.0,
                                color_bounds=[0,self.pf.h.max_level],
                                cmap_name=cmap)[0,:,:]*1.0/255.
        colors[:,3] = alpha

                
        order  = [0, 1, 1, 2, 2, 3, 3, 0]
        order += [4, 5, 5, 6, 6, 7, 7, 4]
        order += [0, 4, 1, 5, 2, 6, 3, 7]
        
        vertices = np.empty([corners.shape[2]*2*12,3])
        for i in xrange(3):
            vertices[:,i] = corners[order,i,:].ravel(order='F')

        px, py, dz = self.project_to_plane(vertices, res=im.shape[:2])
        
        # Must normalize the image
        nim = im.rescale(inline=False)
        enhance_rgba(nim)
        nim.add_background_color('black', inline=True)
       
        lines(nim, px, py, colors, 24)
        return nim

    def draw_coordinate_vectors(self, im, length=0.05, thickness=1):
        r"""Draws three coordinate vectors in the corner of a rendering.

        Modifies an existing image to have three lines corresponding to the
        coordinate directions colored by {x,y,z} = {r,g,b}.  Currently only
        functional for plane-parallel volume rendering.

        Parameters
        ----------
        im: Numpy ndarray
            Existing image that has the same resolution as the Camera,
            which will be painted by grid lines.
        length: float, optional
            The length of the lines, as a fraction of the image size.
            Default : 0.05
        thickness : int, optional
            Thickness in pixels of the line to be drawn.

        Returns
        -------
        None

        Modifies
        --------
        im: The original image.

        Examples
        --------
        >>> im = cam.snapshot()
        >>> cam.draw__coordinate_vectors(im)
        >>> im.write_png('render_with_grids.png')

        """
        length_pixels = length * self.resolution[0]
        # Put the starting point in the lower left
        px0 = int(length * self.resolution[0])
        # CS coordinates!
        py0 = int((1.0-length) * self.resolution[1])

        alpha = im[:, :, 3].max()
        if alpha == 0.0:
            alpha = 1.0

        coord_vectors = [np.array([length_pixels, 0.0, 0.0]),
                         np.array([0.0, length_pixels, 0.0]),
                         np.array([0.0, 0.0, length_pixels])]
        colors = [np.array([1.0, 0.0, 0.0, alpha]),
                  np.array([0.0, 1.0, 0.0, alpha]),
                  np.array([0.0, 0.0, 1.0, alpha])]

        for vec, color in zip(coord_vectors, colors):
            dx = int(np.dot(vec, self.orienter.unit_vectors[0]))
            dy = int(np.dot(vec, self.orienter.unit_vectors[1]))
            lines(im, np.array([px0, px0+dx]), np.array([py0, py0+dy]),
                  np.array([color, color]), 1, thickness)

    def draw_line(self, im, x0, x1, color=None):
        r"""Draws a line on an existing volume rendering.

        Given starting and ending positions x0 and x1, draws a line on 
        a volume rendering using the camera orientation.

        Parameters
        ----------
        im: Numpy ndarray
            Existing image that has the same resolution as the Camera, 
            which will be painted by grid lines.
        x0 : Numpy ndarray
            Starting coordinate, in simulation coordinates
        x1 : Numpy ndarray
            Ending coordinate, in simulation coordinates
        color : array like, optional
            Color of the line (r, g, b, a). Defaults to white. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> im = cam.snapshot() 
        >>> cam.draw_line(im, np.array([0.1,0.2,0.3], np.array([0.5,0.6,0.7)))
        >>> write_bitmap(im, 'render_with_line.png')

        """
        if color is None: color = np.array([1.0,1.0,1.0,1.0])

        dx0 = ((x0-self.origin)*self.orienter.unit_vectors[1]).sum()
        dx1 = ((x1-self.origin)*self.orienter.unit_vectors[1]).sum()
        dy0 = ((x0-self.origin)*self.orienter.unit_vectors[0]).sum()
        dy1 = ((x1-self.origin)*self.orienter.unit_vectors[0]).sum()
        py0 = int(self.resolution[0]*(dx0/self.width[0]))
        py1 = int(self.resolution[0]*(dx1/self.width[0]))
        px0 = int(self.resolution[1]*(dy0/self.width[1]))
        px1 = int(self.resolution[1]*(dy1/self.width[1]))
        lines(im, np.array([px0,px1]), np.array([py0,py1]), color=np.array([color,color]))

    def draw_domain(self,im,alpha=0.3):
        r"""Draws domain edges on an existing volume rendering.

        Draws a white wireframe on the domain edges.

        Parameters
        ----------
        im: Numpy ndarray
            Existing image that has the same resolution as the Camera, 
            which will be painted by grid lines.
        alpha : float, optional
            The alpha value for the wireframe being drawn.  Used to control
            how bright the lines are with respect to the image.
            Default : 0.3
        
        Returns
        -------
        None

        Examples
        --------
        >>> im = cam.snapshot() 
        >>> cam.draw_domain(im)
        >>> write_bitmap(im, 'render_with_domain_boundary.png')

        """
        # Must normalize the image
        nim = im.rescale(inline=False)
        enhance_rgba(nim)
        nim.add_background_color('black', inline=True)
 
        self.draw_box(nim, self.pf.domain_left_edge, self.pf.domain_right_edge,
                        color=np.array([1.0,1.0,1.0,alpha]))
        return nim

    def draw_box(self, im, le, re, color=None):
        r"""Draws a box on an existing volume rendering.

        Draws a box defined by a left and right edge by modifying an
        existing volume rendering

        Parameters
        ----------
        im: Numpy ndarray
            Existing image that has the same resolution as the Camera, 
            which will be painted by grid lines.
        le: Numpy ndarray
            Left corner of the box 
        re : Numpy ndarray
            Right corner of the box 
        color : array like, optional
            Color of the box (r, g, b, a). Defaults to white. 
        
        Returns
        -------
        None

        Examples
        --------
        >>> im = cam.snapshot() 
        >>> cam.draw_box(im, np.array([0.1,0.2,0.3], np.array([0.5,0.6,0.7)))
        >>> write_bitmap(im, 'render_with_box.png')

        """

        if color is None:
            color = np.array([1.0,1.0,1.0,1.0]) 
        corners = get_corners(le,re)
        order  = [0, 1, 1, 2, 2, 3, 3, 0]
        order += [4, 5, 5, 6, 6, 7, 7, 4]
        order += [0, 4, 1, 5, 2, 6, 3, 7]
        
        vertices = np.empty([24,3])
        for i in xrange(3):
            vertices[:,i] = corners[order,i,:].ravel(order='F')

        px, py, dz = self.project_to_plane(vertices, res=im.shape[:2])
       
        lines(im, px, py, color.reshape(1,4), 24)

    def look_at(self, new_center, north_vector = None):
        r"""Change the view direction based on a new focal point.

        This will recalculate all the necessary vectors and vector planes to orient
        the image plane so that it points at a new location.

        Parameters
        ----------
        new_center : array_like
            The new "center" of the view port -- the focal point for the
            camera.
        north_vector : array_like, optional
            The "up" direction for the plane of rays.  If not specific,
            calculated automatically.
        """
        normal_vector = self.front_center - new_center
        self.orienter.switch_orientation(normal_vector=normal_vector,
                                         north_vector = north_vector)

    def switch_view(self, normal_vector=None, width=None, center=None, north_vector=None):
        r"""Change the view based on any of the view parameters.

        This will recalculate the orientation and width based on any of
        normal_vector, width, center, and north_vector.

        Parameters
        ----------
        normal_vector: array_like, optional
            The new looking vector.
        width: float or array of floats, optional
            The new width.  Can be a single value W -> [W,W,W] or an
            array [W1, W2, W3] (left/right, top/bottom, front/back)
        center: array_like, optional
            Specifies the new center.
        north_vector : array_like, optional
            The 'up' direction for the plane of rays.  If not specific,
            calculated automatically.
        """
        if width is None:
            width = self.width
        if not iterable(width):
            width = (width, width, width) # left/right, tom/bottom, front/back 
        self.width = width
        if center is not None:
            self.center = center
        if north_vector is None:
            north_vector = self.orienter.north_vector
        if normal_vector is None:
            normal_vector = self.orienter.normal_vector
        self.orienter.switch_orientation(normal_vector = normal_vector,
                                         north_vector = north_vector)
        self._setup_box_properties(width, self.center, self.orienter.unit_vectors)
        
    def new_image(self):
        image = np.zeros((self.resolution[0], self.resolution[1], 4), dtype='float64', order='C')
        return image

    def get_sampler_args(self, image):
        rotp = np.concatenate([self.orienter.inv_mat.ravel('F'), self.back_center.ravel()])
        args = (rotp, self.box_vectors[2], self.back_center,
                (-self.width[0]/2.0, self.width[0]/2.0,
                 -self.width[1]/2.0, self.width[1]/2.0),
                image, self.orienter.unit_vectors[0], self.orienter.unit_vectors[1],
                np.array(self.width, dtype='float64'), self.transfer_function, self.sub_samples)
        return args

    star_trees = None
    def get_sampler(self, args):
        kwargs = {}
        if self.star_trees is not None:
            kwargs = {'star_list': self.star_trees}
        if self.use_light:
            if self.light_dir is None:
                self.set_default_light_dir()
            temp_dir = np.empty(3,dtype='float64')
            temp_dir = self.light_dir[0] * self.orienter.unit_vectors[1] + \
                    self.light_dir[1] * self.orienter.unit_vectors[2] + \
                    self.light_dir[2] * self.orienter.unit_vectors[0]
            if self.light_rgba is None:
                self.set_default_light_rgba()
            sampler = LightSourceRenderSampler(*args, light_dir=temp_dir,
                    light_rgba=self.light_rgba, **kwargs)
        else:
            sampler = self._sampler_object(*args, **kwargs)
        return sampler

    def finalize_image(self, image):
        view_pos = self.front_center + self.orienter.unit_vectors[2] * 1.0e6 * self.width[2]
        image = self.volume.reduce_tree_images(image, view_pos)
        if self.transfer_function.grey_opacity is False:
            image[:,:,3]=1.0
        return image

    def _render(self, double_check, num_threads, image, sampler):
        pbar = get_pbar("Ray casting", (self.volume.brick_dimensions + 1).prod(axis=-1).sum())
        total_cells = 0
        if double_check:
            for brick in self.volume.bricks:
                for data in brick.my_data:
                    if np.any(np.isnan(data)):
                        raise RuntimeError

        view_pos = self.front_center + self.orienter.unit_vectors[2] * 1.0e6 * self.width[2]
        for brick in self.volume.traverse(view_pos):
            sampler(brick, num_threads=num_threads)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)

        pbar.finish()
        image = sampler.aimage
        image = self.finalize_image(image)
        return image

    def show_tf(self):
        if self._pylab is None: 
            import pylab
            self._pylab = pylab
        if self._tf_figure is None:
            self._tf_figure = self._pylab.figure(2)
            self.transfer_function.show(ax=self._tf_figure.axes)
        self._pylab.draw()

    def annotate(self, ax, enhance=True):
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        cb = self._pylab.colorbar(ax.images[0], pad=0.0, fraction=0.05, drawedges=True, shrink=0.9)
        label = self.pf.field_info[self.fields[0]].get_label()
        if self.log_fields[0]:
            label = '$\\rm{log}\\/ $' + label
        self.transfer_function.vert_cbar(ax=cb.ax, label=label)

    def show_mpl(self, im, enhance=True):
        if self._pylab is None:
            import pylab
            self._pylab = pylab
        if self._render_figure is None:
            self._render_figure = self._pylab.figure(1)
        self._render_figure.clf()

        if enhance:
            nz = im[im > 0.0]
            nim = im / (nz.mean() + 6.0 * np.std(nz))
            nim[nim > 1.0] = 1.0
            nim[nim < 0.0] = 0.0
            del nz
        else:
            nim = im
        ax = self._pylab.imshow(nim[:,:,:3]/nim[:,:,:3].max(), origin='upper')
        return ax

    def draw(self):
        self._pylab.draw()
    
    def save_annotated(self, fn, image, enhance=True, dpi=100):
        image = image.swapaxes(0,1) 
        ax = self.show_mpl(image, enhance=enhance)
        self.annotate(ax.axes, enhance)
        self._pylab.savefig(fn, bbox_inches='tight', facecolor='black', dpi=dpi)
        
    def save_image(self, image, fn=None, clip_ratio=None, transparent=False):
        if self.comm.rank == 0 and fn is not None:
            if transparent:
                image.write_png(fn, clip_ratio=clip_ratio, rescale=True,
                                background=None)
            else:
                image.write_png(fn, clip_ratio=clip_ratio, rescale=True,
                                background='black')

    def initialize_source(self):
        return self.volume.initialize_source()

    def get_information(self):
        info_dict = {'fields':self.fields,
                     'type':self.__class__.__name__,
                     'east_vector':self.orienter.unit_vectors[0],
                     'north_vector':self.orienter.unit_vectors[1],
                     'normal_vector':self.orienter.unit_vectors[2],
                     'width':self.width,
                     'dataset':self.pf.fullpath}
        return info_dict

    def snapshot(self, fn = None, clip_ratio = None, double_check = False,
                 num_threads = 0, transparent=False):
        r"""Ray-cast the camera.

        This method instructs the camera to take a snapshot -- i.e., call the ray
        caster -- based on its current settings.

        Parameters
        ----------
        fn : string, optional
            If supplied, the image will be saved out to this before being
            returned.  Scaling will be to the maximum value.
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()
        double_check : bool, optional
            Optionally makes sure that the data contains only valid entries.
            Used for debugging.
        num_threads : int, optional
            If supplied, will use 'num_threads' number of OpenMP threads during
            the rendering.  Defaults to 0, which uses the environment variable
            OMP_NUM_THREADS.
        transparent: bool, optional
            Optionally saves out the 4-channel rgba image, which can appear 
            empty if the alpha channel is low everywhere. Default: False

        Returns
        -------
        image : array
            An (N,M,3) array of the final returned values, in float64 form.
        """
        if num_threads is None:
            num_threads=get_num_threads()
        image = self.new_image()
        args = self.get_sampler_args(image)
        sampler = self.get_sampler(args)
        self.initialize_source()
        image = ImageArray(self._render(double_check, num_threads, 
                                        image, sampler),
                           info=self.get_information())
        self.save_image(image, fn=fn, clip_ratio=clip_ratio, 
                       transparent=transparent)
        return image

    def show(self, clip_ratio = None):
        r"""This will take a snapshot and display the resultant image in the
        IPython notebook.

        If yt is being run from within an IPython session, and it is able to
        determine this, this function will snapshot and send the resultant
        image to the IPython notebook for display.

        If yt can't determine if it's inside an IPython session, it will raise
        YTNotInsideNotebook.

        Parameters
        ----------
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()

        Examples
        --------

        >>> cam.show()

        """
        if "__IPYTHON__" in dir(__builtin__):
            from IPython.core.displaypub import publish_display_data
            image = self.snapshot()[:,:,:3]
            if clip_ratio is not None: clip_ratio *= image.std()
            data = write_bitmap(image, None, clip_ratio)
            publish_display_data(
                'yt.visualization.volume_rendering.camera.Camera',
                {'image/png' : data}
            )
        else:
            raise YTNotInsideNotebook


    def set_default_light_dir(self):
        self.light_dir = [1.,1.,1.]

    def set_default_light_rgba(self):
        self.light_rgba = [1.,1.,1.,1.]

    def zoom(self, factor):
        r"""Change the distance to the focal point.

        This will zoom the camera in by some `factor` toward the focal point,
        along the current view direction, modifying the left/right and up/down
        extents as well.

        Parameters
        ----------
        factor : float
            The factor by which to reduce the distance to the focal point.


        Notes
        -----

        You will need to call snapshot() again to get a new image.

        """
        self.width = [w / factor for w in self.width]
        self._setup_box_properties(self.width, self.center, self.orienter.unit_vectors)

    def zoomin(self, final, n_steps, clip_ratio = None):
        r"""Loop over a zoomin and return snapshots along the way.

        This will yield `n_steps` snapshots until the current view has been
        zooming in to a final factor of `final`.

        Parameters
        ----------
        final : float
            The zoom factor, with respect to current, desired at the end of the
            sequence.
        n_steps : int
            The number of zoom snapshots to make.
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()


        Examples
        --------

        >>> for i, snapshot in enumerate(cam.zoomin(100.0, 10)):
        ...     iw.write_bitmap(snapshot, "zoom_%04i.png" % i)
        """
        f = final**(1.0/n_steps)
        for i in xrange(n_steps):
            self.zoom(f)
            yield self.snapshot(clip_ratio = clip_ratio)

    def move_to(self, final, n_steps, final_width=None, exponential=False, clip_ratio = None):
        r"""Loop over a look_at

        This will yield `n_steps` snapshots until the current view has been
        moved to a final center of `final` with a final width of final_width.

        Parameters
        ----------
        final : array_like
            The final center to move to after `n_steps`
        n_steps : int
            The number of look_at snapshots to make.
        final_width: float or array_like, optional
            Specifies the final width after `n_steps`.  Useful for
            moving and zooming at the same time.
        exponential : boolean
            Specifies whether the move/zoom transition follows an
            exponential path toward the destination or linear
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()
            
        Examples
        --------

        >>> for i, snapshot in enumerate(cam.move_to([0.2,0.3,0.6], 10)):
        ...     iw.write_bitmap(snapshot, "move_%04i.png" % i)
        """
        self.center = np.array(self.center)
        dW = None
        if exponential:
            if final_width is not None:
                if not iterable(final_width):
                    width = np.array([final_width, final_width, final_width]) 
                    # left/right, top/bottom, front/back 
                if (self.center == 0.0).all():
                    self.center += (np.array(final) - self.center) / (10. * n_steps)
                final_zoom = final_width/np.array(self.width)
                dW = final_zoom**(1.0/n_steps)
            else:
                dW = np.array([1.0,1.0,1.0])
            position_diff = (np.array(final)/self.center)*1.0
            dx = position_diff**(1.0/n_steps)
        else:
            if final_width is not None:
                if not iterable(final_width):
                    width = np.array([final_width, final_width, final_width]) 
                    # left/right, top/bottom, front/back
                dW = (1.0*final_width-np.array(self.width))/n_steps
            else:
                dW = np.array([0.0,0.0,0.0])
            dx = (np.array(final)-self.center)*1.0/n_steps
        for i in xrange(n_steps):
            if exponential:
                self.switch_view(center=self.center*dx, width=self.width*dW)
            else:
                self.switch_view(center=self.center+dx, width=self.width+dW)
            yield self.snapshot(clip_ratio = clip_ratio)

    def rotate(self, theta, rot_vector=None):
        r"""Rotate by a given angle

        Rotate the view.  If `rot_vector` is None, rotation will occur
        around the `north_vector`.

        Parameters
        ----------
        theta : float, in radians
             Angle (in radians) by which to rotate the view.
        rot_vector  : array_like, optional
            Specify the rotation vector around which rotation will
            occur.  Defaults to None, which sets rotation around
            `north_vector`

        Examples
        --------

        >>> cam.rotate(np.pi/4)
        """
        rotate_all = rot_vector is not None
        if rot_vector is None:
            rot_vector = self.rotation_vector
        else:
            rot_vector = ensure_numpy_array(rot_vector)
            rot_vector = rot_vector/np.linalg.norm(rot_vector)
          
        R = get_rotation_matrix(theta, rot_vector)

        normal_vector = self.front_center-self.center

        if rotate_all:
            self.switch_view(
                normal_vector=np.dot(R, normal_vector),
                north_vector=np.dot(R, self.orienter.unit_vectors[1]))
        else:
            self.switch_view(normal_vector=np.dot(R, normal_vector))


    def pitch(self, theta):
        r"""Rotate by a given angle about the horizontal axis

        Pitch the view.

        Parameters
        ----------
        theta : float, in radians
             Angle (in radians) by which to pitch the view.

        Examples
        --------

        >>> cam.roll(np.pi/4)
        """
        rot_vector = self.orienter.unit_vectors[0]
        R = get_rotation_matrix(theta, rot_vector)
        normal_vector = self.front_center-self.center
        self.switch_view(
                normal_vector=np.dot(R, self.orienter.unit_vectors[2]),
                north_vector=np.dot(R, self.orienter.unit_vectors[1]))
        if self.orienter.steady_north:
            self.orienter.north_vector = self.orienter.unit_vectors[1]
 
    def yaw(self, theta):
        r"""Rotate by a given angle about the vertical axis

        Yaw the view.

        Parameters
        ----------
        theta : float, in radians
             Angle (in radians) by which to yaw the view.

        Examples
        --------

        >>> cam.roll(np.pi/4)
        """
        rot_vector = self.orienter.unit_vectors[1]
        R = get_rotation_matrix(theta, rot_vector)
        normal_vector = self.front_center-self.center
        self.switch_view(
                normal_vector=np.dot(R, self.orienter.unit_vectors[2]))
 
    def roll(self, theta):
        r"""Rotate by a given angle about the view normal axis

        Roll the view.

        Parameters
        ----------
        theta : float, in radians
             Angle (in radians) by which to roll the view.

        Examples
        --------

        >>> cam.roll(np.pi/4)
        """
        rot_vector = self.orienter.unit_vectors[2]
        R = get_rotation_matrix(theta, rot_vector)
        self.switch_view(
                normal_vector=np.dot(R, self.orienter.unit_vectors[2]),
                north_vector=np.dot(R, self.orienter.unit_vectors[1]))
        if self.orienter.steady_north:
            self.orienter.north_vector = np.dot(R, self.orienter.north_vector)

    def rotation(self, theta, n_steps, rot_vector=None, clip_ratio = None):
        r"""Loop over rotate, creating a rotation

        This will yield `n_steps` snapshots until the current view has been
        rotated by an angle `theta`

        Parameters
        ----------
        theta : float, in radians
            Angle (in radians) by which to rotate the view.
        n_steps : int
            The number of look_at snapshots to make.
        rot_vector  : array_like, optional
            Specify the rotation vector around which rotation will
            occur.  Defaults to None, which sets rotation around the
            original `north_vector`
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()

        Examples
        --------

        >>> for i, snapshot in enumerate(cam.rotation(np.pi, 10)):
        ...     iw.write_bitmap(snapshot, 'rotation_%04i.png' % i)
        """

        dtheta = (1.0*theta)/n_steps
        for i in xrange(n_steps):
            self.rotate(dtheta, rot_vector=rot_vector)
            yield self.snapshot(clip_ratio = clip_ratio)

data_object_registry["camera"] = Camera

class InteractiveCamera(Camera):
    frames = []

    def snapshot(self, fn = None, clip_ratio = None):
        import matplotlib.pylab as pylab
        pylab.figure(2)
        self.transfer_function.show()
        pylab.draw()
        im = Camera.snapshot(self, fn, clip_ratio)
        pylab.figure(1)
        pylab.imshow(im / im.max())
        pylab.draw()
        self.frames.append(im)

    def rotation(self, theta, n_steps, rot_vector=None):
        for frame in Camera.rotation(self, theta, n_steps, rot_vector):
            if frame is not None:
                self.frames.append(frame)

    def zoomin(self, final, n_steps):
        for frame in Camera.zoomin(self, final, n_steps):
            if frame is not None:
                self.frames.append(frame)

    def clear_frames(self):
        del self.frames
        self.frames = []

    def save(self,fn):
        self._pylab.savefig(fn, bbox_inches='tight', facecolor='black')

    def save_frames(self, basename, clip_ratio=None):
        for i, frame in enumerate(self.frames):
            fn = basename + '_%04i.png'%i
            if clip_ratio is not None:
                write_bitmap(frame, fn, clip_ratio*image.std())
            else:
                write_bitmap(frame, fn)

data_object_registry["interactive_camera"] = InteractiveCamera

class PerspectiveCamera(Camera):
    expand_factor = 1.0
    def __init__(self, *args, **kwargs):
        self.expand_factor = kwargs.pop('expand_factor', 1.0)
        Camera.__init__(self, *args, **kwargs)

    def get_sampler_args(self, image):
        # We should move away from pre-generation of vectors like this and into
        # the usage of on-the-fly generation in the VolumeIntegrator module
        # We might have a different width and back_center
        dl = (self.back_center - self.front_center)
        self.front_center += self.expand_factor*dl
        self.back_center -= dl

        px = np.linspace(-self.width[0]/2.0, self.width[0]/2.0,
                         self.resolution[0])[:,None]
        py = np.linspace(-self.width[1]/2.0, self.width[1]/2.0,
                         self.resolution[1])[None,:]
        inv_mat = self.orienter.inv_mat
        positions = np.zeros((self.resolution[0], self.resolution[1], 3),
                          dtype='float64', order='C')
        positions[:,:,0] = inv_mat[0,0]*px+inv_mat[0,1]*py+self.back_center[0]
        positions[:,:,1] = inv_mat[1,0]*px+inv_mat[1,1]*py+self.back_center[1]
        positions[:,:,2] = inv_mat[2,0]*px+inv_mat[2,1]*py+self.back_center[2]
        bounds = (px.min(), px.max(), py.min(), py.max())

        # We are likely adding on an odd cutting condition here
        vectors = self.front_center - positions
        positions = self.front_center - 1.0*(((self.back_center-self.front_center)**2).sum())**0.5*vectors
        vectors = (self.front_center - positions)

        uv = np.ones(3, dtype='float64')
        image.shape = (self.resolution[0]**2,1,4)
        vectors.shape = (self.resolution[0]**2,1,3)
        positions.shape = (self.resolution[0]**2,1,3)
        args = (positions, vectors, self.back_center, 
                (0.0,1.0,0.0,1.0),
                image, uv, uv,
                np.zeros(3, dtype='float64'), 
                self.transfer_function, self.sub_samples)
        return args

    def _render(self, double_check, num_threads, image, sampler):
        pbar = get_pbar("Ray casting", (self.volume.brick_dimensions + 1).prod(axis=-1).sum())
        total_cells = 0
        if double_check:
            for brick in self.volume.bricks:
                for data in brick.my_data:
                    if np.any(np.isnan(data)):
                        raise RuntimeError

        for brick in self.volume.traverse(self.front_center):
            sampler(brick, num_threads=num_threads)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)

        pbar.finish()
        image = self.finalize_image(sampler.aimage)
        return image

    def finalize_image(self, image):
        view_pos = self.front_center
        image.shape = self.resolution[0], self.resolution[0], 4
        image = self.volume.reduce_tree_images(image, view_pos)
        if self.transfer_function.grey_opacity is False:
            image[:,:,3]=1.0
        return image

def corners(left_edge, right_edge):
    return np.array([
      [left_edge[:,0], left_edge[:,1], left_edge[:,2]],
      [right_edge[:,0], left_edge[:,1], left_edge[:,2]],
      [right_edge[:,0], right_edge[:,1], left_edge[:,2]],
      [right_edge[:,0], right_edge[:,1], right_edge[:,2]],
      [left_edge[:,0], right_edge[:,1], right_edge[:,2]],
      [left_edge[:,0], left_edge[:,1], right_edge[:,2]],
      [right_edge[:,0], left_edge[:,1], right_edge[:,2]],
      [left_edge[:,0], right_edge[:,1], left_edge[:,2]],
    ], dtype='float64')

class HEALpixCamera(Camera):

    _sampler_object = None 
    
    def __init__(self, center, radius, nside,
                 transfer_function = None, fields = None,
                 sub_samples = 5, log_fields = None, volume = None,
                 pf = None, use_kd=True, no_ghost=False, use_light=False,
                 inner_radius = 10):
        print "Because of recent relicensing, we currently cannot provide"
        print "HEALpix functionality.  Please visit yt-users for more"
        print "information."
        raise NotImplementedError
        ParallelAnalysisInterface.__init__(self)
        if pf is not None: self.pf = pf
        self.center = np.array(center, dtype='float64')
        self.radius = radius
        self.inner_radius = inner_radius
        self.nside = nside
        self.use_kd = use_kd
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function

        if isinstance(self.transfer_function, ProjectionTransferFunction):
            self._sampler_object = InterpolatedProjectionSampler
            self._needs_tf = 0
        else:
            self._sampler_object = VolumeRenderSampler
            self._needs_tf = 1

        if fields is None: fields = ["Density"]
        self.fields = fields
        self.sub_samples = sub_samples
        self.log_fields = log_fields
        if self.log_fields is None:
            self.log_fields = [self.pf.field_info[f].take_log for f in self.fields]
        self.use_light = use_light
        self.light_dir = None
        self.light_rgba = None
        if volume is None:
            volume = AMRKDTree(self.pf, fields=self.fields, no_ghost=no_ghost,
                               log_fields=log_fields)
        self.use_kd = isinstance(volume, AMRKDTree)
        self.volume = volume

    def new_image(self):
        image = np.zeros((12 * self.nside ** 2, 1, 4), dtype='float64', order='C')
        return image

    def get_sampler_args(self, image):
        nv = 12 * self.nside ** 2
        vs = arr_pix2vec_nest(self.nside, np.arange(nv))
        vs.shape = (nv, 1, 3)
        vs += 1e-8
        uv = np.ones(3, dtype='float64')
        positions = np.ones((nv, 1, 3), dtype='float64') * self.center
        dx = min(g.dds.min() for g in self.pf.h.find_point(self.center)[0])
        positions += self.inner_radius * dx * vs
        vs *= self.radius
        args = (positions, vs, self.center,
                (0.0, 1.0, 0.0, 1.0),
                image, uv, uv,
                np.zeros(3, dtype='float64'))
        if self._needs_tf:
            args += (self.transfer_function,)
        args += (self.sub_samples,)
        return args

    def _render(self, double_check, num_threads, image, sampler):
        pbar = get_pbar("Ray casting", (self.volume.brick_dimensions + 1).prod(axis=-1).sum())
        total_cells = 0
        if double_check:
            for brick in self.volume.bricks:
                for data in brick.my_data:
                    if np.any(np.isnan(data)):
                        raise RuntimeError
        
        view_pos = self.center
        for brick in self.volume.traverse(view_pos):
            sampler(brick, num_threads=num_threads)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)
        
        pbar.finish()
        image = sampler.aimage

        self.finalize_image(image)

        return image

    def finalize_image(self, image):
        view_pos = self.center
        image = self.volume.reduce_tree_images(image, view_pos)
        return image

    def get_information(self):
        info_dict = {'fields':self.fields,
                     'type':self.__class__.__name__,
                     'center':self.center,
                     'radius':self.radius,
                     'dataset':self.pf.fullpath}
        return info_dict


    def snapshot(self, fn = None, clip_ratio = None, double_check = False,
                 num_threads = 0, clim = None, label = None):
        r"""Ray-cast the camera.

        This method instructs the camera to take a snapshot -- i.e., call the ray
        caster -- based on its current settings.

        Parameters
        ----------
        fn : string, optional
            If supplied, the image will be saved out to this before being
            returned.  Scaling will be to the maximum value.
        clip_ratio : float, optional
            If supplied, the 'max_val' argument to write_bitmap will be handed
            clip_ratio * image.std()

        Returns
        -------
        image : array
            An (N,M,3) array of the final returned values, in float64 form.
        """
        if num_threads is None:
            num_threads=get_num_threads()
        image = self.new_image()
        args = self.get_sampler_args(image)
        sampler = self.get_sampler(args)
        self.volume.initialize_source()
        image = ImageArray(self._render(double_check, num_threads, 
                                        image, sampler),
                           info=self.get_information())
        self.save_image(image, fn=fn, clim=clim, label = label)
        return image

    def save_image(self, image, fn=None, clim=None, label = None):
        if self.comm.rank == 0 and fn is not None:
            # This assumes Density; this is a relatively safe assumption.
            if label is None:
                label = "Projected %s" % (self.fields[0])
            if clim is not None:
                cmin, cmax = clim
            else:
                cmin = cmax = None
            plot_allsky_healpix(image[:,0,0], self.nside, fn, label, 
                                cmin = cmin, cmax = cmax)

class AdaptiveHEALpixCamera(Camera):
    def __init__(self, center, radius, nside,
                 transfer_function = None, fields = None,
                 sub_samples = 5, log_fields = None, volume = None,
                 pf = None, use_kd=True, no_ghost=False,
                 rays_per_cell = 0.1, max_nside = 8192):
        raise NotImplementedError
        ParallelAnalysisInterface.__init__(self)
        if pf is not None: self.pf = pf
        self.center = np.array(center, dtype='float64')
        self.radius = radius
        self.use_kd = use_kd
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function
        if fields is None: fields = ["Density"]
        self.fields = fields
        self.sub_samples = sub_samples
        self.log_fields = log_fields
        if volume is None:
            volume = AMRKDTree(self.pf, fields=self.fields, no_ghost=no_ghost,
                               log_fields=log_fields)
        self.use_kd = isinstance(volume, AMRKDTree)
        self.volume = volume
        self.initial_nside = nside
        self.rays_per_cell = rays_per_cell
        self.max_nside = max_nside

    def snapshot(self, fn = None):
        tfp = TransferFunctionProxy(self.transfer_function)
        tfp.ns = self.sub_samples
        self.volume.initialize_source()
        mylog.info("Adaptively rendering.")
        pbar = get_pbar("Ray casting",
                        (self.volume.brick_dimensions + 1).prod(axis=-1).sum())
        total_cells = 0
        bricks = [b for b in self.volume.traverse(None, self.center, None)][::-1]
        left_edges = np.array([b.LeftEdge for b in bricks])
        right_edges = np.array([b.RightEdge for b in bricks])
        min_dx = min(((b.RightEdge[0] - b.LeftEdge[0])/b.my_data[0].shape[0]
                     for b in bricks))
        # We jitter a bit if we're on a boundary of our initial grid
        for i in range(3):
            if bricks[0].LeftEdge[i] == self.center[i]:
                self.center += 1e-2 * min_dx
            elif bricks[0].RightEdge[i] == self.center[i]:
                self.center -= 1e-2 * min_dx
        ray_source = AdaptiveRaySource(self.center, self.rays_per_cell,
                                       self.initial_nside, self.radius,
                                       bricks, left_edges, right_edges, self.max_nside)
        for i,brick in enumerate(bricks):
            ray_source.integrate_brick(brick, tfp, i, left_edges, right_edges,
                                       bricks)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)
        pbar.finish()
        info, values = ray_source.get_rays()
        return info, values

class StereoPairCamera(Camera):
    def __init__(self, original_camera, relative_separation = 0.005):
        ParallelAnalysisInterface.__init__(self)
        self.original_camera = original_camera
        self.relative_separation = relative_separation

    def split(self):
        oc = self.original_camera
        uv = oc.orienter.unit_vectors
        c = oc.center
        fc = oc.front_center
        wx, wy, wz = oc.width
        left_normal = fc + uv[1] * 0.5*self.relative_separation * wx - c
        right_normal = fc - uv[1] * 0.5*self.relative_separation * wx - c
        left_camera = Camera(c, left_normal, oc.width,
                             oc.resolution, oc.transfer_function, north_vector=uv[0],
                             volume=oc.volume, fields=oc.fields, log_fields=oc.log_fields,
                             sub_samples=oc.sub_samples, pf=oc.pf)
        right_camera = Camera(c, right_normal, oc.width,
                             oc.resolution, oc.transfer_function, north_vector=uv[0],
                             volume=oc.volume, fields=oc.fields, log_fields=oc.log_fields,
                             sub_samples=oc.sub_samples, pf=oc.pf)
        return (left_camera, right_camera)

class FisheyeCamera(Camera):
    def __init__(self, center, radius, fov, resolution,
                 transfer_function = None, fields = None,
                 sub_samples = 5, log_fields = None, volume = None,
                 pf = None, no_ghost=False, rotation = None, use_light=False):
        ParallelAnalysisInterface.__init__(self)
        self.use_light = use_light
        self.light_dir = None
        self.light_rgba = None
        if rotation is None: rotation = np.eye(3)
        self.rotation_matrix = rotation
        if pf is not None: self.pf = pf
        self.center = np.array(center, dtype='float64')
        self.radius = radius
        self.fov = fov
        if iterable(resolution):
            raise RuntimeError("Resolution must be a single int")
        self.resolution = resolution
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function
        if fields is None: fields = ["Density"]
        self.fields = fields
        self.sub_samples = sub_samples
        self.log_fields = log_fields
        if volume is None:
            volume = AMRKDTree(self.pf, fields=self.fields, no_ghost=no_ghost,
                               log_fields=log_fields)
        self.volume = volume

    def get_information(self):
        return {}

    def new_image(self):
        image = np.zeros((self.resolution**2,1,4), dtype='float64', order='C')
        return image
        
    def get_sampler_args(self, image):
        vp = arr_fisheye_vectors(self.resolution, self.fov)
        vp.shape = (self.resolution**2,1,3)
        vp2 = vp.copy()
        for i in range(3):
            vp[:,:,i] = (vp2 * self.rotation_matrix[:,i]).sum(axis=2)
        del vp2
        vp *= self.radius
        uv = np.ones(3, dtype='float64')
        positions = np.ones((self.resolution**2, 1, 3), dtype='float64') * self.center

        args = (positions, vp, self.center,
                (0.0, 1.0, 0.0, 1.0),
                image, uv, uv,
                np.zeros(3, dtype='float64'),
                self.transfer_function, self.sub_samples)
        return args


    def finalize_image(self, image):
        image.shape = self.resolution, self.resolution, 4

    def _render(self, double_check, num_threads, image, sampler):
        pbar = get_pbar("Ray casting", (self.volume.brick_dimensions + 1).prod(axis=-1).sum())
        total_cells = 0
        if double_check:
            for brick in self.volume.bricks:
                for data in brick.my_data:
                    if np.any(np.isnan(data)):
                        raise RuntimeError
        
        view_pos = self.center
        for brick in self.volume.traverse(view_pos):
            sampler(brick, num_threads=num_threads)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)
        
        pbar.finish()
        image = sampler.aimage

        self.finalize_image(image)

        return image

class MosaicCamera(Camera):
    def __init__(self, center, normal_vector, width,
                 resolution, transfer_function = None,
                 north_vector = None, steady_north=False,
                 volume = None, fields = None,
                 log_fields = None,
                 sub_samples = 5, pf = None,
                 use_kd=True, l_max=None, no_ghost=True,
                 tree_type='domain',expand_factor=1.0,
                 le=None, re=None, nimx=1, nimy=1, procs_per_wg=None,
                 preload=True, use_light=False):

        ParallelAnalysisInterface.__init__(self)

        self.procs_per_wg = procs_per_wg
        if pf is not None: self.pf = pf
        if not iterable(resolution):
            resolution = (int(resolution/nimx), int(resolution/nimy))
        self.resolution = resolution
        self.nimx = nimx
        self.nimy = nimy
        self.sub_samples = sub_samples
        if not iterable(width):
            width = (width, width, width) # front/back, left/right, top/bottom
        self.width = np.array([width[0], width[1], width[2]])
        self.center = center
        self.steady_north = steady_north
        self.expand_factor = expand_factor
        # This seems to be necessary for now.  Not sure what goes wrong when not true.
        if north_vector is not None: self.steady_north=True
        self.north_vector = north_vector
        self.normal_vector = normal_vector
        if fields is None: fields = ["Density"]
        self.fields = fields
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function
        self.log_fields = log_fields
        self.use_kd = use_kd
        self.l_max = l_max
        self.no_ghost = no_ghost
        self.preload = preload
        
        self.use_light = use_light
        self.light_dir = None
        self.light_rgba = None
        self.le = le
        self.re = re
        self.width[0]/=self.nimx
        self.width[1]/=self.nimy
        
        self.orienter = Orientation(normal_vector, north_vector=north_vector, steady_north=steady_north)
        self.rotation_vector = self.orienter.north_vector
        # self._setup_box_properties(width, center, self.orienter.unit_vectors)
        
        if self.no_ghost:
            mylog.info('Warning: no_ghost is currently True (default). This may lead to artifacts at grid boundaries.')
        self.tree_type = tree_type
        self.volume = volume

        # self.cameras = np.empty(self.nimx*self.nimy)

    def build_volume(self, volume, fields, log_fields, l_max, no_ghost, tree_type, le, re):
        if volume is None:
            if self.use_kd:
                volume = AMRKDTree(self.pf, l_max=l_max, fields=self.fields, 
                                   no_ghost=no_ghost, tree_type=tree_type, 
                                   log_fields=log_fields, le=le, re=re)
            else:
                volume = HomogenizedVolume(fields, pf=self.pf, log_fields=log_fields)
        else:
            self.use_kd = isinstance(volume, AMRKDTree)
        return volume

    def new_image(self):
        image = np.zeros((self.resolution[0], self.resolution[1], 4), dtype='float64', order='C')
        return image

    def _setup_box_properties(self, width, center, unit_vectors):
        owidth = deepcopy(width)
        self.width = width
        self.origin = self.center - 0.5*self.nimx*self.width[0]*self.orienter.unit_vectors[0] \
                                  - 0.5*self.nimy*self.width[1]*self.orienter.unit_vectors[1] \
                                  - 0.5*self.width[2]*self.orienter.unit_vectors[2]
        dx = self.width[0]
        dy = self.width[1]
        offi = (self.imi + 0.5)
        offj = (self.imj + 0.5)
        mylog.info("Mosaic offset: %f %f" % (offi,offj))
        global_center = self.center
        self.center = self.origin
        self.center += offi*dx*self.orienter.unit_vectors[0]
        self.center += offj*dy*self.orienter.unit_vectors[1]
        
        self.box_vectors = np.array([self.orienter.unit_vectors[0]*dx*self.nimx,
                                     self.orienter.unit_vectors[1]*dy*self.nimy,
                                     self.orienter.unit_vectors[2]*self.width[2]])
        self.back_center = self.center - 0.5*self.width[0]*self.orienter.unit_vectors[2]
        self.front_center = self.center + 0.5*self.width[0]*self.orienter.unit_vectors[2]
        self.center = global_center
        self.width = owidth

    def snapshot(self, fn = None, clip_ratio = None, double_check = False,
                 num_threads = 0):

        my_storage = {}
        offx,offy = np.meshgrid(range(self.nimx),range(self.nimy))
        offxy = zip(offx.ravel(), offy.ravel())

        for sto, xy in parallel_objects(offxy, self.procs_per_wg, storage = my_storage, 
                                        dynamic=True):
            self.volume = self.build_volume(self.volume, self.fields, self.log_fields, 
                                   self.l_max, self.no_ghost, 
                                   self.tree_type, self.le, self.re)
            self.initialize_source()

            self.imi, self.imj = xy
            mylog.debug('Working on: %i %i' % (self.imi, self.imj))
            self._setup_box_properties(self.width, self.center, self.orienter.unit_vectors)
            image = self.new_image()
            args = self.get_sampler_args(image)
            sampler = self.get_sampler(args)
            image = self._render(double_check, num_threads, image, sampler)
            sto.id = self.imj*self.nimx + self.imi
            sto.result = image
        image = self.reduce_images(my_storage)
        self.save_image(image, fn=fn, clip_ratio=clip_ratio)
        return image

    def reduce_images(self,im_dict):
        final_image = 0
        if self.comm.rank == 0:
            offx,offy = np.meshgrid(range(self.nimx),range(self.nimy))
            offxy = zip(offx.ravel(), offy.ravel())
            nx,ny = self.resolution
            final_image = np.empty((nx*self.nimx, ny*self.nimy, 4),
                        dtype='float64',order='C')
            for xy in offxy: 
                i, j = xy
                ind = j*self.nimx+i
                final_image[i*nx:(i+1)*nx, j*ny:(j+1)*ny,:] = im_dict[ind]
        return final_image

data_object_registry["mosaic_camera"] = MosaicCamera


class MosaicFisheyeCamera(Camera):
    r"""A fisheye lens camera, taking adantage of image plane decomposition
    for parallelism.

    The camera represents the eye of an observer, which will be used to
    generate ray-cast volume renderings of the domain. In this case, the
    rays are defined by a fisheye lens

    Parameters
    ----------
    center : array_like
        The current "center" of the observer, from which the rays will be
        cast
    radius : float
        The radial distance to cast to
    resolution : int
        The number of pixels in each direction.  Must be a single int.
    volume : `yt.extensions.volume_rendering.HomogenizedVolume`, optional
        The volume to ray cast through.  Can be specified for finer-grained
        control, but otherwise will be automatically generated.
    fields : list of fields, optional
        This is the list of fields we want to volume render; defaults to
        Density.
    log_fields : list of bool, optional
        Whether we should take the log of the fields before supplying them to
        the volume rendering mechanism.
    sub_samples : int, optional
        The number of samples to take inside every cell per ray.
    pf : `~yt.data_objects.api.StaticOutput`
        For now, this is a require parameter!  But in the future it will become
        optional.  This is the parameter file to volume render.
    l_max: int, optional
        Specifies the maximum level to be rendered.  Also
        specifies the maximum level used in the AMRKDTree
        construction.  Defaults to None (all levels), and only
        applies if use_kd=True.
    no_ghost: bool, optional
        Optimization option.  If True, homogenized bricks will
        extrapolate out from grid instead of interpolating from
        ghost zones that have to first be calculated.  This can
        lead to large speed improvements, but at a loss of
        accuracy/smoothness in resulting image.  The effects are
        less notable when the transfer function is smooth and
        broad. Default: False
    nimx: int, optional
        The number by which to decompose the image plane into in the x
        direction.  Must evenly divide the resolution.
    nimy: int, optional
        The number by which to decompose the image plane into in the y 
        direction.  Must evenly divide the resolution.
    procs_per_wg: int, optional
        The number of processors to use on each sub-image. Within each
        subplane, the volume will be decomposed using the AMRKDTree with
        procs_per_wg processors.  

    Notes
    -----
        The product of nimx*nimy*procs_per_wg must be equal to or less than
        the total number of mpi processes.  

        Unlike the non-Mosaic camera, this will only return each sub-image
        to the root processor of each sub-image workgroup in order to save
        memory.  To save the final image, one must then call
        MosaicFisheyeCamera.save_image('filename')

    Examples
    --------

    >>> from yt.mods import *
    
    >>> pf = load('DD1717')
    
    >>> N = 512 # Pixels (1024^2)
    >>> c = (pf.domain_right_edge + pf.domain_left_edge)/2. # Center
    >>> radius = (pf.domain_right_edge - pf.domain_left_edge)/2.
    >>> fov = 180.0
    
    >>> field='Density'
    >>> mi,ma = pf.h.all_data().quantities['Extrema']('Density')[0]
    >>> mi,ma = np.log10(mi), np.log10(ma)
    
    # You may want to comment out the above lines and manually set the min and max
    # of the log of the Density field. For example:
    # mi,ma = -30.5,-26.5
    
    # Another good place to center the camera is close to the maximum density.
    # v,c = pf.h.find_max('Density')
    # c -= 0.1*radius
    
   
    # Construct transfer function
    >>> tf = ColorTransferFunction((mi-1, ma+1),nbins=1024)
    
    # Sample transfer function with Nc gaussians.  Use col_bounds keyword to limit
    # the color range to the min and max values, rather than the transfer function
    # bounds.
    >>> Nc = 5
    >>> tf.add_layers(Nc,w=0.005, col_bounds = (mi,ma), alpha=np.logspace(-2,0,Nc),
    >>>         colormap='RdBu_r')
    >>> 
    # Create the camera object. Use the keyword: no_ghost=True if a lot of time is
    # spent creating vertex-centered data. In this case I'm running with 8
    # processors, and am splitting the image plane into 4 pieces and using 2
    # processors on each piece.
    >>> cam = MosaicFisheyeCamera(c, radius, fov, N,
    >>>         transfer_function = tf, 
    >>>         sub_samples = 5, 
    >>>         pf=pf, 
    >>>         nimx=2,nimy=2,procs_per_wg=2)
    
    # Take a snapshot
    >>> im = cam.snapshot()
    
    # Save the image
    >>> cam.save_image('fisheye_mosaic.png')

    """
    def __init__(self, center, radius, fov, resolution, focal_center=None,
                 transfer_function=None, fields=None,
                 sub_samples=5, log_fields=None, volume=None,
                 pf=None, l_max=None, no_ghost=False,nimx=1, nimy=1, procs_per_wg=None,
                 rotation=None):

        ParallelAnalysisInterface.__init__(self)
        self.image_decomp = self.comm.size>1
        if self.image_decomp:
            PP = ProcessorPool()
            npatches = nimy*nimx
            if procs_per_wg is None:
                if (PP.size % npatches):
                    raise RuntimeError("Cannot evenly divide %i procs to %i patches" % (PP.size,npatches))
                else:
                    procs_per_wg = PP.size / npatches
            if (PP.size != npatches*procs_per_wg):
               raise RuntimeError("You need %i processors to utilize %i procs per one patch in [%i,%i] grid" 
                     % (npatches*procs_per_wg,procs_per_wg,nimx,nimy))
 
            for j in range(nimy):
                for i in range(nimx):
                    PP.add_workgroup(size=procs_per_wg, name='%04i_%04i'%(i,j))
                    
            for wg in PP.workgroups:
                if self.comm.rank in wg.ranks:
                    my_wg = wg
            
            self.global_comm = self.comm
            self.comm = my_wg.comm
            self.wg = my_wg
            self.imi = int(self.wg.name[0:4])
            self.imj = int(self.wg.name[5:9])
            mylog.info('My new communicator has the name %s' % self.wg.name)
            self.nimx = nimx
            self.nimy = nimy
        else:
            self.imi = 0
            self.imj = 0
            self.nimx = 1
            self.nimy = 1
        if pf is not None: self.pf = pf
        
        if rotation is None: rotation = np.eye(3)
        self.rotation_matrix = rotation
        
        self.normal_vector = np.array([0.,0.,1])
        self.north_vector = np.array([1.,0.,0.])
        self.east_vector = np.array([0.,1.,0.])
        self.rotation_vector = self.north_vector

        if iterable(resolution):
            raise RuntimeError("Resolution must be a single int")
        self.resolution = resolution
        self.center = np.array(center, dtype='float64')
        self.focal_center = focal_center
        self.radius = radius
        self.fov = fov
        if transfer_function is None:
            transfer_function = ProjectionTransferFunction()
        self.transfer_function = transfer_function
        if fields is None: fields = ["Density"]
        self.fields = fields
        self.sub_samples = sub_samples
        self.log_fields = log_fields
        if volume is None:
            volume = AMRKDTree(self.pf, fields=self.fields, no_ghost=no_ghost,
                               log_fields=log_fields,l_max=l_max)
        self.volume = volume
        self.vp = None
        self.image = None 

    def get_vector_plane(self):
        if self.focal_center is not None:
            rvec =  np.array(self.focal_center) - np.array(self.center)
            rvec /= (rvec**2).sum()**0.5
            angle = np.arccos( (self.normal_vector*rvec).sum()/( (self.normal_vector**2).sum()**0.5 *
                (rvec**2).sum()**0.5))
            rot_vector = np.cross(rvec, self.normal_vector)
            rot_vector /= (rot_vector**2).sum()**0.5
            
            self.rotation_matrix = get_rotation_matrix(angle,rot_vector)
            self.normal_vector = np.dot(self.rotation_matrix,self.normal_vector)
            self.north_vector = np.dot(self.rotation_matrix,self.north_vector)
            self.east_vector = np.dot(self.rotation_matrix,self.east_vector)
        else:
            self.focal_center = self.center + self.radius*self.normal_vector  
        dist = ((self.focal_center - self.center)**2).sum()**0.5
        # We now follow figures 4-7 of:
        # http://paulbourke.net/miscellaneous/domefisheye/fisheye/
        # ...but all in Cython.
        
        self.vp = arr_fisheye_vectors(self.resolution, self.fov, self.nimx, 
                self.nimy, self.imi, self.imj)
        
        self.vp = rotate_vectors(self.vp, self.rotation_matrix)

        self.center = self.focal_center - dist*self.normal_vector
        self.vp *= self.radius
        nx, ny = self.vp.shape[0], self.vp.shape[1]
        self.vp.shape = (nx*ny,1,3)

    def snapshot(self):
        if self.vp is None:
            self.get_vector_plane()

        nx,ny = self.resolution/self.nimx, self.resolution/self.nimy
        image = np.zeros((nx*ny,1,3), dtype='float64', order='C')
        uv = np.ones(3, dtype='float64')
        positions = np.ones((nx*ny, 1, 3), dtype='float64') * self.center
        vector_plane = VectorPlane(positions, self.vp, self.center,
                        (0.0, 1.0, 0.0, 1.0), image, uv, uv)
        tfp = TransferFunctionProxy(self.transfer_function)
        tfp.ns = self.sub_samples
        self.volume.initialize_source()
        mylog.info("Rendering fisheye of %s^2", self.resolution)
        pbar = get_pbar("Ray casting",
                        (self.volume.brick_dimensions + 1).prod(axis=-1).sum())

        total_cells = 0
        for brick in self.volume.traverse(None, self.center, image):
            brick.cast_plane(tfp, vector_plane)
            total_cells += np.prod(brick.my_data[0].shape)
            pbar.update(total_cells)
        pbar.finish()
        image.shape = (nx, ny, 3)

        if self.image is not None:
            del self.image
        image = ImageArray(image,
                           info=self.get_information())
        self.image = image
        return image

    def save_image(self, fn, clip_ratio=None):
        if '.png' not in fn:
            fn = fn + '.png'
        
        try:
            image = self.image
        except:
            mylog.error('You must first take a snapshot')
            raise(UserWarning)
        
        image = self.image
        nx,ny = self.resolution/self.nimx, self.resolution/self.nimy
        if self.image_decomp:
            if self.comm.rank == 0:
                if self.global_comm.rank == 0:
                    final_image = np.empty((nx*self.nimx, 
                        ny*self.nimy, 3),
                        dtype='float64',order='C')
                    final_image[:nx, :ny, :] = image
                    for j in range(self.nimy):
                        for i in range(self.nimx):
                            if i==0 and j==0: continue
                            arr = self.global_comm.recv_array((self.wg.size)*(j*self.nimx + i), tag = (self.wg.size)*(j*self.nimx + i))

                            final_image[i*nx:(i+1)*nx, j*ny:(j+1)*ny,:] = arr
                            del arr
                    if clip_ratio is not None:
                        write_bitmap(final_image, fn, clip_ratio*final_image.std())
                    else:
                        write_bitmap(final_image, fn)
                else:
                    self.global_comm.send_array(image, 0, tag = self.global_comm.rank)
        else:
            if self.comm.rank == 0:
                if clip_ratio is not None:
                    write_bitmap(image, fn, clip_ratio*image.std())
                else:
                    write_bitmap(image, fn)
        return

    def rotate(self, theta, rot_vector=None, keep_focus=True):
        r"""Rotate by a given angle

        Rotate the view.  If `rot_vector` is None, rotation will occur
        around the `north_vector`.

        Parameters
        ----------
        theta : float, in radians
             Angle (in radians) by which to rotate the view.
        rot_vector  : array_like, optional
            Specify the rotation vector around which rotation will
            occur.  Defaults to None, which sets rotation around
            `north_vector`

        Examples
        --------

        >>> cam.rotate(np.pi/4)
        """
        if rot_vector is None:
            rot_vector = self.north_vector

        dist = ((self.focal_center - self.center)**2).sum()**0.5

        R = get_rotation_matrix(theta, rot_vector)

        self.vp = rotate_vectors(self.vp, R)
        self.normal_vector = np.dot(R,self.normal_vector)
        self.north_vector = np.dot(R,self.north_vector)
        self.east_vector = np.dot(R,self.east_vector)

        if keep_focus:
            self.center = self.focal_center - dist*self.normal_vector

    def rotation(self, theta, n_steps, rot_vector=None, keep_focus=True):
        r"""Loop over rotate, creating a rotation

        This will yield `n_steps` snapshots until the current view has been
        rotated by an angle `theta`

        Parameters
        ----------
        theta : float, in radians
            Angle (in radians) by which to rotate the view.
        n_steps : int
            The number of look_at snapshots to make.
        rot_vector  : array_like, optional
            Specify the rotation vector around which rotation will
            occur.  Defaults to None, which sets rotation around the
            original `north_vector`

        Examples
        --------

        >>> for i, snapshot in enumerate(cam.rotation(np.pi, 10)):
        ...     iw.write_bitmap(snapshot, 'rotation_%04i.png' % i)
        """

        dtheta = (1.0*theta)/n_steps
        for i in xrange(n_steps):
            self.rotate(dtheta, rot_vector=rot_vector, keep_focus=keep_focus)
            yield self.snapshot()

    def move_to(self,final,n_steps,exponential=False):
        r"""Loop over a look_at

        This will yield `n_steps` snapshots until the current view has been
        moved to a final center of `final`.

        Parameters
        ----------
        final : array_like
            The final center to move to after `n_steps`
        n_steps : int
            The number of look_at snapshots to make.
        exponential : boolean
            Specifies whether the move/zoom transition follows an
            exponential path toward the destination or linear

        Examples
        --------

        >>> for i, snapshot in enumerate(cam.move_to([0.2,0.3,0.6], 10)):
        ...     cam.save_image('move_%04i.png' % i)
        """
        if exponential:
            position_diff = (np.array(final)/self.center)*1.0
            dx = position_diff**(1.0/n_steps)
        else:
            dx = (np.array(final) - self.center)*1.0/n_steps
        for i in xrange(n_steps):
            if exponential:
                self.center *= dx
            else:
                self.center += dx
            yield self.snapshot()

def allsky_projection(pf, center, radius, nside, field, weight = None,
                      inner_radius = 10, rotation = None, source = None):
    r"""Project through a parameter file, through an allsky-method
    decomposition from HEALpix, and return the image plane.

    This function will accept the necessary items to integrate through a volume
    over 4pi and return the integrated field of view to the user.  Note that if
    a weight is supplied, it will multiply the pre-interpolated values
    together.

    Parameters
    ----------
    pf : `~yt.data_objects.api.StaticOutput`
        This is the parameter file to volume render.
    center : array_like
        The current "center" of the view port -- the focal point for the
        camera.
    radius : float or list of floats
        The radius to integrate out to of the image.
    nside : int
        The HEALpix degree.  The number of rays integrated is 12*(Nside**2)
        Must be a power of two!
    field : string
        The field to project through the volume
    weight : optional, default None
        If supplied, the field will be pre-multiplied by this, then divided by
        the integrated value of this field.  This returns an average rather
        than a sum.
    inner_radius : optional, float, defaults to 0.05
        The radius of the inner clipping plane, in units of the dx at the point
        at which the volume rendering is centered.  This avoids unphysical
        effects of nearby cells.
    rotation : optional, 3x3 array
        If supplied, the vectors will be rotated by this.  You can construct
        this by, for instance, calling np.array([v1,v2,v3]) where those are the
        three reference planes of an orthogonal frame (see ortho_find).
    source : data container, default None
        If this is supplied, this gives the data source from which the all sky
        projection pulls its data from.

    Returns
    -------
    image : array
        An ((Nside**2)*12,1,3) array of the final integrated values, in float64 form.

    Examples
    --------

    >>> image = allsky_projection(pf, [0.5, 0.5, 0.5], 1.0/pf['mpc'],
                      32, "Temperature", "Density")
    >>> plot_allsky_healpix(image, 32, "healpix.png")

    """
    # We manually modify the ProjectionTransferFunction to get it to work the
    # way we want, with a second field that's also passed through.
    print "Because of recent relicensing, we currently cannot provide"
    print "HEALpix functionality.  Please visit yt-users for more"
    print "information."
    raise NotImplementedError
    fields = [field]
    center = np.array(center, dtype='float64')
    if weight is not None:
        # This is a temporary field, which we will remove at the end.
        def _make_wf(f, w):
            def temp_weightfield(a, b):
                tr = b[f].astype("float64") * b[w]
                return tr
            return temp_weightfield
        pf.field_info.add_field("temp_weightfield",
            function=_make_wf(field, weight))
        fields = ["temp_weightfield", weight]
    nv = 12*nside**2
    image = np.zeros((nv,1,4), dtype='float64', order='C')
    vs = arr_pix2vec_nest(nside, np.arange(nv))
    vs.shape = (nv, 1, 3)
    if rotation is not None:
        vs2 = vs.copy()
        for i in range(3):
            vs[:,:,i] = (vs2 * rotation[:,i]).sum(axis=2)
    else:
        vs += 1e-8
    positions = np.ones((nv, 1, 3), dtype='float64', order='C') * center
    dx = min(g.dds.min() for g in pf.h.find_point(center)[0])
    positions += inner_radius * dx * vs
    vs *= radius
    uv = np.ones(3, dtype='float64')
    if source is not None:
        grids = source._grids
    else:
        grids = pf.h.sphere(center, radius)._grids
    sampler = ProjectionSampler(positions, vs, center, (0.0, 0.0, 0.0, 0.0),
                                image, uv, uv, np.zeros(3, dtype='float64'))
    pb = get_pbar("Sampling ", len(grids))
    for i,grid in enumerate(grids):
        if source is not None:
            data = [grid[field] * source._get_cut_mask(grid) * \
                grid.child_mask.astype('float64')
                for field in fields]
        else:
            data = [grid[field] * grid.child_mask.astype('float64')
                for field in fields]
        pg = PartitionedGrid(
            grid.id, data,
            grid.LeftEdge, grid.RightEdge,
            grid.ActiveDimensions.astype("int64"))
        grid.clear_data()
        sampler(pg)
        pb.update(i)
    pb.finish()
    image = sampler.aimage
    if weight is None:
        dl = radius * pf.units[pf.field_info[field].projection_conversion]
        image *= dl
    else:
        image[:,:,0] /= image[:,:,1]
        pf.field_info.pop("temp_weightfield")
        for g in pf.h.grids:
            if "temp_weightfield" in g.keys():
                del g["temp_weightfield"]
    return image[:,0,0]

def plot_allsky_healpix(image, nside, fn, label = "", rotation = None,
                        take_log = True, resolution=512, cmin=None, cmax=None):
    print "Because of recent relicensing, we currently cannot provide"
    print "HEALpix functionality.  Please visit yt-users for more"
    print "information."
    raise NotImplementedError
    import matplotlib.figure
    import matplotlib.backends.backend_agg
    if rotation is None: rotation = np.eye(3).astype("float64")

    img, count = pixelize_healpix(nside, image, resolution, resolution, rotation)

    fig = matplotlib.figure.Figure((10, 5))
    ax = fig.add_subplot(1,1,1,projection='aitoff')
    if take_log: func = np.log10
    else: func = lambda a: a
    implot = ax.imshow(func(img), extent=(-np.pi,np.pi,-np.pi/2,np.pi/2),
                       clip_on=False, aspect=0.5, vmin=cmin, vmax=cmax)
    cb = fig.colorbar(implot, orientation='horizontal')
    cb.set_label(label)
    ax.xaxis.set_ticks(())
    ax.yaxis.set_ticks(())
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    canvas.print_figure(fn)
    return img, count

class ProjectionCamera(Camera):
    def __init__(self, center, normal_vector, width, resolution,
            field, weight=None, volume=None, no_ghost = False, 
            le=None, re=None,
            north_vector=None, pf=None, interpolated=False):

        if not interpolated:
            volume = 1

        self.interpolated = interpolated
        self.field = field
        self.weight = weight
        self.resolution = resolution

        fields = [field]
        if self.weight is not None:
            # This is a temporary field, which we will remove at the end.
            def _make_wf(f, w):
                def temp_weightfield(a, b):
                    tr = b[f].astype("float64") * b[w]
                    return tr
                return temp_weightfield
            pf.field_info.add_field("temp_weightfield",
                function=_make_wf(self.field, self.weight))
            fields = ["temp_weightfield", self.weight]
        
        self.fields = fields
        self.log_fields = [False]*len(self.fields)
        Camera.__init__(self, center, normal_vector, width, resolution, None,
                fields = fields, pf=pf, volume=volume,
                log_fields=self.log_fields, 
                le=le, re=re, north_vector=north_vector,
                no_ghost=no_ghost)
        self.center = center

    def get_sampler(self, args):
        if self.interpolated:
            sampler = InterpolatedProjectionSampler(*args)
        else:
            sampler = ProjectionSampler(*args)
        return sampler

    def initialize_source(self):
        if self.interpolated:
            Camera.initialize_source(self)
        else:
            pass

    def get_sampler_args(self, image):
        rotp = np.concatenate([self.orienter.inv_mat.ravel('F'), self.back_center.ravel()])
        args = (rotp, self.box_vectors[2], self.back_center,
            (-self.width[0]/2., self.width[0]/2.,
             -self.width[1]/2., self.width[1]/2.),
            image, self.orienter.unit_vectors[0], self.orienter.unit_vectors[1],
                np.array(self.width, dtype='float64'), self.sub_samples)
        return args

    def finalize_image(self,image):
        pf = self.pf
        if self.weight is None:
            dl = self.width[2] * pf.units[pf.field_info[self.field].projection_conversion]
            image *= dl
        else:
            image[:,:,0] /= image[:,:,1]

        return image[:,:,0]


    def _render(self, double_check, num_threads, image, sampler):
        # Calculate the eight corners of the box
        # Back corners ...
        if self.interpolated:
            return Camera._render(self, double_check, num_threads, image,
                    sampler)
        pf = self.pf
        width = self.width[2]
        north_vector = self.orienter.unit_vectors[0]
        east_vector = self.orienter.unit_vectors[1]
        normal_vector = self.orienter.unit_vectors[2]
        fields = self.fields

        mi = pf.domain_right_edge.copy()
        ma = pf.domain_left_edge.copy()
        for off1 in [-1, 1]:
            for off2 in [-1, 1]:
                for off3 in [-1, 1]:
                    this_point = (self.center + width/2. * off1 * north_vector
                                         + width/2. * off2 * east_vector
                                         + width/2. * off3 * normal_vector)
                    np.minimum(mi, this_point, mi)
                    np.maximum(ma, this_point, ma)
        # Now we have a bounding box.
        grids = pf.h.region(self.center, mi, ma)._grids

        pb = get_pbar("Sampling ", len(grids))
        for i,grid in enumerate(grids):
            data = [(grid[field] * grid.child_mask).astype("float64")
                    for field in fields]
            pg = PartitionedGrid(
                grid.id, data,
                grid.LeftEdge, grid.RightEdge, grid.ActiveDimensions.astype("int64"))
            grid.clear_data()
            sampler(pg, num_threads = num_threads)
            pb.update(i)
        pb.finish()

        image = self.finalize_image(sampler.aimage)
        return image

    def save_image(self, image, fn=None, clip_ratio=None):
        if self.pf.field_info[self.field].take_log:
            im = np.log10(image)
        else:
            im = image
        if self.comm.rank == 0 and fn is not None:
            if clip_ratio is not None:
                write_image(im, fn)
            else:
                write_image(im, fn)

    def snapshot(self, fn = None, clip_ratio = None, double_check = False,
                 num_threads = 0):

        if num_threads is None:
            num_threads=get_num_threads()

        fields = [self.field]
        resolution = self.resolution

        image = self.new_image()

        args = self.get_sampler_args(image)

        sampler = self.get_sampler(args)

        self.initialize_source()

        image = ImageArray(self._render(double_check, num_threads, 
                                        image, sampler),
                           info=self.get_information())

        self.save_image(image, fn=fn, clip_ratio=clip_ratio)

        return image
    snapshot.__doc__ = Camera.snapshot.__doc__

data_object_registry["projection_camera"] = ProjectionCamera

def off_axis_projection(pf, center, normal_vector, width, resolution,
                        field, weight = None, 
                        volume = None, no_ghost = False, interpolated = False,
                        north_vector = None):
    r"""Project through a parameter file, off-axis, and return the image plane.

    This function will accept the necessary items to integrate through a volume
    at an arbitrary angle and return the integrated field of view to the user.
    Note that if a weight is supplied, it will multiply the pre-interpolated
    values together, then create cell-centered values, then interpolate within
    the cell to conduct the integration.

    Parameters
    ----------
    pf : `~yt.data_objects.api.StaticOutput`
        This is the parameter file to volume render.
    center : array_like
        The current 'center' of the view port -- the focal point for the
        camera.
    normal_vector : array_like
        The vector between the camera position and the center.
    width : float or list of floats
        The current width of the image.  If a single float, the volume is
        cubical, but if not, it is left/right, top/bottom, front/back
    resolution : int or list of ints
        The number of pixels in each direction.
    field : string
        The field to project through the volume
    weight : optional, default None
        If supplied, the field will be pre-multiplied by this, then divided by
        the integrated value of this field.  This returns an average rather
        than a sum.
    volume : `yt.extensions.volume_rendering.HomogenizedVolume`, optional
        The volume to ray cast through.  Can be specified for finer-grained
        control, but otherwise will be automatically generated.
    no_ghost: bool, optional
        Optimization option.  If True, homogenized bricks will
        extrapolate out from grid instead of interpolating from
        ghost zones that have to first be calculated.  This can
        lead to large speed improvements, but at a loss of
        accuracy/smoothness in resulting image.  The effects are
        less notable when the transfer function is smooth and
        broad. Default: True
    interpolated : optional, default False
        If True, the data is first interpolated to vertex-centered data, 
        then tri-linearly interpolated along the ray. Not suggested for 
        quantitative studies.

    Returns
    -------
    image : array
        An (N,N) array of the final integrated values, in float64 form.

    Examples
    --------

    >>> image = off_axis_projection(pf, [0.5, 0.5, 0.5], [0.2,0.3,0.4],
                      0.2, N, "Temperature", "Density")
    >>> write_image(np.log10(image), "offaxis.png")

    """
    projcam = ProjectionCamera(center, normal_vector, width, resolution,
                               field, weight=weight, pf=pf, volume=volume,
                               no_ghost=no_ghost, interpolated=interpolated, 
                               north_vector=north_vector)
    image = projcam.snapshot()
    if weight is not None:
        pf.field_info.pop("temp_weightfield")
    del projcam
    return image[:,:]

