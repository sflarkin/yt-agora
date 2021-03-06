"""
This is the preliminary interface to VTK.  Note that as of VTK 5.2, it still
requires a patchset prepared here:
http://yt.enzotools.org/files/vtk_composite_data.zip

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2007-2009 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from enthought.tvtk.tools import ivtk
from enthought.tvtk.api import tvtk 
from enthought.traits.api import \
    Float, HasTraits, Instance, Range, Any, Delegate, Tuple, File, Int, Str, \
    CArray, List, Button, Bool, Property, cached_property
from enthought.traits.ui.api import View, Item, HGroup, VGroup, TableEditor, \
    Handler, Controller, RangeEditor, EnumEditor, InstanceEditor
from enthought.traits.ui.menu import \
    Menu, Action, Separator, OKCancelButtons, OKButton
from enthought.traits.ui.table_column import ObjectColumn
from enthought.tvtk.pyface.api import DecoratedScene

import enthought.pyface.api as pyface

#from yt.reason import *
import sys
import numpy as na
import time, pickle, os, os.path
import yt.lagos as lagos
from yt.funcs import *
from yt.logger import ravenLogger as mylog
from yt.extensions.HierarchySubset import \
        ExtractedHierarchy, ExtractedParameterFile

#from enthought.tvtk.pyface.ui.wx.wxVTKRenderWindowInteractor \
     #import wxVTKRenderWindowInteractor

from enthought.mayavi.core.lut_manager import LUTManager

#wxVTKRenderWindowInteractor.USE_STEREO = 1

class TVTKMapperWidget(HasTraits):
    alpha = Float(1.0)
    post_call = Any
    lut_manager = Instance(LUTManager)

    def _alpha_changed(self, old, new):
        self.lut_manager.lut.alpha_range = (new, new)
        self.post_call()

class MappingPlane(TVTKMapperWidget):
    plane = Instance(tvtk.Plane)
    _coord_redit = editor=RangeEditor(format="%0.2e",
                              low_name='vmin', high_name='vmax',
                              auto_set=False, enter_set=True)
    auto_set = Bool(False)
    traits_view = View(Item('coord', editor=_coord_redit),
                       Item('auto_set'),
                       Item('alpha', editor=RangeEditor(
                              low=0.0, high=1.0,
                              enter_set=True, auto_set=False)),
                       Item('lut_manager', show_label=False,
                            editor=InstanceEditor(), style='custom'))
    vmin = Float
    vmax = Float

    def _auto_set_changed(self, old, new):
        if new is True:
            self._coord_redit.auto_set = True
            self._coord_redit.enter_set = False
        else:
            self._coord_redit.auto_set = False
            self._coord_redit.enter_set = True

    def __init__(self, vmin, vmax, vdefault, **traits):
        HasTraits.__init__(self, **traits)
        self.vmin = vmin
        self.vmax = vmax
        trait = Range(float(vmin), float(vmax), value=vdefault)
        self.add_trait("coord", trait)
        self.coord = vdefault

    def _coord_changed(self, old, new):
        orig = self.plane.origin[:]
        orig[self.axis] = new
        self.plane.origin = orig
        self.post_call()

class MappingMarchingCubes(TVTKMapperWidget):
    operator = Instance(tvtk.MarchingCubes)
    mapper = Instance(tvtk.HierarchicalPolyDataMapper)
    vmin = Float
    vmax = Float
    auto_set = Bool(False)
    _val_redit = RangeEditor(format="%0.2f",
                             low_name='vmin', high_name='vmax',
                             auto_set=False, enter_set=True)
    traits_view = View(Item('value', editor=_val_redit),
                       Item('auto_set'),
                       Item('alpha', editor=RangeEditor(
                            low=0.0, high=1.0,
                            enter_set=True, auto_set=False,)),
                       Item('lut_manager', show_label=False,
                            editor=InstanceEditor(), style='custom'))

    def __init__(self, vmin, vmax, vdefault, **traits):
        HasTraits.__init__(self, **traits)
        self.vmin = vmin
        self.vmax = vmax
        trait = Range(float(vmin), float(vmax), value=vdefault)
        self.add_trait("value", trait)
        self.value = vdefault

    def _auto_set_changed(self, old, new):
        if new is True:
            self._val_redit.auto_set = True
            self._val_redit.enter_set = False
        else:
            self._val_redit.auto_set = False
            self._val_redit.enter_set = True

    def _value_changed(self, old, new):
        self.operator.set_value(0, new)
        self.post_call()

class MappingIsoContour(MappingMarchingCubes):
    operator = Instance(tvtk.ContourFilter)

class CameraPosition(HasTraits):
    position = CArray(shape=(3,), dtype='float64')
    focal_point = CArray(shape=(3,), dtype='float64')
    view_up = CArray(shape=(3,), dtype='float64')
    clipping_range = CArray(shape=(2,), dtype='float64')
    distance = Float
    num_steps = Int(10)
    orientation_wxyz = CArray(shape=(4,), dtype='float64')

class CameraControl(HasTraits):
    # Traits
    positions = List(CameraPosition)
    yt_scene = Instance('YTScene')
    center = Delegate('yt_scene')
    scene = Delegate('yt_scene')
    camera = Instance(tvtk.OpenGLCamera)
    reset_position = Instance(CameraPosition)
    fps = Float(25.0)
    export_filename = 'frames'
    periodic = Bool

    # UI elements
    snapshot = Button()
    play = Button()
    export_frames = Button()
    reset_path = Button()
    recenter = Button()
    save_path = Button()
    load_path = Button()
    export_path = Button()

    table_def = TableEditor(
        columns = [ ObjectColumn(name='position'),
                    ObjectColumn(name='focal_point'),
                    ObjectColumn(name='view_up'),
                    ObjectColumn(name='clipping_range'),
                    ObjectColumn(name='num_steps') ],
        reorderable=True, deletable=True,
        sortable=True, sort_model=True,
        show_toolbar=True,
        selection_mode='row',
        selected = 'reset_position'
                )

    default_view = View(
                VGroup(
                  HGroup(
                    Item('camera', show_label=False),
                    Item('recenter', show_label=False),
                    label='Camera'),
                  HGroup(
                    Item('snapshot', show_label=False),
                    Item('play', show_label=False),
                    Item('export_frames',show_label=False),
                    Item('reset_path', show_label=False),
                    Item('save_path', show_label=False),
                    Item('load_path', show_label=False),
                    Item('export_path', show_label=False),
                    Item('export_filename'),
                    Item('periodic'),
                    Item('fps'),
                    label='Playback'),
                  VGroup(
                    Item('positions', show_label=False,
                        editor=table_def),
                    label='Camera Path'),
                 ),
                resizable=True, title="Camera Path Editor",
                       )

    def _reset_position_changed(self, old, new):
        if new is None: return
        cam = self.scene.camera
        cam.position = new.position
        cam.focal_point = new.focal_point
        cam.view_up = new.view_up
        cam.clipping_range = new.clipping_range
        self.scene.render()

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)

    def take_snapshot(self):
        cam = self.scene.camera
        self.positions.append(CameraPosition(
                position=cam.position,
                focal_point=cam.focal_point,
                view_up=cam.view_up,
                clipping_range=cam.clipping_range,
                distance=cam.distance,
                orientation_wxyz=cam.orientation_wxyz))

    def _export_path_fired(self): 
        dlg = pyface.FileDialog(
            action='save as',
            wildcard="*.cpath",
        )
        if dlg.open() == pyface.OK:
            print "Saving:", dlg.path
            self.export_camera_path(dlg.path)

    def export_camera_path(self, fn):
        to_dump = dict(positions=[], focal_points=[],
                       view_ups=[], clipping_ranges=[],
                       distances=[], orientation_wxyzs=[])
        def _write(cam):
            to_dump['positions'].append(cam.position)
            to_dump['focal_points'].append(cam.focal_point)
            to_dump['view_ups'].append(cam.view_up)
            to_dump['clipping_ranges'].append(cam.clipping_range)
            to_dump['distances'].append(cam.distance)
            to_dump['orientation_wxyzs'].append(cam.orientation_wxyz)
        self.step_through(0.0, callback=_write)
        pickle.dump(to_dump, open(fn, "wb"))

    def _save_path_fired(self): 
        dlg = pyface.FileDialog(
            action='save as',
            wildcard="*.cpath",
        )
        if dlg.open() == pyface.OK:
            print "Saving:", dlg.path
            self.dump_camera_path(dlg.path)

    def dump_camera_path(self, fn):
        to_dump = dict(positions=[], focal_points=[],
                       view_ups=[], clipping_ranges=[],
                       distances=[], orientation_wxyzs=[],
                       num_stepss=[])
        for p in self.positions:
            to_dump['positions'].append(p.position)
            to_dump['focal_points'].append(p.focal_point)
            to_dump['view_ups'].append(p.view_up)
            to_dump['clipping_ranges'].append(p.clipping_range)
            to_dump['distances'].append(p.distance)
            to_dump['num_stepss'].append(p.num_steps) # stupid s
            to_dump['orientation_wxyzs'].append(p.orientation_wxyz)
        pickle.dump(to_dump, open(fn, "wb"))

    def _load_path_fired(self):
        dlg = pyface.FileDialog(
            action='open',
            wildcard="*.cpath",
        )
        if dlg.open() == pyface.OK:
            print "Loading:", dlg.path
            self.load_camera_path(dlg.path)

    def load_camera_path(self, fn):
        to_use = pickle.load(open(fn, "rb"))
        self.positions = []
        for i in range(len(to_use['positions'])):
            dd = {}
            for kw in to_use:
                # Strip the s
                dd[kw[:-1]] = to_use[kw][i]
            self.positions.append(
                CameraPosition(**dd))

    def _recenter_fired(self):
        self.camera.focal_point = self.center
        self.scene.render()

    def _snapshot_fired(self):
        self.take_snapshot()

    def _play_fired(self):
        self.step_through()

    def _export_frames_fired(self):
        self.step_through(save_frames=True)

    def _reset_path_fired(self):
        self.positions = []

    def step_through(self, pause = 1.0, callback=None, save_frames=False):
        cam = self.scene.camera
        frame_counter=0
        if self.periodic:
            cyclic_pos = self.positions + [self.positions[0]]
        else:
            cyclic_pos = self.positions
        for i in range(len(cyclic_pos)-1):
            pos1 = cyclic_pos[i]
            pos2 = cyclic_pos[i+1]
            r = pos1.num_steps
            for p in range(pos1.num_steps):
                po = _interpolate(pos1.position, pos2.position, p, r)
                fp = _interpolate(pos1.focal_point, pos2.focal_point, p, r)
                vu = _interpolate(pos1.view_up, pos2.view_up, p, r)
                cr = _interpolate(pos1.clipping_range, pos2.clipping_range, p, r)
                _set_cpos(cam, po, fp, vu, cr)
                self.scene.render()
                if callback is not None: callback(cam)
                if save_frames:
                    self.scene.save("%s_%0.5d.png" % (self.export_filename,frame_counter))
                else:
                    time.sleep(pause * 1.0/self.fps)
                frame_counter += 1

def _interpolate(q1, q2, p, r):
    return q1 + p*(q2 - q1)/float(r)

def _set_cpos(cam, po, fp, vu, cr):
    cam.position = po
    cam.focal_point = fp
    cam.view_up = vu
    cam.clipping_range = cr

class HierarchyImporter(HasTraits):
    pf = Any
    min_grid_level = Int(0)
    max_level = Int(1)
    number_of_levels = Range(0, 13)
    max_import_levels = Property(depends_on='min_grid_level')
    field = Str("Density")
    field_list = List
    center_on_max = Bool(True)
    center = CArray(shape = (3,), dtype = 'float64')
    cache = Bool(True)
    smoothed = Bool(True)
    show_grids = Bool(True)

    def _field_list_default(self):
        fl = self.pf.h.field_list
        df = self.pf.h.derived_field_list
        fl.sort(); df.sort()
        return fl + df
    
    default_view = View(Item('min_grid_level',
                              editor=RangeEditor(low=0,
                                                 high_name='max_level')),
                        Item('number_of_levels', 
                              editor=RangeEditor(low=1,
                                                 high_name='max_import_levels')),
                        Item('field', editor=EnumEditor(name='field_list')),
                        Item('center_on_max'),
                        Item('center', enabled_when='not object.center_on_max'),
                        Item('smoothed'),
                        Item('cache', label='Pre-load data'),
                        Item('show_grids'),
                        buttons=OKCancelButtons)

    def _center_default(self):
        return [0.5,0.5,0.5]

    @cached_property
    def _get_max_import_levels(self):
        return min(13, self.pf.h.max_level - self.min_grid_level + 1)

class HierarchyImportHandler(Controller):
    importer = Instance(HierarchyImporter)
    

    def close(self, info, is_ok):
        if is_ok: 
            yt_scene = YTScene(
                importer=self.importer)
        super(Controller, self).close(info, True)
        return


class YTScene(HasTraits):

    # Traits
    importer = Instance(HierarchyImporter)
    pf = Delegate("importer")
    min_grid_level = Delegate("importer")
    number_of_levels = Delegate("importer")
    field = Delegate("importer")
    center = CArray(shape = (3,), dtype = 'float64')
    center_on_max = Delegate("importer")
    smoothed = Delegate("importer")
    cache = Delegate("importer")
    show_grids = Delegate("importer")

    camera_path = Instance(CameraControl)
    #window = Instance(ivtk.IVTKWithCrustAndBrowser)
    #python_shell = Delegate('window')
    #scene = Delegate('window')
    scene = Instance(HasTraits)
    operators = List(HasTraits)

    # State variables
    _grid_boundaries_actor = None

    # Views
    def _window_default(self):
        # Should experiment with passing in a pipeline browser
        # that has two root objects -- one for TVTKBases, i.e. the render
        # window, and one that accepts our objects
        return ivtk.IVTKWithCrustAndBrowser(size=(800,600), stereo=1)

    def _camera_path_default(self):
        return CameraControl(yt_scene=self, camera=self.scene.camera)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        max_level = min(self.pf.h.max_level,
                        self.min_grid_level + self.number_of_levels - 1)
        self.extracted_pf = ExtractedParameterFile(self.pf,
                             self.min_grid_level, max_level, offset=None)
        self.extracted_hierarchy = self.extracted_pf.h
        self._hdata_set = tvtk.HierarchicalBoxDataSet()
        self._ugs = []
        self._grids = []
        self._min_val = 1e60
        self._max_val = -1e60
        gid = 0
        if self.cache:
            for grid_set in self.extracted_hierarchy.get_levels():
                for grid in grid_set:
                    grid[self.field]
        for l, grid_set in enumerate(self.extracted_hierarchy.get_levels()):
            gid = self._add_level(grid_set, l, gid)
        if self.show_grids:
            self.toggle_grid_boundaries()
            
    def _center_default(self):
        return self.extracted_hierarchy._convert_coords(
                [0.5, 0.5, 0.5])

    def do_center_on_max(self):
        self.center = self.extracted_hierarchy._convert_coords(
            self.pf.h.find_max("Density")[1])
        self.scene.camera.focal_point = self.center

    def _add_level(self, grid_set, level, gid):
        for grid in grid_set:
            self._hdata_set.set_refinement_ratio(level, 2)
            gid = self._add_grid(grid, gid, level)
        return gid

    def _add_grid(self, grid, gid, level=0):
        mylog.debug("Adding grid %s on level %s (%s)",
                    grid.id, level, grid.Level)
        if grid in self._grids: return
        self._grids.append(grid)

        scalars = grid.get_vertex_centered_data(self.field, smoothed=self.smoothed)

        left_index = grid.get_global_startindex()
        origin = grid.LeftEdge
        dds = grid.dds
        right_index = left_index + scalars.shape - 1
        ug = tvtk.UniformGrid(origin=origin, spacing=dds,
                              dimensions=grid.ActiveDimensions+1)
        if self.field not in self.pf.field_info or \
            self.pf.field_info[self.field].take_log:
            scalars = na.log10(scalars)
        ug.point_data.scalars = scalars.transpose().ravel()
        ug.point_data.scalars.name = self.field
        if grid.Level != self.min_grid_level + self.number_of_levels - 1:
            ug.cell_visibility_array = grid.child_mask.transpose().ravel()
        else:
            ug.cell_visibility_array = na.ones(
                    grid.ActiveDimensions, dtype='int').ravel()
        self._ugs.append((grid,ug))
        self._hdata_set.set_data_set(level, gid, left_index, right_index, ug)

        self._min_val = min(self._min_val, scalars.min())
        self._max_val = max(self._max_val, scalars.max())

        gid += 1
        return gid

    def _add_data_to_ug(self, field):
        for g, ug in self._ugs:
            scalars_temp = grid.get_vertex_centered_data(field, smoothed=self.smoothed)
            ii = ug.point_data.add_array(scalars_temp.transpose().ravel())
            ug.point_data.get_array(ii).name = field

    def zoom(self, dist, unit='1'):
        vec = self.scene.camera.focal_point - \
              self.scene.camera.position
        self.scene.camera.position += \
            vec * dist/self._grids[0].pf[unit]
        self.scene.render()

    def toggle_grid_boundaries(self):
        if self._grid_boundaries_actor is None:
            # We don't need to track this stuff right now.
            ocf = tvtk.OutlineCornerFilter(
                    executive=tvtk.CompositeDataPipeline(),
                    corner_factor = 0.5)
            ocf.input = self._hdata_set
            ocm = tvtk.HierarchicalPolyDataMapper(
                input_connection = ocf.output_port)
            self._grid_boundaries_actor = tvtk.Actor(mapper = ocm)
            self.scene.add_actor(self._grid_boundaries_actor)
        else:
            self._grid_boundaries_actor.visibility = \
            (not self._grid_boundaries_actor.visibility)

    def _add_sphere(self, origin=(0.0,0.0,0.0), normal=(0,1,0)):
        sphere = tvtk.Sphere(center=origin, radius=0.25)
        cutter = tvtk.Cutter(executive = tvtk.CompositeDataPipeline(),
                             cut_function = sphere)
        cutter.input = self._hdata_set
        lut_manager = LUTManager(data_name=self.field, scene=self.scene)
        smap = tvtk.HierarchicalPolyDataMapper(
                        scalar_range=(self._min_val, self._max_val),
                        lookup_table=lut_manager.lut,
                        input_connection = cutter.output_port)
        sactor = tvtk.Actor(mapper=smap)
        self.scene.add_actors(sactor)
        return sphere, lut_manager

    def _add_plane(self, origin=(0.0,0.0,0.0), normal=(0,1,0)):
        plane = tvtk.Plane(origin=origin, normal=normal)
        cutter = tvtk.Cutter(executive = tvtk.CompositeDataPipeline(),
                             cut_function = plane)
        cutter.input = self._hdata_set
        lut_manager = LUTManager(data_name=self.field, scene=self.scene)
        smap = tvtk.HierarchicalPolyDataMapper(
                        scalar_range=(self._min_val, self._max_val),
                        lookup_table=lut_manager.lut,
                        input_connection = cutter.output_port)
        sactor = tvtk.Actor(mapper=smap)
        self.scene.add_actors(sactor)
        return plane, lut_manager

    def add_plane(self, origin=(0.0,0.0,0.0), normal=(0,1,0)):
        self.operators.append(self._add_plane(origin, normal))
        return self.operators[-1]

    def _add_axis_plane(self, axis):
        normal = [0,0,0]
        normal[axis] = 1
        np, lut_manager = self._add_plane(self.center, normal=normal)
        LE = self.extracted_hierarchy.min_left_edge
        RE = self.extracted_hierarchy.max_right_edge
        self.operators.append(MappingPlane(
                vmin=LE[axis], vmax=RE[axis],
                vdefault = self.center[axis],
                post_call = self.scene.render,
                plane = np, axis=axis, coord=0.0,
                lut_manager = lut_manager,
                scene=self.scene))

    def add_x_plane(self):
        self._add_axis_plane(0)
        return self.operators[-1]

    def add_y_plane(self):
        self._add_axis_plane(1)
        return self.operators[-1]

    def add_z_plane(self):
        self._add_axis_plane(2)
        return self.operators[-1]

    def add_contour(self, val=None):
        if val is None: 
            if self._min_val != self._min_val:
                self._min_val = 1.0
            val = (self._max_val+self._min_val) * 0.5
        cubes = tvtk.MarchingCubes(
                    executive = tvtk.CompositeDataPipeline())
        cubes.input = self._hdata_set
        cubes.set_value(0, val)
        lut_manager = LUTManager(data_name=self.field, scene=self.scene)
        cube_mapper = tvtk.HierarchicalPolyDataMapper(
                                input_connection = cubes.output_port,
                                lookup_table=lut_manager.lut)
        cube_mapper.color_mode = 'map_scalars'
        cube_mapper.scalar_range = (self._min_val, self._max_val)
        cube_actor = tvtk.Actor(mapper=cube_mapper)
        self.scene.add_actors(cube_actor)
        self.operators.append(MappingMarchingCubes(operator=cubes,
                    vmin=self._min_val, vmax=self._max_val,
                    vdefault=val,
                    mapper = cube_mapper,
                    post_call = self.scene.render,
                    lut_manager = lut_manager,
                    scene=self.scene))
        return self.operators[-1]

    def add_isocontour(self, val=None):
        if val is None: val = (self._max_val+self._min_val) * 0.5
        isocontour = tvtk.ContourFilter(
                    executive = tvtk.CompositeDataPipeline())
        isocontour.input = self._hdata_set
        isocontour.generate_values(1, (val, val))
        lut_manager = LUTManager(data_name=self.field, scene=self.scene)
        isocontour_normals = tvtk.PolyDataNormals(
            executive=tvtk.CompositeDataPipeline())
        isocontour_normals.input_connection = isocontour.output_port
        iso_mapper = tvtk.HierarchicalPolyDataMapper(
                                input_connection = isocontour_normals.output_port,
                                lookup_table=lut_manager.lut)
        iso_mapper.scalar_range = (self._min_val, self._max_val)
        iso_actor = tvtk.Actor(mapper=iso_mapper)
        self.scene.add_actors(iso_actor)
        self.operators.append(MappingIsoContour(operator=isocontour,
                    vmin=self._min_val, vmax=self._max_val,
                    vdefault=val,
                    mapper = iso_mapper,
                    post_call = self.scene.render,
                    lut_manager = lut_manager,
                    scene=self.scene))
        return self.operators[-1]

def get_all_parents(grid):
    parents = []
    if len(grid.Parents) == 0: return grid
    for parent in grid.Parents: parents.append(get_all_parents(parent))
    return list(set(parents))

def run_vtk():
    import yt.lagos as lagos

    gui = pyface.GUI()
    importer = HierarchyImporter()
    importer.edit_traits(handler = HierarchyImportHandler(
            importer = importer))
    #ehds.edit_traits()
    gui.start_event_loop()


if __name__=="__main__":
    print "This code probably won't work.  But if you want to give it a try,"
    print "you need:"
    print
    print "VTK (CVS)"
    print "Mayavi2 (from Enthought)"
    print
    print "If you have 'em, give it a try!"
    print
    run_vtk()
