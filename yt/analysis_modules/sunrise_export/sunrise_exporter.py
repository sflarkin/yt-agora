"""
Code to export from yt to Sunrise



"""
#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import time
import numpy as np
import pyfits
import yt
from yt.funcs import *
import yt.utilities.lib.api as amr_utils
#import pdb


def export_to_sunrise(ds, fn, star_particle_type, fc, fwidth, nocts_wide=None,
        debug=False,ad=None,**kwargs):
    r"""Convert the contents of a dataset to a FITS file format that Sunrise
    understands.

    This function will accept a dataset, and from that dataset
    construct a depth-first octree containing all of the data in the parameter
    file.  This octree will be written to a FITS file.  It will probably be
    quite big, so use this function with caution!  Sunrise is a tool for
    generating synthetic spectra, available at
    http://sunrise.googlecode.com/ .

    Parameters
    ----------
    ds : `Dataset`
       The dataset to convert.
    fn : string
       The filename of the output FITS file.
    fc : array
       The center of the extraction region
    fwidth  : array  
       Ensure this radius around the center is enclosed
       Array format is (nx,ny,nz) where each element is floating point
       in unitary position units where 0 is leftmost edge and 1
       the rightmost. 

    Notes
    -----

    Note that the process of generating simulated images from Sunrise will
    require substantial user input; see the Sunrise wiki at
    http://sunrise.googlecode.com/ for more information.

    """
    fc = fc.in_units('code_length').value
    fwidth = fwidth.in_units('code_length').value
    Nocts_root = ds.domain_dimensions/2
    
    #we must round the dle,dre to the nearest root grid cells
    ile,ire,super_level,nocts_wide= \
            round_nocts_wide(Nocts_root,fc-fwidth,fc+fwidth,nwide=nocts_wide)

    assert np.all((ile-ire)==(ile-ire)[0])
    print "rounding specified region:"
    print "from [%1.5f %1.5f %1.5f]-[%1.5f %1.5f %1.5f]"%(tuple(fc-fwidth)+tuple(fc+fwidth))
    print "to (integer)   [%07i %07i %07i]-[%07i %07i %07i]"%(tuple(ile)+tuple(ire))
    assert(len(np.unique(ds.domain_width)) == 1)
    domain_width = ds.domain_width[0]
    fle,fre = ile*domain_width/Nocts_root, ire*domain_width/Nocts_root
    print "to (float)  [%1.5f %1.5f %1.5f]-[%1.5f %1.5f %1.5f]"%(tuple(fle)+tuple(fre))

    #Create a list of the star particle properties in PARTICLE_DATA
    #Include ID, parent-ID, position, velocity, creation_mass, 
    #formation_time, mass, age_m, age_l, metallicity, L_bol
    particle_data,nstars = prepare_star_particles(ds,star_particle_type,fle=fle,fre=fre,
                                                  ad=ad,**kwargs)

    #Create the refinement hilbert octree in GRIDSTRUCTURE
    #For every leaf (not-refined) cell we have a column n GRIDDATA
    #Include mass_gas, mass_metals, gas_temp_m, gas_teff_m, cell_volume, SFR
    #since the octree always starts with one cell, an our 0-level mesh
    #may have many cells, we must create the octree region sitting 
    #ontop of the first mesh by providing a negative level
    output, refined, nrefined, nleafs = prepare_octree(ds,ile,fle=fle,fre=fre,
                                                       ad=ad,start_level=super_level,
                                                       debug=debug)

    create_fits_file(ds,fn,output,refined,particle_data,fle,fre)

    return fle, fre, ile, ire, nrefined, nleafs, nstars 


def prepare_star_particles(ds,star_type,pos=None,vel=None, age=None,
                          creation_time=None,initial_mass=None,
                          current_mass=None,metallicity=None,
                          radius = None,
                          fle=[0.,0.,0.],fre=[1.,1.,1.],
                          ad=None):
    if ad is None:
        ad = ds.all_data()
    nump = ad[star_type,"particle_ones"]
    assert nump.sum()>1 #make sure we select more than a single particle
    
    if pos is None:
        pos = yt.YTArray([ad[star_type,"particle_position_%s" % ax]
                        for ax in 'xyz']).transpose()

    idx = np.all(pos > fle, axis=1) & np.all(pos < fre, axis=1)
    assert np.sum(idx)>0 #make sure we select more than a single particle
    
    pos = pos[idx].in_units('kpc') #unitary units -> kpc
 
    if creation_time is None:
        formation_time = ad[star_type,"particle_creation_time"][idx].in_units('yr')

    if age is None:
        age = (ds.current_time - formation_time).in_units('yr')

    if vel is None:
        vel = yt.YTArray([ad[star_type,"particle_velocity_%s" % ax]
                        for ax in 'xyz']).transpose()
        # Velocity is cm/s, we want it to be kpc/yr
        #vel *= (ds["kpc"]/ds["cm"]) / (365*24*3600.)
        vel = vel[idx].in_units('kpc/yr')
    
    if initial_mass is None:
        #in solar masses
        initial_mass = ad[star_type,"particle_mass_initial"][idx].in_units('Msun')
    
    if current_mass is None:
        #in solar masses
        current_mass = ad[star_type,"particle_mass"][idx].in_units('Msun')
    
    if metallicity is None:
        #this should be in dimensionless units, metals mass / particle mass
        metallicity = ad[star_type,"particle_metallicity"][idx]
    
    if radius is None:
        radius = ds.arr(metallicity*0.0 + 10.0/1000.0, 'kpc') #10pc radius
    
    #create every column
    col_list = []
    col_list.append(pyfits.Column("ID", format="J", array=np.arange(current_mass.size).astype('int32')))
    col_list.append(pyfits.Column("parent_ID", format="J", array=np.arange(current_mass.size).astype('int32')))
    col_list.append(pyfits.Column("position", format="3D", array=pos, unit="kpc"))
    col_list.append(pyfits.Column("velocity", format="3D", array=vel, unit="kpc/yr"))
    col_list.append(pyfits.Column("creation_mass", format="D", array=initial_mass, unit="Msun"))
    col_list.append(pyfits.Column("formation_time", format="D", array=formation_time, unit="yr"))
    col_list.append(pyfits.Column("radius", format="D", array=radius, unit="kpc"))
    col_list.append(pyfits.Column("mass", format="D", array=current_mass, unit="Msun"))
    col_list.append(pyfits.Column("age", format="D", array=age,unit='yr'))
    #For particles, Sunrise takes 
    #the dimensionless metallicity, not the mass of the metals
    col_list.append(pyfits.Column("metallicity", format="D",
        array=metallicity,unit="dimensionless")) 
    
    #make the table
    cols = pyfits.ColDefs(col_list)
    pd_table = pyfits.BinTableHDU.from_columns(cols)
#    pd_table = pyfits.new_table(cols)
    pd_table.name = "PARTICLEDATA"
    
    #make sure we have nonzero particle number
    assert pd_table.data.shape[0]>0
    return pd_table, np.sum(idx)


def prepare_octree(ds, ile, fle=[0.,0.,0.], fre=[1.,1.,1.],
                   ad=None, start_level=0, debug=True):
    if ad is None:
        #we keep passing ad around to not regenerate the data all the time
        ad = ds.all_data()

    def _MetalMass(field, data):
        return (data['metal_ia_density']*data['cell_volume']).in_units('Msun')
    ad.ds.add_field('MetalMassMsun', function=_MetalMass, units='Msun')
 
    def _TempTimesMass(field, data):
        te = data['thermal_energy']
        hd = data['H_nuclei_density']
        temp = (2.0*te/(3.0*hd*yt.physical_constants.kb)).in_units('K')
        return temp*data["cell_mass"].in_units('Msun')
    ad.ds.add_field('TemperatureTimesCellMassMsun', function=_TempTimesMass, units='K*Msun')

    def _cellMassMsun(field, data):
        return data["cell_mass"].in_units('Msun')
    ad.ds.add_field('CellMassMsun', function=_cellMassMsun, units='Msun')

    def _cellVolumeKpc(field, data):
        return data["cell_volume"].in_units('kpc**3')
    ad.ds.add_field('CellVolumeKpc', function=_cellVolumeKpc, units='kpc**3')

    fields = ["CellMassMsun","TemperatureTimesCellMassMsun","MetalMassMsun","CellVolumeKpc"]

    #gather the field data from octs
    print "Retrieving field data"
    field_data = [] 
    for fi,f in enumerate(fields):
        field_data = ad[f]
    del field_data

    #Initialize dicitionary with arrays containig the needed
    #properites of all octs
    octs_dic = {}
    total_octs = ad.index.total_octs
    LeftEdge =  np.empty((total_octs,3), dtype='float64')
    dx = np.empty(total_octs, dtype='float64')
    Level = np.empty(total_octs, dtype='int32')
    Fields = np.empty((total_octs, len(fields), 2, 2, 2), dtype='float64')
    octs_dic['LeftEdge'], octs_dic['dx'], octs_dic['Level'], octs_dic['Fields'] = \
        amr_utils.fill_octree_arrays(ad, fields, total_octs, LeftEdge, dx, Level, Fields)
#    octs_dic = np.load('octs_dic.npy')[()]

    #initialize arrays to be passed to the recursion algo
    octs_enclosed = np.argwhere(np.all(octs_dic['LeftEdge'] >= fle, axis=1) &
                           np.all(octs_dic['LeftEdge'] < fre, axis=1))[:,0]
    nocts_enclosed = len(octs_enclosed) 
    print 'Total_octs = %d , Nocts_enclosed = %d' % (total_octs, nocts_enclosed)
    output   = np.zeros((8*nocts_enclosed, len(fields)), dtype='float64')
    refined  = np.zeros(8*nocts_enclosed, dtype='uint32')
    levels   = np.zeros(8*nocts_enclosed, dtype='int32')
#    ids      = np.zeros(8*nocts_enclosed, dtype='uint64')
    start_time = time.time()

    '''
    pos = position()
    hs  = hilbert_state()
    if debug:  debug = print_oct
    pbar = get_pbar("Building Hilbert Depth First Octree", nocts_enclosed)
    RecurseOctreeDepthFirstHilbert(ile,
                                   -1, #we start on the root grid
                                   pos,
                                   hs, 
                                   output, refined, levels, ids,
                                   octs_dic,
                                   start_level,
                                   debug=debug,
                                   tracker=pbar)
    pbar.finish()
    '''                               
    pos = amr_utils.position_c()
    hs  = amr_utils.hilbert_state_c()   
#   pbar = get_pbar("Building Hilbert Depth First Octree", nocts_enclosed)
    amr_utils.RecurseOctreeDepthFirstHilbert_c(ile,
                                               -1, #we start on the root grid
                                               pos,
                                               hs, 
                                               output, refined, levels,
                                               octs_dic['LeftEdge'][octs_enclosed],
                                               octs_dic['dx'][octs_enclosed],
                                               octs_dic['Level'][octs_enclosed],
                                               octs_dic['Fields'][octs_enclosed],
                                               start_level,
                                               start_time,                                              
                                               tracker=None)
#    pbar.finish()
    #by time we get it here the 'current' position is actually 
    #for the next spot, so we're off by 1
    print 'Took %1.2e seconds'%(time.time()-start_time)
    print '# of refined octs = %i, # of leaves %i'%(pos.refined_pos, pos.output_pos) 
    print 'first few entries :',refined[:12]
    output  = output[:pos.output_pos]
    refined = refined[:pos.refined_pos] 
    levels = levels[:pos.refined_pos] 
    return output, refined, pos.refined_pos, pos.output_pos


def RecurseOctreeDepthFirstHilbert(child_index,# Integer position index of the children oct within a 
                                               # parent oct or within the root grid of octs (super level) 
                            oct_id, # ID of parent oct (-1 if root grid of octs) 
                            pos, # The output hydro data position and refinement position
                            hilbert,  # The hilbert state
                            output, # Holds the hydro data
                            refined, # Holds the refinement status  of Octs, 0s and 1s
                            levels, # For a given Oct, what is the level
                            ids, # Record of the octs IDs. Oct IDs match indexes in octs_dic arrays
                            octs_dic, # Dictionary with arrays of octs properties and fields
                            level, # level of the parent oct, if level < 0 we are at the super level (root grid of octs)
                            debug=None,tracker=None):

    if tracker:
        if pos.refined_pos%500 == 0 : tracker.update(pos.refined_pos)

    if debug: debug(vars())
  
    if oct_id == -1:
        # we are at the root grid of octs
        children = np.argwhere(octs_dic['Level'] == 0)[:,0]
        cLE =  octs_dic['LeftEdge'][children]
        dx = octs_dic['dx'][children]
        assert (len(np.unique(dx)) == 1 and
                np.unique(dx) == octs_dic['dx'].max()) 
        pdx = 2*dx[0]
        thischild = np.all(cLE ==  np.array([child_index[0]*pdx,
                                             child_index[1]*pdx,
                                             child_index[2]*pdx]), axis=1)
        assert len(children[thischild] == 1)
        child_oct_id = children[thischild][0] 
    else:    
        children, child_index_mask = get_oct_children(oct_id, octs_dic)   
        child_oct_id = child_index_mask[child_index[0], child_index[1], child_index[2]]

    #record the refinement state
    levels[pos.refined_pos]  = level
    is_leaf = (child_oct_id == -1) and (level > 0) #Don't subdivide if we are on a superlevel
    refined[pos.refined_pos] = not is_leaf #True is oct, False is leaf
    ids[pos.refined_pos] = child_oct_id 
    pos.refined_pos += 1 

    if debug:  print 'child_oct_id = %d, refined_pos = %d'%(child_oct_id, pos.refined_pos)

    if is_leaf: 
        #then we have hit a leaf cell; write it out
        if debug: print 'Is leaf, writing to output. output_pos =  ', pos.output_pos 
        fields = octs_dic['Fields'][oct_id]
        for field_index in range(fields.shape[0]):
            output[pos.output_pos,field_index] = \
                    fields[field_index,child_index[0],child_index[1],child_index[2]]
        pos.output_pos+= 1 
    else:
        assert child_oct_id > -1
 
        for (vertex, hilbert_child) in hilbert:
            #vertex is a combination of three 0s and 1s to 
            #denote each of the 8 octs
            if level < 0:
                next_oct_id = oct_id #Don't descend if we're a superlevel
                #child_ile = child_index + np.array(vertex)*2**(-level)
                next_child_index = child_index + np.array(vertex)*2**(-(level+1))
                next_child_index = next_child_index.astype('int')

            else:
                next_oct_id = child_oct_id  #Descend

                # Get the floating point left edge of the oct we descended into and the current oct/grid
                child_oct_le = octs_dic['LeftEdge'][child_oct_id]
                child_oct_dx = octs_dic['dx'][child_oct_id]
                if oct_id == -1:
                    parent_oct_le = child_index*pdx
                else:    
                    parent_oct_le = octs_dic['LeftEdge'][oct_id] + child_index*octs_dic['dx'][oct_id]

                # Then translate onto the subgrid integer index 
                child_oct_ile = np.floor((parent_oct_le - child_oct_le)/child_oct_dx)

                next_child_index = child_oct_ile+np.array(vertex)
                next_child_index = next_child_index.astype('int')

            RecurseOctreeDepthFirstHilbert(next_child_index, next_oct_id, pos,
                                           hilbert_child, output, refined, levels,
                                           octs_dic, level+1, ids = ids,
                                           debug=debug,tracker=tracker)


def create_fits_file(ds, fn, output, refined, particle_data, fle, fre):
    #first create the grid structure
    structure = pyfits.Column("structure", format="B", array=refined.astype("bool"))
    cols = pyfits.ColDefs([structure])
    st_table = pyfits.BinTableHDU.from_columns(cols)
    st_table.name = "GRIDSTRUCTURE"
    st_table.header.set("hierarch lengthunit", "kpc", comment="Length unit for grid")
    fre = ds.arr(fre, 'code_length').in_units('kpc').value
    fle = ds.arr(fle, 'code_length').in_units('kpc').value
    fdx = fre-fle
    for i,a in enumerate('xyz'):
        st_table.header.set("min%s" % a, fle[i])
        st_table.header.set("max%s" % a, fre[i])
        st_table.header.set("n%s" % a, fdx[i])
        st_table.header.set("subdiv%s" % a, 2)
    st_table.header.set("subdivtp", "OCTREE", "Type of grid subdivision")

    #not the hydro grid data
    fields = ["CellMassMsun","TemperatureTimesCellMassMsun", "MetalMassMsun", "CellVolumeKpc"]
    fd = {}
    for i,f in enumerate(fields): 
        fd[f]=output[:,i]
    del output
    col_list = []
    cell_mass = fd["CellMassMsun"]
    size = cell_mass.size
    tm = cell_mass.sum()
    col_list.append(pyfits.Column("mass_gas", format='D',
                    array=cell_mass, unit="Msun"))
    col_list.append(pyfits.Column("mass_metals", format='D',
                    array=fd['MetalMassMsun'], unit="Msun"))
    # col_list.append(pyfits.Column("mass_stars", format='D',
    #                 array=np.zeros(size,dtype='D'),unit="Msun"))
    # col_list.append(pyfits.Column("mass_stellar_metals", format='D',
    #                 array=np.zeros(size,dtype='D'),unit="Msun"))
    # col_list.append(pyfits.Column("age_m", format='D',
    #                 array=np.zeros(size,dtype='D'),unit="yr*Msun"))
    # col_list.append(pyfits.Column("age_l", format='D',
    #                 array=np.zeros(size,dtype='D'),unit="yr*Msun"))
    # col_list.append(pyfits.Column("L_bol", format='D',
    #                 array=np.zeros(size,dtype='D')))
    # col_list.append(pyfits.Column("L_lambda", format='D',
    #                 array=np.zeros(size,dtype='D')))
    # The units for gas_temp are really K*Msun. For older Sunrise versions
    # you must set the unit to just K  
    col_list.append(pyfits.Column("gas_temp_m", format='D',
                    array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
    col_list.append(pyfits.Column("gas_teff_m", format='D',
                    array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
    col_list.append(pyfits.Column("cell_volume", format='D',
                    array=fd['CellVolumeKpc'], unit="kpc^3"))
    col_list.append(pyfits.Column("SFR", format='D',
                    array=np.zeros(size, dtype='D')))
    cols = pyfits.ColDefs(col_list)
    mg_table = pyfits.BinTableHDU.from_columns(cols)
#    mg_table = pyfits.new_table(cols)
    mg_table.header.set("M_g_tot", tm)
    mg_table.header.set("timeunit", "yr")
    mg_table.header.set("tempunit", "K")
    mg_table.name = "GRIDDATA"

    # Add a dummy Primary; might be a better way to do this!
    col_list = [pyfits.Column("dummy", format="E", array=np.zeros(1, dtype='float32'))]
    cols = pyfits.ColDefs(col_list)
    md_table = pyfits.BinTableHDU.from_columns(cols)
#    md_table = pyfits.new_table(cols)
    md_table.header.set("snaptime", ds.current_time.in_units('yr').value[()])
    md_table.name = "YT"

    phdu = pyfits.PrimaryHDU()
    phdu.header.set('nbodycod','yt')
    hls = [phdu, st_table, mg_table,md_table]
    hls.append(particle_data)
    hdus = pyfits.HDUList(hls)
    hdus.writeto(fn, clobber=True)


def nearest_power(x):
    #round to the nearest power of 2
    x-=1
    x |= x >> 1
    x |= x >> 2 
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x+=1 
    return x


def round_nocts_wide(dds,fle,fre,nwide=None):
    fc = (fle+fre)/2.0
    assert np.all(fle < fc)
    assert np.all(fre > fc)
    ic = np.rint(fc*dds) #nearest vertex to the center
    ile,ire = ic.astype('int32'),ic.astype('int32')
    cfle,cfre = fc.copy(),fc.copy()
    idx = np.array([0,0,0]) #just a random non-equal array
    width = 0.0
    if nwide is None:
        #expand until borders are included and
        #we have an equaly-sized, non-zero box
        idxq,out=False,True
        while not out or not idxq:
            cfle,cfre = fc-width, fc+width
            ile = np.rint(cfle*dds).astype('int32')
            ire = np.rint(cfre*dds).astype('int32')
            idx = ire-ile
            width += 0.1/dds
            #quit if idxq is true:
            idxq = idx[0]>0 and np.all(idx==idx[0])
            out  = np.all(fle>cfle) and np.all(fre<cfre) 
            out &= abs(np.log2(idx[0])-np.rint(np.log2(idx[0])))<1e-5 #nwide should be a power of 2
            assert width[0] < 1.1 #can't go larger than the simulation volume
        nwide = idx[0]
    else:
        #expand until we are nwide cells span
        while not np.all(idx==nwide):
            assert np.any(idx<=nwide)
            cfle,cfre = fc-width, fc+width
            ile = np.rint(cfle*dds).astype('int32')
            ire = np.rint(cfre*dds).astype('int32')
            idx = ire-ile
            width += 1e-2*1.0/dds
    assert np.all(idx==nwide)
    assert idx[0]>0
    maxlevel = -np.rint(np.log2(nwide)).astype('int32')
    assert abs(np.log2(nwide)-np.rint(np.log2(nwide)))<1e-5 #nwide should be a power of 2
    return ile,ire,maxlevel,nwide


def round_nearest_edge(ds,fle,fre):
    dds = ds.domain_dimensions/2
    ile = np.floor(fle*dds).astype('int')
    ire = np.ceil(fre*dds).astype('int') 
    
    #this is the number of cells the super octree needs to expand to
    #must round to the nearest power of 2
    width = np.max(ire-ile)
    width = nearest_power(width)
    
    maxlevel = -np.rint(np.log2(width)).astype('int')
    return ile,ire,maxlevel


class position:
    def __init__(self):
        self.output_pos = 0
        self.refined_pos = 0

class hilbert_state():
    def __init__(self,dim=None,sgn=None,octant=None):
        if dim is None: dim = [0,1,2]
        if sgn is None: sgn = [1,1,1]
        if octant is None: octant = 5
        self.dim = dim
        self.sgn = sgn
        self.octant = octant
    def flip(self,i):
        self.sgn[i]*=-1
    def swap(self,i,j):
        temp = self.dim[i]
        self.dim[i]=self.dim[j]
        self.dim[j]=temp
        axis = self.sgn[i]
        self.sgn[i] = self.sgn[j]
        self.sgn[j] = axis
    def reorder(self,i,j,k):
        ndim = [self.dim[i],self.dim[j],self.dim[k]] 
        nsgn = [self.sgn[i],self.sgn[j],self.sgn[k]]
        self.dim = ndim
        self.sgn = nsgn
    def copy(self):
        return hilbert_state([self.dim[0],self.dim[1],self.dim[2]],
                             [self.sgn[0],self.sgn[1],self.sgn[2]],
                             self.octant)
    def descend(self,o):
        child = self.copy()
        child.octant = o
        if o==0:
            child.swap(0,2)
        elif o==1:
            child.swap(1,2)
        elif o==2:
            pass
        elif o==3:
            child.flip(0)
            child.flip(2)
            child.reorder(2,0,1)
        elif o==4:
            child.flip(0)
            child.flip(1)
            child.reorder(2,0,1)
        elif o==5:
            pass
        elif o==6:
            child.flip(1)
            child.flip(2)
            child.swap(1,2)
        elif o==7:
            child.flip(0)
            child.flip(2)
            child.swap(0,2)
        return child

    def __iter__(self):
        vertex = np.array([0,0,0]).astype('int32')
        j=0
        for i in range(3):
            vertex[self.dim[i]] = 0 if self.sgn[i]>0 else 1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] += self.sgn[0]
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[1]] += self.sgn[1] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] -= self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[2]] += self.sgn[2] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] += self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[1]] -= self.sgn[1] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] -= self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)


def get_oct_children(parent_oct_id, octs_dic):
    """
    Find the children of the given parent oct
    """

    LE = octs_dic['LeftEdge']
    dx = octs_dic['dx']
    RE = octs_dic['LeftEdge'] + 2.0*np.array([dx, dx, dx]).transpose()
    Level = octs_dic['Level']

    pLE = LE[parent_oct_id]
    pdx = dx[parent_oct_id]
    pRE = pLE + 2.0*np.array([pdx, pdx, pdx]).transpose()
    pLevel = Level[parent_oct_id]
    children = np.argwhere(np.all(LE >= pLE, axis=1) &
                           np.all(RE <= pRE, axis=1) & 
                           (Level == pLevel + 1))[:,0] 
 
    child_index_mask = np.zeros((2, 2, 2), 'int32') - 1   
    if len(children):
        #assert children.size == 8 # Not all octs have 8 children (check if that is ok)
        assert children.size <= 8 # But they must have less than 8 then
        cLE = LE[children]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    thischild = np.all(cLE == (pLE + [i*pdx, j*pdx, k*pdx]), axis=1)
                    if children[thischild]:
                        assert len(children[thischild]) == 1
                        child_index_mask[i,j,k] = children[thischild][0] 
       
    return children, child_index_mask


def print_oct(data,nd=None,nc=None):
    ci = data['child_index']    
    o_id  = data['oct_id']    
    octs_dic  = data['octs_dic']
    l  = data['level']
    
    if o_id == -1:
        LE = np.array([0,0,0])
        dx = 2*octs_dic['dx'][0]
        children = np.argwhere(octs_dic['Level'] == 0)[:,0]
    else:    
        LE = octs_dic['LeftEdge'][o_id]
        dx = octs_dic['dx'][o_id]
        children, child_index_mask = get_oct_children(o_id, octs_dic) 

    fle = LE + dx*ci
    fre = LE + dx*(ci+1)
    if nd is not None:
        fle *= nd
        fre *= nd
        if nc is not None:
            fle -= nc
            fre -= nc
    txt  = '%1i '
    txt += '%1i '
    txt += '%1.3f '*3+'- '
    txt += '%1.3f '*3
#    if l<2:
    print ''
    print txt%((l,)+(o_id,)+tuple(fle)+tuple(fre))
    print children
    if o_id > -1:
        print child_index_mask
       
 
