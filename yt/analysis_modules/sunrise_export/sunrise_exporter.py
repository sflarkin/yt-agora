"""
Code to export from yt to Sunrise

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: UCSD
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010 Matthew Turk.  All Rights Reserved.

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

try:
    import pyfits
except ImportError:
    # We silently fail here
    pass

import time
import numpy as na

from yt.funcs import *
import yt.utilities.amr_utils as amr_utils

def export_to_sunrise(pf, fn, write_particles = True, subregion_bounds = None):
    r"""Convert the contents of a dataset to a FITS file format that Sunrise
    understands.

    This function will accept a parameter file, and from that parameter file
    construct a depth-first octree containing all of the data in the parameter
    file.  This octree will be written to a FITS file.  It will probably be
    quite big, so use this function with caution!  Sunrise is a tool for
    generating synthetic spectra, available at
    http://sunrise.googlecode.com/ .

    Parameters
    ----------
    pf : `StaticOutput`
        The parameter file to convert.
    fn : string
        The filename of the FITS file.
    write_particles : bool, default is True
        Whether to write out the star particles or not
    subregion_bounds : list of tuples
        This is a list of tuples describing the subregion of the top grid to
        export.  This will only work when only *one* root grid exists.
        It is of the format:
        [ (start_index_x, nx), (start_index_y, ny), (start_index_z, nz) ]
        where nx, ny, nz are the number of cells to extract.

    Notes
    -----
    Note that the process of generating simulated images from Sunrise will
    require substantial user input; see the Sunrise wiki at
    http://sunrise.googlecode.com/ for more information.

    """
    # Now particles
    #  output_file->addTable("PARTICLEDATA" , 0);
    # addKey("timeunit", time_unit, "Time unit is "+time_unit);
    # addKey("tempunit", temp_unit, "Temperature unit is "+temp_unit);
    # 
    # addColumn(Tint, "ID", 1, "" );
    # addColumn(Tdouble, "position", 3, length_unit );
    # addColumn(Tdouble, "stellar_radius", 1, length_unit );
    # addColumn(Tdouble, "L_bol", 1, L_bol_unit );
    # addColumn(Tdouble, "mass_stars", 1, mass_unit );
    # addColumn(Tdouble, "mass_stellar_metals", 1, mass_unit );
    # addColumn(Tdouble, "age_m", 1, time_unit+"*"+mass_unit );
    # addColumn(Tdouble, "age_l", 1, time_unit+"*"+mass_unit );
    # addColumn(Tfloat, "L_lambda", L_lambda.columns(), 
    #			L_lambda_unit );
    #	output->addKey("logflux", true, "Column L_lambda values are log (L_lambda)");

    col_list = []
    DLE, DRE = pf.domain_left_edge, pf.domain_right_edge
    reg = pf.h.region((DRE+DLE)/2.0, DLE, DRE)

    if write_particles:
        pi = reg["particle_type"] == 2

        pmass = reg["ParticleMassMsun"][pi]
        col_list.append(pyfits.Column(
            "ID", format="I", array=na.arange(pmass.size)))
        pos = na.array([reg["particle_position_%s" % ax][pi]*pf['kpc']
                            for ax in 'xyz']).transpose()
        col_list.append(pyfits.Column("position", format="3D",
            array=pos))
        col_list.append(pyfits.Column("mass_stars", format="D",
            array=pmass))
        age = pf["years"] * (pf["InitialTime"] - reg["creation_time"][pi])
        col_list.append(pyfits.Column("age_m", format="D", array=age))
        col_list.append(pyfits.Column("age_l", format="D", array=age))
        col_list.append(pyfits.Column("mass_stellar_metals", format="D",
            array=0.02*pmass*reg["metallicity_fraction"][pi])) # wrong?
        col_list.append(pyfits.Column("L_bol", format="D",
            array=na.zeros(pmass.size)))

        # Still missing: L_bol, L_lambda, stellar_radius
        cols = pyfits.ColDefs(col_list)
        pd_table = pyfits.new_table(cols)
        pd_table.name = "PARTICLEDATA"

    output, refined = generate_flat_octree(pf,
            ["CellMassMsun","Temperature", "Metal_Density",
             "CellVolumeCode"], subregion_bounds = subregion_bounds)
    cvcgs = output["CellVolumeCode"].astype('float64') * pf['cm']**3.0

    # First the structure
    structure = pyfits.Column("structure", format="B", array=refined.astype("bool"))
    cols = pyfits.ColDefs([structure])
    st_table = pyfits.new_table(cols)
    st_table.name = "STRUCTURE"

    # Now we update our table with units
    # ("lengthunit", length_unit, "Length unit for grid");
    # ("minx", getmin () [0], length_unit_comment);
    # ("miny", getmin () [1], length_unit_comment);
    # ("minz", getmin () [2], length_unit_comment);
    # ("maxx", getmax () [0], length_unit_comment);
    # ("maxy", getmax () [1], length_unit_comment);
    # ("maxz", getmax () [2], length_unit_comment);
    # ("nx", g_.getn () [0], "");
    # ("ny", g_.getn () [1], "");
    # ("nz", g_.getn () [2], "");
    # ("subdivtp", subdivtp, "Type of grid subdivision");
    # ("subdivx", sub_div[0], "");
    # ("subdivy", sub_div[1], "");
    # ("subdivz", sub_div[2], "");

    st_table.header.update("hierarch lengthunit", 1.0, comment="Length unit for grid")
    for i,a in enumerate('xyz'):
        st_table.header.update("min%s" % a, pf.domain_left_edge[i] * pf['kpc'])
        st_table.header.update("max%s" % a, pf.domain_right_edge[i] * pf['kpc'])
        st_table.header.update("n%s" % a, pf.domain_dimensions[i])
        st_table.header.update("subdiv%s" % a, 2)
    st_table.header.update("subdivtp", "UNIFORM", "Type of grid subdivision")

    # Now grid data itself
    # ("M_g_tot", total_quantities.m_g(), "[" + mass_unit +
    #         "] Total gas mass in all cells");
    # ("SFR_tot", total_quantities.SFR, "[" + SFR_unit +
    #         "] Total star formation rate of all cells");
    # ("timeunit", time_unit, "Time unit is "+time_unit);
    # ("tempunit", temp_unit, "Temperature unit is "+time_unit);

    # (Tdouble, "mass_gas", 1, mass_unit );
    # (Tdouble, "SFR", 1, SFR_unit );
    # (Tdouble, "mass_metals", 1, mass_unit );
    # (Tdouble, "gas_temp_m", 1, temp_unit+"*"+mass_unit );
    # (Tdouble, "gas_teff_m", 1, temp_unit+"*"+mass_unit );
    # (Tdouble, "cell_volume", 1, length_unit + "^3" );

    col_list = []
    size = output["CellMassMsun"].size
    tm = output["CellMassMsun"].sum()
    col_list.append(pyfits.Column("mass_gas", format='D',
                    array=output.pop('CellMassMsun')))
    col_list.append(pyfits.Column("mass_metals", format='D',
                    array=output.pop('Metal_Density')*cvcgs/1.989e33))
    col_list.append(pyfits.Column("gas_temp_m", format='D',
                    array=output['Temperature']))
    col_list.append(pyfits.Column("gas_teff_m", format='D',
                    array=output.pop('Temperature')))
    col_list.append(pyfits.Column("cell_volume", format='D',
                    array=output.pop('CellVolumeCode').astype('float64')*pf['kpc']**3.0))
    col_list.append(pyfits.Column("SFR", format='D',
                    array=na.zeros(size, dtype='D')))
    cols = pyfits.ColDefs(col_list)
    mg_table = pyfits.new_table(cols)
    mg_table.header.update("M_g_tot", tm)
    mg_table.header.update("timeunit", 1.0)
    mg_table.header.update("tempunit", 1.0)
    mg_table.name = "GRIDDATA"

    # Add a dummy Primary; might be a better way to do this!
    hls = [pyfits.PrimaryHDU(), st_table, mg_table]
    if write_particles: hls.append(pd_table)
    hdus = pyfits.HDUList(hls)
    hdus.writeto(fn, clobber=True)

def initialize_octree_list(pf, fields):
    o_length = r_length = 0
    grids = []
    levels_finest, levels_all = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for g in pf.h.grids:
        ff = na.array([g[f] for f in fields])
        grids.append(amr_utils.OctreeGrid(
                        g.child_index_mask.astype('int32'),
                        ff.astype("float64"),
                        g.LeftEdge.astype('float64'),
                        g.ActiveDimensions.astype('int32'),
                        na.ones(1,dtype='float64') * g.dds[0], g.Level,
                        g._id_offset))
        levels_all[g.Level] += g.ActiveDimensions.prod()
        levels_finest[g.Level] += g.child_mask.ravel().sum()
        g.clear_data()
    ogl = amr_utils.OctreeGridList(grids)
    return ogl, levels_finest, levels_all

def generate_flat_octree(pf, fields, subregion_bounds = None):
    """
    Generates two arrays, one of the actual values in a depth-first flat
    octree array, and the other of the values describing the refinement.
    This allows for export to a code that understands this.  *field* is the
    field used in the data array.
    """
    fields = ensure_list(fields)
    ogl, levels_finest, levels_all = initialize_octree_list(pf, fields)
    o_length = na.sum(levels_finest.values())
    r_length = na.sum(levels_all.values())
    output = na.zeros((o_length,len(fields)), dtype='float64')
    refined = na.zeros(r_length, dtype='int32')
    position = amr_utils.position()
    if subregion_bounds is None:
        sx, sy, sz = 0, 0, 0
        nx, ny, nz = ogl[0].dimensions
    else:
        ss, ns = zip(*subregion_bounds)
        sx, sy, sz = ss
        nx, ny, nz = ns
    print "Running from %s for %s cells" % (
            (sx,sy,sz), (nx,ny,nz))
    t1 = time.time()
    amr_utils.RecurseOctreeDepthFirst(
               sx, sy, sz, nx, ny, nz,
               position, ogl[0].offset,
               output, refined, ogl)
    t2 = time.time()
    print "Finished.  Took %0.3e seconds." % (t2-t1)
    dd = {}
    for i, field in enumerate(fields):
        dd[field] = output[:position.output_pos,i]
    return dd, refined

def generate_levels_octree(pf, fields):
    fields = ensure_list(fields) + ["Ones", "Ones"]
    ogl, levels_finest, levels_all = initialize_octree_list(fields)
    o_length = na.sum(levels_finest.values())
    r_length = na.sum(levels_all.values())
    output = na.zeros((r_length,len(fields)), dtype='float64')
    genealogy = na.zeros((r_length, 3), dtype='int32') - 1 # init to -1
    corners = na.zeros((r_length, 3), dtype='float64')
    position = na.add.accumulate(
                na.array([0] + [levels_all[v] for v in
                    sorted(levels_all)[:-1]], dtype='int32'))
    pp = position.copy()
    amr_utils.RecurseOctreeByLevels(0, 0, 0,
               ogl[0].dimensions[0],
               ogl[0].dimensions[1],
               ogl[0].dimensions[2],
               position.astype('int32'), 1,
               output, genealogy, corners, ogl)
    return output, genealogy, levels_all, levels_finest, pp, corners

