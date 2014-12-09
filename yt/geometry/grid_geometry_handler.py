"""
AMR index container class



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import h5py
import numpy as np
import string, re, gc, time, cPickle
import weakref

from itertools import chain, izip

from yt.funcs import *
from yt.utilities.logger import ytLogger as mylog
from yt.arraytypes import blankRecordArray
from yt.config import ytcfg
from yt.fields.field_info_container import NullFunc
from yt.geometry.geometry_handler import \
    Index, YTDataChunk, ChunkDataCache
from yt.utilities.definitions import MAXLEVEL
from yt.utilities.physical_constants import sec_per_year
from yt.utilities.io_handler import io_registry
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface
from yt.utilities.lib.GridTree import GridTree, MatchPointsToGrids

from yt.data_objects.data_containers import data_object_registry

class GridIndex(Index):
    """The index class for patch and block AMR datasets. """
    float_type = 'float64'
    _preload_implemented = False
    _index_properties = ("grid_left_edge", "grid_right_edge",
                         "grid_levels", "grid_particle_count",
                         "grid_dimensions")

    def _setup_geometry(self):
        mylog.debug("Counting grids.")
        self._count_grids()

        mylog.debug("Initializing grid arrays.")
        self._initialize_grid_arrays()

        mylog.debug("Parsing index.")
        self._parse_index()

        mylog.debug("Constructing grid objects.")
        self._populate_grid_objects()

        mylog.debug("Re-examining index")
        self._initialize_level_stats()

    def __del__(self):
        del self.grid_dimensions
        del self.grid_left_edge
        del self.grid_right_edge
        del self.grid_levels
        del self.grid_particle_count
        del self.grids

    @property
    def parameters(self):
        return self.dataset.parameters

    def _detect_output_fields_backup(self):
        # grab fields from backup file as well, if present
        return
        try:
            backup_filename = self.dataset.backup_filename
            f = h5py.File(backup_filename, 'r')
            g = f["data"]
            grid = self.grids[0] # simply check one of the grids
            grid_group = g["grid_%010i" % (grid.id - grid._id_offset)]
            for field_name in grid_group:
                if field_name != 'particles':
                    self.field_list.append(field_name)
        except KeyError:
            return
        except IOError:
            return

    def select_grids(self, level):
        """
        Returns an array of grids at *level*.
        """
        return self.grids[self.grid_levels.flat == level]

    def get_levels(self):
        for level in range(self.max_level+1):
            yield self.select_grids(level)

    def _initialize_grid_arrays(self):
        mylog.debug("Allocating arrays for %s grids", self.num_grids)
        self.grid_dimensions = np.ones((self.num_grids,3), 'int32')
        self.grid_left_edge = self.ds.arr(np.zeros((self.num_grids,3),
                                    self.float_type), 'code_length')
        self.grid_right_edge = self.ds.arr(np.ones((self.num_grids,3),
                                    self.float_type), 'code_length')
        self.grid_levels = np.zeros((self.num_grids,1), 'int32')
        self.grid_particle_count = np.zeros((self.num_grids,1), 'int32')

    def clear_all_data(self):
        """
        This routine clears all the data currently being held onto by the grids
        and the data io handler.
        """
        for g in self.grids: g.clear_data()
        self.io.queue.clear()

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        return self.select_grids(self.grid_levels.max())[0].dds[:].min()

    def _initialize_level_stats(self):
        # Now some statistics:
        #   0 = number of grids
        #   1 = number of cells
        #   2 = blank
        desc = {'names': ['numgrids','numcells','level'],
                'formats':['Int64']*3}
        self.level_stats = blankRecordArray(desc, MAXLEVEL)
        self.level_stats['level'] = [i for i in range(MAXLEVEL)]
        self.level_stats['numgrids'] = [0 for i in range(MAXLEVEL)]
        self.level_stats['numcells'] = [0 for i in range(MAXLEVEL)]
        for level in xrange(self.max_level+1):
            self.level_stats[level]['numgrids'] = np.sum(self.grid_levels == level)
            li = (self.grid_levels[:,0] == level)
            self.level_stats[level]['numcells'] = self.grid_dimensions[li,:].prod(axis=1).sum()

    @property
    def grid_corners(self):
        return np.array([
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
        ], dtype='float64')

    def lock_grids_to_parents(self):
        r"""This function locks grid edges to their parents.

        This is useful in cases where the grid structure may be somewhat
        irregular, or where setting the left and right edges is a lossy
        process.  It is designed to correct situations where left/right edges
        may be set slightly incorrectly, resulting in discontinuities in images
        and the like.
        """
        mylog.info("Locking grids to parents.")
        for i, g in enumerate(self.grids):
            si = g.get_global_startindex()
            g.LeftEdge = self.ds.domain_left_edge + g.dds * si
            g.RightEdge = g.LeftEdge + g.ActiveDimensions * g.dds
            self.grid_left_edge[i,:] = g.LeftEdge
            self.grid_right_edge[i,:] = g.RightEdge

    def print_stats(self):
        """
        Prints out (stdout) relevant information about the simulation
        """
        header = "%3s\t%6s\t%14s\t%14s" % ("level","# grids", "# cells",
                                           "# cells^3")
        print header
        print "%s" % (len(header.expandtabs())*"-")
        for level in xrange(MAXLEVEL):
            if (self.level_stats['numgrids'][level]) == 0:
                break
            print "% 3i\t% 6i\t% 14i\t% 14i" % \
                  (level, self.level_stats['numgrids'][level],
                   self.level_stats['numcells'][level],
                   np.ceil(self.level_stats['numcells'][level]**(1./3)))
            dx = self.select_grids(level)[0].dds[0]
        print "-" * 46
        print "   \t% 6i\t% 14i" % (self.level_stats['numgrids'].sum(), self.level_stats['numcells'].sum())
        print "\n"
        try:
            print "z = %0.8f" % (self["CosmologyCurrentRedshift"])
        except:
            pass
        print "t = %0.8e = %0.8e s = %0.8e years" % \
            (self.ds.current_time.in_units("code_time"),
             self.ds.current_time.in_units("s"),
             self.ds.current_time.in_units("yr"))
        print "\nSmallest Cell:"
        u=[]
        for item in ("Mpc", "pc", "AU", "cm"):
            print "\tWidth: %0.3e %s" % (dx.in_units(item), item)

    def _find_field_values_at_points(self, fields, coords):
        r"""Find the value of fields at a set of coordinates.

        Returns the values [field1, field2,...] of the fields at the given
        (x, y, z) points. Returns a numpy array of field values cross coords
        """
        coords = YTArray(ensure_numpy_array(coords),'code_length', registry=self.ds.unit_registry)
        grids = self._find_points(coords[:,0], coords[:,1], coords[:,2])[0]
        fields = ensure_list(fields)
        mark = np.zeros(3, dtype=np.int)
        out = []

        # create point -> grid mapping
        grid_index = {}
        for coord_index, grid in enumerate(grids):
            if not grid_index.has_key(grid):
                grid_index[grid] = []
            grid_index[grid].append(coord_index)

        out = np.zeros((len(fields),len(coords)), dtype=np.float64)
        for grid in grid_index:
            cellwidth = (grid.RightEdge - grid.LeftEdge) / grid.ActiveDimensions
            for field in fields:
                for coord_index in grid_index[grid]:
                    mark = ((coords[coord_index,:] - grid.LeftEdge) / cellwidth).astype('int')
                    out[:,coord_index] = grid[field][mark[0],mark[1],mark[2]]
        return out


    def _find_points(self, x, y, z) :
        """
        Returns the (objects, indices) of leaf grids containing a number of (x,y,z) points
        """
        x = ensure_numpy_array(x)
        y = ensure_numpy_array(y)
        z = ensure_numpy_array(z)
        if not len(x) == len(y) == len(z):
            raise AssertionError("Arrays of indices must be of the same size")

        grid_tree = self._get_grid_tree()
        pts = MatchPointsToGrids(grid_tree, len(x), x, y, z)
        ind = pts.find_points_in_tree()
        return self.grids[ind], ind

    def _get_grid_tree(self) :

        left_edge = self.ds.arr(np.zeros((self.num_grids, 3)),
                               'code_length')
        right_edge = self.ds.arr(np.zeros((self.num_grids, 3)),
                                'code_length')
        level = np.zeros((self.num_grids), dtype='int64')
        parent_ind = np.zeros((self.num_grids), dtype='int64')
        num_children = np.zeros((self.num_grids), dtype='int64')

        for i, grid in enumerate(self.grids) :

            left_edge[i,:] = grid.LeftEdge
            right_edge[i,:] = grid.RightEdge
            level[i] = grid.Level
            if grid.Parent is None :
                parent_ind[i] = -1
            else :
                parent_ind[i] = grid.Parent.id - grid.Parent._id_offset
            num_children[i] = np.int64(len(grid.Children))

        return GridTree(self.num_grids, left_edge, right_edge, parent_ind,
                        level, num_children)

    def convert(self, unit):
        return self.dataset.conversion_factors[unit]

    def _identify_base_chunk(self, dobj):
        if dobj._type_name == "grid":
            dobj._chunk_info = np.empty(1, dtype='object')
            dobj._chunk_info[0] = dobj
        elif getattr(dobj, "_grids", None) is None:
            gi = dobj.selector.select_grids(self.grid_left_edge,
                                            self.grid_right_edge,
                                            self.grid_levels)
            grids = list(sorted(self.grids[gi], key = lambda g: g.filename))
            dobj._chunk_info = np.empty(len(grids), dtype='object')
            for i, g in enumerate(grids):
                dobj._chunk_info[i] = g
        if getattr(dobj, "size", None) is None:
            dobj.size = self._count_selection(dobj)
        if getattr(dobj, "shape", None) is None:
            dobj.shape = (dobj.size,)
        dobj._current_chunk = list(self._chunk_all(dobj, cache = False))[0]

    def _count_selection(self, dobj, grids = None):
        if grids is None: grids = dobj._chunk_info
        count = sum((g.count(dobj.selector) for g in grids))
        return count

    def _chunk_all(self, dobj, cache = True):
        gobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        yield YTDataChunk(dobj, "all", gobjs, dobj.size, cache)
        
    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        gobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        if sort in ("+level", "level"):
            giter = sorted(gobjs, key = g.Level)
        elif sort == "-level":
            giter = sorted(gobjs, key = -g.Level)
        elif sort is None:
            giter = gobjs
        if preload_fields is None: preload_fields = []
        preload_fields, _ = self._split_fields(preload_fields)
        if self._preload_implemented and len(preload_fields) > 0 and ngz == 0:
            giter = ChunkDataCache(list(giter), preload_fields, self)
        for i, og in enumerate(giter):
            if ngz > 0:
                g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
            else:
                g = og
            size = self._count_selection(dobj, [og])
            if size == 0: continue
            # We don't want to cache any of the masks or icoords or fcoords for
            # individual grids.
            yield YTDataChunk(dobj, "spatial", [g], size, cache = False)

    _grid_chunksize = 1000
    def _chunk_io(self, dobj, cache = True, local_only = False,
                  preload_fields = None):
        # local_only is only useful for inline datasets and requires
        # implementation by subclasses.
        if preload_fields is None:
            preload_fields = []
        preload_fields, _ = self._split_fields(preload_fields)
        gfiles = defaultdict(list)
        gobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for g in gobjs:
            gfiles[g.filename].append(g)
        for fn in sorted(gfiles):
            # We can apply a heuristic here to make sure we aren't loading too
            # many grids all at once.
            gs = gfiles[fn]
            size = self._grid_chunksize
            
            for grids in (gs[pos:pos + size] for pos
                          in xrange(0, len(gs), size)):
                dc = YTDataChunk(dobj, "io", grids,
                        self._count_selection(dobj, grids),
                        cache = cache)
                with self.io.preload(dc, preload_fields, fn):
                    yield dc
