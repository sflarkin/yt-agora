"""
The strings we need for weaving.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@license:
  Copyright (C) 2007 Matthew Turk.  All Rights Reserved.

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

ProfileBinningWeighted = \
"""
int bin;
int element;
int mybin;
npy_float64 myweight;          // should be higher precision
npy_float64 weights[num_bins]; // should be higher precision

for (bin = 0; bin < num_bins; bin++) {
  weights[bin] = 0;
  profilevalues(bin) = 0;
}

for (element = 0; element < num_elements ; element++) {
  mybin = binindices(element);
  profilevalues(mybin) += weightvalues(mybin)*fieldvalues(element);
  weights[mybin] += weightvalues(mybin);
}

for (bin = 0 ; bin < num_bins ; bin++ ) {
  profilevalues(bin) = profilevalues(bin) / weights[bin];
}
"""


ProfileBinningAccumulation = \
"""
int bin;
int element;
int mybin;
npy_float64 myweight;          // should be higher precision
npy_float64 weights[num_bins]; // should be higher precision

for (bin = 0; bin < num_bins; bin++) {
  weights[bin] = 0;
  profilevalues(bin) = 0;
}

for (element = 0; element < num_elements ; element++) {
  mybin = binindices(element);
  profilevalues(mybin) += fieldvalues(element);
}

for (bin = 1 ; bin < num_bins ; bin++ ) {
  profilevalues(bin) += profilevalues(bin-1);
}
"""

ProjectionRefineCoarseData = \
"""
/*
fpoints,  finedata_x,  finedata_y, finedata_vals, finedata_wgt,
cpoints,  coarsedata_x,  coarsedata_y, coarsedata_vals, coarsedata_wgt,
refinementFactor,  totalRefined

#define MIPCOMB(A,B) ((A) > (B) ? (A) : (B))
#define SUMCOMB(A,B) (A + B)

long fi, ci;

int flagged[cpoints], tr = 0;

long double rf = refinementFactor;

npy_int64 coarseCell_x, coarseCell_y;

for (ci=0; ci<cpoints; ci++) flagged[ci]=0;

for (fi = 0; fi < fpoints; fi++) {
    coarseCell_x = floorl((finedata_x(fi))/rf);
    coarseCell_y = floorl((finedata_y(fi))/rf);
    for (ci = 0; ci < cpoints; ci++) {
        if ((coarseCell_x == coarsedata_x(ci)) &&
            (coarseCell_y == coarsedata_y(ci))) {
                tr += 1;
                finedata_vals(fi) = %(COMBTYPE)sCOMB(coarsedata_vals[ci], finedata_vals[fi]);
                finedata_wgt(fi)  = %(COMBTYPE)sCOMB(coarsedata_wgt[ci],  finedata_wgt[fi]);
                flagged[ci] = 1;
                break;  // Each fine cell maps to one and only one coarse
                        // cell
        }
    }
}
*totalRefined = tr;

for (ci=0; ci<cpoints; ci++) {
    if (flagged[ci]==1) {
        coarsedata_x(ci)=-1;
        coarsedata_y(ci)=-1;
        coarsedata_vals(ci)=0.0;
        coarsedata_wgt(ci)=0.0;
    }
}
*/

"""

_baseDataCubeRefineCoarseData = \
r"""

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) < (B) ? (B) : (A))

int bx_m, by_m, bz_m, xg, yg, zg, xm, ym, zm;
int ixm, iym, izm;
int max_bx_m, max_by_m, max_bz_m;

for (xg = 0; xg < nx_g; xg++) {
  if (leftEdgeGrid(0)+dx_g*xg > cubeRightEdge(0)) continue;
  if (leftEdgeGrid(0)+dx_g*(xg+1) < leftEdgeCube(0)) continue;
  bx_m = MAX(ceill((leftEdgeGrid(0)+dx_g*xg - leftEdgeCube(0))/dx_m),0);
  max_bx_m = MIN(floorl((leftEdgeGrid(0)+dx_g*(xg+1) - leftEdgeCube(0))/dx_m),nx_m);
  for (yg = 0; yg < ny_g; yg++) {
    if (leftEdgeGrid(1)+dy_g*yg > cubeRightEdge(1)) continue;
    if (leftEdgeGrid(1)+dy_g*(yg+1) < leftEdgeCube(1)) continue;
    by_m = MAX(ceill((leftEdgeGrid(1)+dy_g*yg - leftEdgeCube(1))/dy_m),0);
    max_by_m = MIN(floorl((leftEdgeGrid(1)+dy_g*(yg+1) - leftEdgeCube(1))/dy_m),ny_m);
    for (zg = 0; zg < nz_g; zg++) {
      if (!lastLevel)
        if (childMask(xg, yg, zg) == 0) continue;
      if (leftEdgeGrid(2)+dz_g*zg > cubeRightEdge(2)) continue;
      if (leftEdgeGrid(2)+dz_g*(zg+1) < leftEdgeCube(2)) continue;
      bz_m = MAX(ceill((leftEdgeGrid(2)+dz_g*zg - leftEdgeCube(2))/dz_m),0);
      max_bz_m = MIN(floorl((leftEdgeGrid(2)+dz_g*(zg+1) - leftEdgeCube(2))/dz_m),nz_m);
      for (xm = bx_m; xm < max_bx_m ; xm++) {
        //if (xm < 0 || xm > nx_m) continue;
        for (ym = by_m; ym < max_by_m ; ym++) {
          //if (ym < 0 || ym > ny_m) continue;
          for (zm = bz_m; zm < max_bz_m ; zm++) {
            //if (zm < 0 || zm > nz_m) continue;
            %s
          }
        }
      }
    }
  }
}
"""

DataCubeRefineCoarseData = _baseDataCubeRefineCoarseData % \
 ("cubeData(xm,ym,zm) = fieldData(xg,yg,zg);")
DataCubeReplaceData = _baseDataCubeRefineCoarseData % \
 ("fieldData(xg,yg,zg) = cubeData(xm,ym,zm);")
