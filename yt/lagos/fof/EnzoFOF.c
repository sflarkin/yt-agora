/************************************************************************
* Copyright (C) 2008-2009 Matthew Turk.  All Rights Reserved.
*
* This file is part of yt.
*
* yt is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
************************************************************************/

//
// EnzoFOF
//   A module for running friends-of-friends halo finding on a set of particles
//

#include "Python.h"
#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <ctype.h>
#include "kd.h"
#include "tipsydefs.h"

#include "numpy/ndarrayobject.h"


static PyObject *_FOFerror;

static PyObject *
Py_EnzoFOF(PyObject *obj, PyObject *args)
{
    PyObject    *oxpos, *oypos, *ozpos;

    PyArrayObject    *xpos, *ypos, *zpos;
    xpos=ypos=zpos=NULL;
    float link = 0.2;

    int i;

    if (!PyArg_ParseTuple(args, "OOO|f",
        &oxpos, &oypos, &ozpos, &link))
    return PyErr_Format(_FOFerror,
            "EnzoFOF: Invalid parameters.");

    /* First the regular source arrays */

    xpos    = (PyArrayObject *) PyArray_FromAny(oxpos,
                    PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                    NPY_INOUT_ARRAY | NPY_UPDATEIFCOPY, NULL);
    if(!xpos){
    PyErr_Format(_FOFerror,
             "EnzoFOF: xpos didn't work.");
    goto _fail;
    }
    int num_particles = PyArray_SIZE(xpos);

    ypos    = (PyArrayObject *) PyArray_FromAny(oypos,
                    PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                    NPY_INOUT_ARRAY | NPY_UPDATEIFCOPY, NULL);
    if((!ypos)||(PyArray_SIZE(ypos) != num_particles)) {
    PyErr_Format(_FOFerror,
             "EnzoFOF: xpos and ypos must be the same length.");
    goto _fail;
    }

    zpos    = (PyArrayObject *) PyArray_FromAny(ozpos,
                    PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                    NPY_INOUT_ARRAY | NPY_UPDATEIFCOPY, NULL);
    if((!zpos)||(PyArray_SIZE(zpos) != num_particles)) {
    PyErr_Format(_FOFerror,
             "EnzoFOF: xpos and zpos must be the same length.");
    goto _fail;
    }

    /* let's get started with the FOF stuff */

	KD kd;
	int nBucket,j;
	float fPeriod[3],fEps;
	int nMembers,nGroup,bVerbose=1;
	int sec,usec;
	
	/* linking length */
	fprintf(stdout, "Link length is %f\n", link);
	fEps = link;
	
	nBucket = 16;
	nMembers = 8;

	for (j=0;j<3;++j) fPeriod[j] = 1.0;

    /* initialize the kd FOF structure */

	kdInit(&kd,nBucket,fPeriod);
	
	/* kdReadTipsy(kd,stdin,bDark,bGas,bStar); */

 	/* Copy positions into kd structure. */

    fprintf(stdout, "Filling in %d particles\n", num_particles);
    kd->nActive = num_particles;
	kd->p = (PARTICLE *)malloc(kd->nActive*sizeof(PARTICLE));
	assert(kd->p != NULL);
	for (i = 0; i < num_particles; i++) {
	  kd->p[i].iOrder = i;
	  kd->p[i].r[0] = (float)(*(npy_float64*) PyArray_GETPTR1(xpos, i));
	  kd->p[i].r[1] = (float)(*(npy_float64*) PyArray_GETPTR1(ypos, i));
	  kd->p[i].r[2] = (float)(*(npy_float64*) PyArray_GETPTR1(zpos, i));
	}
	
	kdBuildTree(kd);
	kdTime(kd,&sec,&usec);
	nGroup = kdFoF(kd,fEps);
	kdTime(kd,&sec,&usec);
	if (bVerbose) printf("Number of initial groups:%d\n",nGroup);
	nGroup = kdTooSmall(kd,nMembers);
	if (bVerbose) {
		printf("Number of groups:%d\n",nGroup);
		printf("FOF CPU TIME: %d.%06d secs\n",sec,usec);
		}
	kdOrder(kd);

	/* kdOutGroup(kd,ach); */
	
    // Now we need to get the groupID, realID.
    // This will give us the index into the original array.
    // Additionally, note that we don't really need to tie the index
    // back to the ID in this code, as we can do that back in the python code.
    // All we need to do is group information.
    
    // Tags are in kd->p[i].iGroup
    PyArrayObject *particle_group_id = (PyArrayObject *)
            PyArray_SimpleNewFromDescr(1, PyArray_DIMS(xpos),
                    PyArray_DescrFromType(NPY_INT32));
    
    for (i = 0; i < num_particles; i++) {
      // group tag is in kd->p[i].iGroup
      *(npy_int32*)(PyArray_GETPTR1(particle_group_id, i)) =
            (npy_int32) kd->p[i].iGroup;
    }

	kdFinish(kd);

    PyArray_UpdateFlags(particle_group_id, NPY_OWNDATA | particle_group_id->flags);
    PyObject *return_value = Py_BuildValue("N", particle_group_id);

    Py_DECREF(xpos);
    Py_DECREF(ypos);
    Py_DECREF(zpos);

    /* We don't need this, as it's done in kdFinish
    if(kd->p!=NULL)free(kd->p);
    */

    return return_value;

_fail:
    Py_XDECREF(xpos);
    Py_XDECREF(ypos);
    Py_XDECREF(zpos);

    if(kd->p!=NULL)free(kd->p);

    return NULL;

}

static PyMethodDef _FOFMethods[] = {
    {"RunFOF", Py_EnzoFOF, METH_VARARGS},
    {NULL, NULL} /* Sentinel */
};

/* platform independent*/
#ifdef MS_WIN32
__declspec(dllexport)
#endif

void initEnzoFOF(void)
{
    PyObject *m, *d;
    m = Py_InitModule("EnzoFOF", _FOFMethods);
    d = PyModule_GetDict(m);
    _FOFerror = PyErr_NewException("EnzoFOF.FOFerror", NULL, NULL);
    PyDict_SetItemString(d, "error", _FOFerror);
    import_array();
}

/*
 * Local Variables:
 * mode: C
 * c-file-style: "python"
 * End:
 */
