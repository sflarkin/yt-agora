"""
Skeleton objects that represent a few fundamental yt data types.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import abc
import json
import urllib2
from tempfile import TemporaryFile
from yt.config import ytcfg
from yt.funcs import *
from yt.utilities.exceptions import *

from .poster.streaminghttp import register_openers
from .poster.encode import multipart_encode
register_openers()

class UploaderBar(object):
    pbar = None
    def __init__(self, my_name = ""):
        self.my_name = my_name

    def __call__(self, name, prog, total):
        if self.pbar is None:
            self.pbar = get_pbar("Uploading %s " % self.my_name, total)
        self.pbar.update(prog)
        if prog == total:
            self.pbar.finish()

class ContainerClass(object):
    pass

class MinimalRepresentation(object):
    __metaclass__ = abc.ABCMeta

    def _update_attrs(self, obj, attr_list):
        for attr in attr_list:
            setattr(self, attr, getattr(obj, attr, None))
        if hasattr(obj, "pf"):
            self.output_hash = obj.pf._hash()
            self._pf_mrep = obj.pf._mrep

    def __init__(self, obj):
        self._update_attrs(obj, self._attr_list)

    @abc.abstractmethod
    def _generate_post(self):
        pass

    @abc.abstractproperty
    def _attr_list(self):
        pass

    def _return_filtered_object(self, attrs):
        new_attrs = tuple(attr for attr in self._attr_list
                          if attr not in attrs)
        new_class = type('Filtered%s' % self.__class__.__name__,
                         (FilteredRepresentation,),
                         {'_attr_list': new_attrs})
        return new_class(self)

    @property
    def _attrs(self):
        return dict( ((attr, getattr(self, attr)) for attr in self._attr_list) )

    @classmethod
    def _from_metadata(cls, metadata):
        cc = ContainerClass()
        for a, v in metadata.values():
            setattr(cc, a, v)
        return cls(cc)

    def upload(self):
        api_key = ytcfg.get("yt","hub_api_key")
        url = ytcfg.get("yt","hub_url")
        if api_key == '': raise YTHubRegisterError
        metadata, (final_name, chunks) = self._generate_post()
        if hasattr(self, "_pf_mrep"):
            self._pf_mrep.upload()
        for i in metadata:
            if isinstance(metadata[i], np.ndarray):
                metadata[i] = metadata[i].tolist()
            elif hasattr(metadata[i], 'dtype'):
                metadata[i] = np.asscalar(metadata[i])
        metadata['obj_type'] = self.type
        if len(chunks) == 0:
            chunk_info = {'chunks': []}
        else:
            chunk_info = {'final_name' : final_name, 'chunks': []}
            for cn, cv in chunks:
                chunk_info['chunks'].append((cn, cv.size * cv.itemsize))
        metadata = json.dumps(metadata)
        chunk_info = json.dumps(chunk_info)
        datagen, headers = multipart_encode({'metadata' : metadata,
                                             'chunk_info' : chunk_info,
                                             'api_key' : api_key})
        request = urllib2.Request(url, datagen, headers)
        # Actually do the request, and get the response
        try:
            rv = urllib2.urlopen(request).read()
        except urllib2.HTTPError as ex:
            if ex.code == 401:
                mylog.error("You must create an API key before uploading.")
                mylog.error("https://data.yt-project.org/getting_started.html")
                return
            else:
                raise ex
        uploader_info = json.loads(rv)
        new_url = url + "/handler/%s" % uploader_info['handler_uuid']
        for i, (cn, cv) in enumerate(chunks):
            remaining = cv.size * cv.itemsize
            f = TemporaryFile()
            np.save(f, cv)
            f.seek(0)
            pbar = UploaderBar("%s, % 2i/% 2i" % (self.type, i+1, len(chunks)))
            datagen, headers = multipart_encode({'chunk_data' : f}, cb = pbar)
            request = urllib2.Request(new_url, datagen, headers)
            rv = urllib2.urlopen(request).read()

        datagen, headers = multipart_encode({'status' : 'FINAL'})
        request = urllib2.Request(new_url, datagen, headers)
        rv = json.loads(urllib2.urlopen(request).read())
        mylog.info("Upload succeeded!  View here: %s", rv['url'])
        return rv

class FilteredRepresentation(MinimalRepresentation):
    def _generate_post(self):
        raise RuntimeError

class MinimalStaticOutput(MinimalRepresentation):
    _attr_list = ("dimensionality", "refine_by", "domain_dimensions",
                  "current_time", "domain_left_edge", "domain_right_edge",
                  "unique_identifier", "current_redshift", "output_hash",
                  "cosmological_simulation", "omega_matter", "omega_lambda",
                  "hubble_constant", "name")
    type = 'simulation_output'

    def __init__(self, obj):
        super(MinimalStaticOutput, self).__init__(obj)
        self.output_hash = obj._hash()
        self.name = str(obj)

    def _generate_post(self):
        metadata = self._attrs
        chunks = []
        return (metadata, (None, chunks))

class MinimalMappableData(MinimalRepresentation):

    _attr_list = ("field_data", "field", "weight_field", "axis", "output_hash",
                  "vm_type")

    def _generate_post(self):
        nobj = self._return_filtered_object(("field_data",))
        metadata = nobj._attrs
        chunks = [(arr, self.field_data[arr]) for arr in self.field_data]
        return (metadata, ('field_data', chunks))

class MinimalProjectionData(MinimalMappableData):
    type = 'proj'
    vm_type = "Projection"

class MinimalSliceData(MinimalMappableData):
    type = 'slice'
    vm_type = "Slice"
    weight_field = "None"

class MinimalImageCollectionData(MinimalRepresentation):
    type = "image_collection"
    _attr_list = ("name", "output_hash", "images", "image_metadata")

    def _generate_post(self):
        nobj = self._return_filtered_object(("images",))
        metadata = nobj._attrs
        chunks = [(fn, d) for fn, d in self.images]
        return (metadata, ('images', chunks))

_hub_categories = ("News", "Documents", "Simulation Management",
                   "Data Management", "Analysis and Visualization",
                   "Paper Repositories", "Astrophysical Utilities",
                   "yt Scripts")

class MinimalProjectDescription(MinimalRepresentation):
    type = "project"
    _attr_list = ("title", "url", "description", "category", "image_url")

    def __init__(self, title, url, description,
                 category, image_url = ""):
        assert(category in _hub_categories)
        self.title = title
        self.url = url
        self.description = description
        self.category = category
        self.image_url = image_url

    def _generate_post(self):
        metadata = self._attrs
        chunks = []
        return (metadata, ("chunks", []))

class MinimalNotebook(MinimalRepresentation):
    type = "notebook"
    _attr_list = ("title",)

    def __init__(self, filename, title = None):
        # First we read in the data
        if not os.path.isfile(filename):
            raise IOError(filename)
        self.data = open(filename).read()
        if title is None:
            title = json.loads(self.data)['metadata']['name']
        self.title = title
        self.data = np.fromstring(self.data, dtype='c')

    def _generate_post(self):
        metadata = self._attrs
        chunks = [ ("notebook", self.data) ]
        return (metadata, ("chunks", chunks))
