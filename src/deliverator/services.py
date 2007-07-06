################################################## 
# raven_services.py 
# generated by ZSI.generate.wsdl2python
##################################################


from services_types import *
import urlparse, types
from ZSI.TCcompound import ComplexType, Struct
from ZSI import client
import ZSI

# Locator
class DeliveratorServerLocator:
    raven_porttype_address = "http://kipac.stanford.edu/Deliverator/RavenMethods"
    def getraven_porttypeAddress(self):
        return DeliveratorServerLocator.raven_porttype_address
    def getraven_porttype(self, url=None, **kw):
        return raven_bindingSOAP(url or DeliveratorServerLocator.raven_porttype_address, **kw)

# Methods
class raven_bindingSOAP:
    def __init__(self, url, **kw):
        kw.setdefault("readerclass", None)
        kw.setdefault("writerclass", None)
        # no resource properties
        self.binding = client.Binding(url=url, **kw)
        # no ws-addressing

    # op: QueryExistingRuns
    def QueryExistingRuns(self, request):
        if isinstance(request, QueryExistingRunsInput) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        kw = {}
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="QueryExistingRuns", **kw)
        # no output wsaction
        response = self.binding.Receive(QueryExistingRunsOutput.typecode)
        return response

    # op: SubmitNewRun
    def SubmitNewRun(self, request):
        if isinstance(request, SubmitNewRunInput) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        kw = {}
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="SubmitNewRun", **kw)
        # no output wsaction
        response = self.binding.Receive(SubmitNewRunOutput.typecode)
        return response

    # op: SubmitNewParameterFile
    def SubmitNewParameterFile(self, request):
        if isinstance(request, SubmitNewParameterFileInput) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        kw = {}
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="SubmitNewParameterFile", **kw)
        # no output wsaction
        response = self.binding.Receive(SubmitNewParameterFileOutput.typecode)
        return response

    # op: SubmitNewImage
    def SubmitNewImage(self, request):
        if isinstance(request, SubmitNewImageInput) is False:
            raise TypeError, "%s incorrect request type" % (request.__class__)
        kw = {}
        # no input wsaction
        self.binding.Send(None, None, request, soapaction="SubmitNewImage", **kw)
        # no output wsaction
        response = self.binding.Receive(SubmitNewImageOutput.typecode)
        return response

QueryExistingRunsInput = ns0.QueryExistingRuns_Dec().pyclass

QueryExistingRunsOutput = ns0.QueryExistingRunsOutput_Dec().pyclass

SubmitNewRunInput = ns0.SubmitNewRun_Dec().pyclass

SubmitNewRunOutput = ns0.SubmitNewRunOutput_Dec().pyclass

SubmitNewParameterFileInput = ns0.SubmitNewParameterFile_Dec().pyclass

SubmitNewParameterFileOutput = ns0.SubmitNewParameterFileOutput_Dec().pyclass

SubmitNewImageInput = ns0.SubmitNewImage_Dec().pyclass

SubmitNewImageOutput = ns0.SubmitNewImageOutput_Dec().pyclass
