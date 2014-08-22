from xml.dom.minidom import parse
import ast

def lambda_bool(x):
      if type(x) == bool:
            return x
      elif x == 'True' or x == 'true':
            return True
      else:
            return False

INT = lambda x: int(x)
LONG = lambda x: long(x)
FLOAT = lambda x: float(x)
STR = lambda x: str(x)
REPR = lambda x: repr(x)
BOOL = lambda_bool
EVAL = lambda x: eval(x) 
TUPLE = lambda x: tuple(x) 
LIST = lambda x: ast.literal_eval(str(x))
NPLIST = lambda x: LIST(list(x))
SET = lambda x: set(x) 
DICT = lambda x: dict(x) 
FROZENSET = lambda x: frozenset(x)
CHR = lambda x: chr(x)
UNICHR = lambda x: unichr(x)
ORD = lambda x: ord(x)
HEX = lambda x: hex(x)
OCT = lambda x: oct(x)

def json2dict(dict, dict_types):
      for k in dict:
            try:
                  dict[k] = dict_types[k](dict[k])
            except KeyError:
                  continue
      return dict

def dict2json(dict):
      import json
      json = '{'
      for k in dict:
            val = ""
            if dict[k] == None:
                  val = "\"\""
            elif type(dict[k]) == str or type(dict[k]) == unicode:
                  val += '\"' + dict[k] + '\"'
            elif type(dict[k]) == bool:
                  if dict[k]:
                        val = "true"
                  else:
                        val = "false"
            elif type(dict[k]) == list:
                  val = "[%s]" % ", ".join(map(lambda x: '"%s"' % str(x),dict[k])) # to force double quotes within list [""]
            else:
                  val = str(dict[k])
            json += '\"' + str(k) + '\":' + val + ','
      return json[:-1] + '}'

def dict2xml(dict, dict_types):
      dict['xml_save'] = False
      xml = "<options>\n"
      for k in dict:
            try:
                  xml += "\t<%s>%s</%s>\n" % (str(k), str(dict_types[k](dict[k])), str(k))
            except KeyError:
                  xml += "\t<%s>%s</%s>\n" % (str(k), str(dict[k]), str(k))
      xml += "</options>"
      return xml

def xml2dict(xml, dict_types):
      dict = {}
      nodes = parse(xml).documentElement.childNodes
      for node in nodes:
            if node.nodeName[0] == '#':
                  continue
            try:
                  if dict_types[node.nodeName] == NPLIST:
                        dict[node.nodeName] = LIST(node.firstChild.nodeValue)
                  else:
                        dict[node.nodeName] = dict_types[node.nodeName](node.firstChild.nodeValue)
            except KeyError:
                  try:
                        dict[node.nodeName] = node.firstChild.nodeValue
                  except AttributeError:
                        dict[node.nodeName] = None
            except AttributeError:
                  dict[node.nodeName] = None
                  
      return dict
