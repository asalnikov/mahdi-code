# -*- coding: utf-8 -*-

import imp
import os

#print "metrics: '%s'" % __path__

MODULE_EXTENSIONS = ('.py', '.pyc', '.pyo')

__all__=set()

#print os.listdir(__path__[0])

for name in os.listdir(__path__[0]):
    #print "now: '%s'" % name
    if name.endswith(MODULE_EXTENSIONS) and name.startswith("metric_"):
        module_name=name.rpartition('.')[0]
        #print module_name
        __all__.add(module_name)

