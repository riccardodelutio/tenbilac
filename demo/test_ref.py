
import numpy as np
import tenbilac


import logging
logging.basicConfig(level=logging.INFO)



net = tenbilac.net.Net(ni=2, nhs=[3, 3], onlyid=True)
net.setidentity()

params = net.get_params_ref()

print params

net.addnoise()

print params

net.setidentity()

print params

print net.report()

