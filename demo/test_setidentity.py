
import numpy as np
import tenbilac


import logging
logging.basicConfig(level=logging.INFO)


ni = 4
ngal = 1
nrea = 2
no = 4


net = tenbilac.net.Net(ni=ni, nhs=[4, 6, 4], no=no, onlyid=True)

net.setidentity()

print net.report()


inputs = np.random.randn(nrea * ni * ngal).reshape((nrea, ni, ngal))

# Setting the first rea input to some well-known stuff:
inputs[0,0,:] = 0.8
inputs[0,1,:] = 0.3
inputs[0,2,:] = 0.1
inputs[0,3,:] = -0.3


#inputs *= 50.0 
#inputs += 1000.0


print "inputs shape", inputs.shape
print inputs

outputs = net.run(inputs)

print "outputs shape", outputs.shape
print outputs
