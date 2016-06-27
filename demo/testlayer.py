
import numpy as np
import tenbilac


import logging
logging.basicConfig(level=logging.INFO)



layer = tenbilac.layer.Layer(ni=2, nn=3, actfct=tenbilac.act.iden)

#layer.addnoise()

# 3D case (i.e., with several realizations)

layer.weights[0, 0] = 1
layer.weights[0, 1] = 2


layer.weights[2, 0] = 3
layer.weights[2, 1] = 0
layer.biases[1] = 42


print layer.report()

# 2 features, 2 galaxies, 4 realizations:
# galaxy_1 is (1, -1), galaxy_2 is (4, -4)
# order of indices: realization, feature, galaxy
input = np.array([
[[1, 4], [-1, -4]],
[[1.1, 4], [-1.3, -4]],
[[1, 4], [-1, -4]],
[[1, 4], [-1, -4]]
])
# Checking the galaxy 1 and 2:
print input[0, 0, 0], input[0, 1, 0]
print input[0, 0, 1], input[0, 1, 1]


print input.shape
#print input.ndim


output = layer.run(input)


print "ouput = ", output
print output.shape


# 2 D case:
"""
layer.weights[1, 1] = 1
layer.biases[2] = 10
print layer.report()

# 2 features, 4 galaxies. Each galaxy is (1, 2)
input = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
#input = np.array([1, 2])

print "input = ", input
output = layer.run(input)
print "ouput = ", output
"""
