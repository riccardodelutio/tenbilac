import numpy as np

"""
Necessary for Backpropagation
ideally will define a class or a function that defines the derivative
of the corresponding activation function automatically

TO BE COMPLETESD !
"""

def sech2(x):
	"""
	Derivative of the default activation function tanh
	"""
	
	return (1.0/np.cosh(x))**2