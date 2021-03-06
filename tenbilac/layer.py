"""
A layer holds the parameters (weights and biases) to compute the ouput of each of its neurons based on the input of the previous layer.
This means that the "input" of a neural network is *not* such a Layer object.
The "first" real Layer in a network is in fact the first hidden layer.

"""


import numpy as np

import logging
logger = logging.getLogger(__name__)

from . import act
from . import der_act

		
class Layer():	
	def __init__(self, ni, nn, actfct=act.tanh, name="None"):
		
		"""
		:param ni: Number of inputs
		:param nn: Number of neurons
		:param actfct: Activation function
		"""

		self.ni = ni
		self.nn = nn
		self.actfct = actfct
		self.deract = der_act.sech2 #Ideally this will depend on the param actfct, TO BE COMPLETED!
		self.name = name
		
		self.weights = np.zeros((self.nn, self.ni)) # first index is neuron, second is input
		self.biases = np.zeros(self.nn) # each neuron has its bias
		
	
	def addnoise(self, wscale=0.1, bscale=0.1):
		"""
		Adds some noise to weights and biases
		"""
		self.weights += wscale * np.random.randn(self.weights.size).reshape(self.weights.shape)
		self.biases += bscale * np.random.randn(self.biases.size)
	
	def zero(self):
		"""
		Sets all weights and biases to zero
		"""
		logger.info("Setting layer '{self.name}' parameters to zero...".format(self=self))
		self.weights *= 0.0
		self.biases *= 0.0
	
	
	def report(self):
		"""
		Returns a text about the weights and biases, useful for debugging
		"""
		
		txt = []
		txt.append("Layer '{name}':".format(name=self.name))
		for inn in range(self.nn):
			txt.append("    output {inn} = {act} ( input * {weights} + {bias} )".format(
				inn=inn, act=self.actfct.__name__, weights=self.weights[inn,:], bias=self.biases[inn]))
		return "\n".join(txt)
	
	
	def nparams(self):
		"""
		Retruns the number of parameters of this layer
		"""
		return self.nn * (self.ni + 1)
	
	
	def run(self, inputs):
		"""
		Computes output from input, as "numpyly" as possible, using only np.dot (note that np.ma.dot does not
		work with 3D masked arrays, and np.tensordot seems not available for masked arrays.
		This means that np.dot determines the order of indices, as following.
		
		If inputs is 1D,
			the input is just an array of features for a single case
			the output is a 1D array with the output of each neuron
			
		If inputs is 2D,
			input indices: (feature, case)
			output indices: (neuron, case) -> so this can be fed into the next layer...
		
		If inputs is 3D, 
			input indices: (realization, feature, case)
			output indices: (realization, neuron, case)
			
		"""
		
		if inputs.ndim == 1:
			return self.actfct(np.dot(self.weights, inputs) + self.biases)
		
		elif inputs.ndim == 2:
			assert inputs.shape[0] == self.ni		
			return self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1)))
		
		elif inputs.ndim == 3:
			assert inputs.shape[1] == self.ni
			
			# Doing this:
			# self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1, 1)))
			# ... gives ouput indices (neuron, realization, case)
			# We need to change the order of those indices:
			
			return np.rollaxis(self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1, 1))), 1)
		
			# Note that np.ma.dot does not work for 3D arrays!
			# We do not care about masks at all here, just compute assuming nothing is masked.
		
		else:
			raise RuntimeError("Input shape error")
		
		
	
	def derivative_run(self, inputs):
		"""
		Necessary for backpropagation
		
		TO BE COMPLETED ! 
		"""
		
		if inputs.ndim == 3:
			assert inputs.shape[1] == self.ni
			return np.rollaxis(self.deract(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1, 1))),1)
		else:
			raise RuntimeError("Only 3D inputs impleted for backpropagation")