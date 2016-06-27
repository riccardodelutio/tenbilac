"""
Error functions
These should work both on unmasked and on masked arrays "as expected".

The typical call is fct(predictions, targets, auxinputs), where auxinputs can be left out if the error function does not use any auxinputs.
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)



	
def msb(predictions, targets, auxinputs=None):
	"""
	Mean square bias
	
	:param predictions: 3D array (realization, neuron, case), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, case)
	
	"""
	
	if predictions.ndim == 3:
	
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, case)
		return np.mean(np.square(biases))
	
	else:
		raise ValueError("Wrong pred shape")
		
		
		

def ssb(predictions, targets, auxinputs=None):
	"""
	Sum square bias
	
	The main purpose of this cost function is to simplify the results when used for backpropagation
	
	:param predictions: 3D array (realization, neuron, case), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, case)
	
	"""
	
	if predictions.ndim == 3:
	
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, case)
		return 0.5*np.sum(np.square(biases))
	
	else:
		raise ValueError("Wrong pred shape")

		
		

def mab(predictions, targets, auxinputs=None):
	"""
	Mean absolute bias
	
	:param predictions: 3D array (realization, neuron, case), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, case)
	
	"""
	
	if predictions.ndim == 3:
	
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, case)
		return np.mean(np.fabs(biases))
	
	else:
		raise ValueError("Wrong pred shape")



	
def msrb(predictions, targets, auxinputs=None, rawterms=False):
	"""
	Mean square relative bias
	
	:param predictions: 3D array (realization, neuron, case), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, case)
	
	:param rawterms: if True, returns the "RB" of "MSRB" as (potentially masked) 2D array.
	
	"""
	
	if predictions.ndim == 3:
	
		#assert type(predictions) == np.ma.MaskedArray
		# Just as a test that this was not forgotten, no real need here.
		# No need, for this, for sure it also works without masks...
		
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, case) : masked, but probably all masks are False.
		stds = np.std(predictions, axis=0) # idem
		
		if type(predictions) == np.ma.MaskedArray:
			reacounts = predictions.shape[0] - np.sum(predictions.mask, axis=0) + 0.0 # Number of realizations, 0.0 makes this floats # This is 2D (label, case)
		elif type(predictions) == np.ndarray:
			reacounts = predictions.shape[0] + 0.0
		else:
			raise RuntimeError("Type error in predictions.")
		
		errsonbiases = stds / np.sqrt(reacounts)
		relativebiases = biases / errsonbiases
		
		if rawterms:
			#return (biases, errsonbiases)
			return relativebiases # 2D (label, case), masked, but probably all masks are False.
		else:
			return np.mean(np.square(relativebiases))
		
	
	else:
		raise ValueError("Wrong pred shape")

	
	


def mse(predictions, targets, auxinputs=None):
	"""
	Standard MSE (mean square error), simply treats multiple realizations as if they were independent cases	
	
	:param predictions: 2D array (neuron, case) or 3D array (realization, neuron, case)
	:param targets: 2D array (neuron, case)

	"""
	
	# This same code works for 2D or 3D predictions:
	return np.mean(np.square(predictions - targets))
	
	#return np.mean(np.square(net.run(inputs[0]) - targets))



def msre(predictions, targets, auxinputs=None):
	"""
	MSE with normalization by the scatter along the realizations (as MSRB is for MSB)

	Mean Square Error / Variance - 1

	"""

	if predictions.ndim == 3:
		
		stds = np.std(predictions, axis=0) # 2D, (label, case)

		return np.mean(np.square((predictions - targets) / stds) - 1.0) # The minus one helps to viz it on a log scale...
	
	else:
		raise ValueError("Wrong pred shape")




def msbwnet(predictions, targets, auxinputs=None):
	"""
	Mean square bias with weights, for training a WNet.
	This is the first errorfunction of a new type, it compares the weighted average of the predictions with the targets.
	
	There should be twice more prediction "neurons" than targets. The second half of the predictions is interpreted as weights
	
	:param predictions: 3D array (realization, neuron, case), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, case)
	
	"""
	
	if predictions.ndim == 3:
	
		nt = targets.shape[0] # the number of targets = number of "predicted parameters" = number of weights for these outputs
		assert predictions.shape[1] == 2 * nt # Indeed, these are the predicted parameters and the corresponding weights.
		
		predparams = predictions[:,:nt,:]
		predweights = 10**predictions[:,nt:,:]
		assert predparams.shape == predweights.shape
	
		biases = np.mean(predparams * predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
	
		return np.mean(np.square(biases))
	
	else:
		raise ValueError("Wrong pred shape")
	


def msbpw(predictions, targets, auxinputs):
	"""
	with pre-weights
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	biases = np.mean(auxinputs * predictions, axis=0) / np.mean(auxinputs, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
	
	return np.mean(np.square(biases))


def msbw(predictions, targets, auxinputs):
	"""
	This errorfct is for Nets predicting weights only. It expresses the mean square weighted bias of the auxinputs
	
	The predictions should be masked, auxinputs as well. Targets are usually not masked.
	"""
	
	#logger.warning("Normalizes the weight, probably a bad idea !")
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	predweights = 10**predictions
	biases = np.mean(auxinputs * predweights, axis=0) / np.mean(predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))
	
	
def msbwnonorm(predictions, targets, auxinputs):
	"""
	That normalization was a very bad idea...
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	predweights = 10**predictions
	biases = np.mean(auxinputs * predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))
	
	
def msbwclip(predictions, targets, auxinputs):
	"""
	Direct, but limiting the weights to be positive.
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	predweights = np.clip(predictions, 1e-6, 1e15) # need large uppper limit, otherwise optimizer stops
	biases = np.mean(auxinputs * predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))


def msbwsquare(predictions, targets, auxinputs):
	"""
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	predweights = predictions**2
	biases = np.mean(auxinputs * predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))


def msbwinvsquare(predictions, targets, auxinputs):
	"""
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	predweights = 1.0/predictions**2
	biases = np.mean(auxinputs * predweights, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))







def msbwsig(predictions, targets, auxinputs):
	"""
	Similar as msbw, but we do not use this dangerous power of 10, and we do not normalize the weights at training
	
	This is for networks whose ouput is from a sigmoid layer, so positive by construction (and between 0 and 1).
	Weights are 1 * output.
	We do NOT normalize the weighted average auxinputs
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	biases = np.mean(auxinputs * 1.0*predictions, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))
	

def msbwsigtwo(predictions, targets, auxinputs):
	"""
	Similar as msbw, but we do not use this dangerous power of 10, and we do not normalize the weights at training
	
	This is for networks whose ouput is from a sigmoid layer, so positive by construction (and between 0 and 1).
	Weights are 2 * output.
	We do NOT normalize the weighted average auxinputs
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	biases = np.mean(auxinputs * 2.0*predictions, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))
	
def msbwsigten(predictions, targets, auxinputs):
	"""
	Similar as msbw, but we do not use this dangerous power of 10, and we do not normalize the weights at training
	
	This is for networks whose ouput is from a sigmoid layer, so positive by construction (and between 0 and 1).
	Weights are 10 * output.
	We do NOT normalize the weighted average auxinputs
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	biases = np.mean(auxinputs * 10.0*predictions, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))

def msbwsigthree(predictions, targets, auxinputs):
	"""
	Similar as msbw, but we do not use this dangerous power of 10, and we do not normalize the weights at training
	
	This is for networks whose ouput is from a sigmoid layer, so positive by construction (and between 0 and 1).
	Weights are 3 * output.
	We do NOT normalize the weighted average auxinputs
	"""
	
	assert predictions.ndim == 3
	assert auxinputs.ndim == 3
	assert targets.ndim == 2
	
	nt = targets.shape[0] # the number of targets = number of "predicted weights" = number of aux inputs (per case)
	assert auxinputs.shape[1] == nt
	assert predictions.shape[1] == nt 
	
	biases = np.mean(auxinputs * 3.0*predictions, axis=0) - targets # The mean is done along realizations, so this is 2D, (label, case)
		
	return np.mean(np.square(biases))
