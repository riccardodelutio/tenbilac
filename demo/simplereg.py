"""
This demo tests the network by asking for a plain regular regression (not inverse),
in 1D.

"""

import numpy as np
import tenbilac
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

n = 200 # Size of training set
noise_scale = 0.05
ntest = 100 # Number of points to draw the regression line

params = np.linspace(0.0, 10.0, n).reshape((1, n))
obs = np.sin(params)/(1.0 + params) + noise_scale*np.random.randn(n).reshape((1, n))

obs_normer = tenbilac.data.Normer(obs)
params_normer = tenbilac.data.Normer(params)

normobs = obs_normer(obs)
normparams = params_normer(params)

testparams = np.linspace(-1.0, 11, ntest).reshape((1, ntest))
normtestparams = params_normer(testparams)

net = tenbilac.net.Net(1, [3])
net.addnoise(wscale=0.3, bscale=0.3)

# We train this normal (non-inverse) regression with params as inputs, and observations as output:
net.train(normparams, normobs, tenbilac.err.mse, maxiter=500)

# Predicting the testparams
normtestpreds = net.run(normtestparams)
testpreds = obs_normer.denorm(normtestpreds)


fig = plt.figure(figsize=(6, 4))

ax = fig.add_subplot(1, 1, 1)
ax.plot(params.T, obs.T, "b.")
ax.plot(testparams.T, testpreds.T, "r-")
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y = f(x) + \mathrm{noise}$", fontsize=18)

plt.tight_layout()
plt.show()	

