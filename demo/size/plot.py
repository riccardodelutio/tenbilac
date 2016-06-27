import numpy as np
import tenbilac
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

(obs_normer, params_normer, normobs, normparams, normuniparams, normuniobs, normtestobs) = tenbilac.utils.readpickle("data.pkl")

# One one plot, we show the actual training data points:
trainparams = params_normer.denorm(normparams) # 2D
trainobs = obs_normer.denorm(normobs) # 3D

# We modify those arrays to get everything into flat 1D arrays
trainparams = np.concatenate(trainobs.shape[0]*[trainparams.T]).T[0]
trainobs = trainobs.ravel()

# Showing all of them looks ugly, just a few:
showtrainindices = np.arange(trainparams.size)
np.random.shuffle(showtrainindices)
showtrainindices = showtrainindices[:10000]


uniparams =  params_normer.denorm(normuniparams)
testobs = obs_normer.denorm(normtestobs)

train_msrb = tenbilac.utils.readpickle("train_msrb.pkl")
testpreds_msrb = params_normer.denorm(train_msrb.net.run(normtestobs))
preds_msrb = params_normer.denorm(train_msrb.net.run(normuniobs))
firstreapreds_msrb = params_normer.denorm(train_msrb.net.run(normuniobs[0]))
biases_msrb = np.mean(preds_msrb, axis=0) - uniparams


train_mse = tenbilac.utils.readpickle("train_mse.pkl")
testpreds_mse = params_normer.denorm(train_mse.net.run(normtestobs))
preds_mse = params_normer.denorm(train_mse.net.run(normuniobs))
firstreapreds_mse = params_normer.denorm(train_mse.net.run(normuniobs[0]))
biases_mse = np.mean(preds_mse, axis=0) - uniparams


train_avg = tenbilac.utils.readpickle("train_avg.pkl")
testpreds_avg = params_normer.denorm(train_avg.net.run(normtestobs))
preds_avg = params_normer.denorm(train_avg.net.run(normuniobs))
firstreapreds_avg = params_normer.denorm(train_avg.net.run(normuniobs[0]))
biases_avg = np.mean(preds_avg, axis=0) - uniparams


trutheta = np.linspace(0.0, 2.2, 100)
trud = np.sqrt(4.0 + np.square(trutheta))

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(2, 2, 1)
ax.plot(trainparams[showtrainindices], trainobs[showtrainindices], marker=".", color="gray", ls="None", ms=2)
ax.plot(testpreds_mse.T, testobs.T, "r-", label="Standard MSE", lw=1.5)
ax.plot(testpreds_avg.T, testobs.T, "b-", label=r"Learning on $< d >$", lw=1.5)
ax.plot(testpreds_msrb.T, testobs.T, "g-", label="Tenbilac MSRB", lw=1.5)
ax.plot(trutheta, trud, "r-", color="black", dashes=(5, 5), lw=1.5, label=r"$d = \sqrt{2^2 + \theta^2}$(truth)")
ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)
ax.set_xlim(-1.2, 2.4)
ax.set_ylim(1.6, 3.1)
ax.legend(loc=2)


ax = fig.add_subplot(2, 2, 2)
ax.plot(uniparams.T, biases_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(uniparams.T, biases_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(uniparams.T, biases_msrb.T, marker=".", color="green", ls="None", ms=2)
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.4, 0.4)


ax = fig.add_subplot(2, 2, 3)
ax.plot(uniparams.T, firstreapreds_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(uniparams.T, firstreapreds_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(uniparams.T, firstreapreds_msrb.T, marker=".", color="green", ls="None", ms=2)
ax.plot([0, 2], [0, 2], color="black", dashes=(5, 5), lw=1.5)
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$\hat{\theta}$", fontsize=18)
ax.set_xlim(-0.2, 2.2)
ax.set_ylim(-3.0, 2.5)



ax = fig.add_subplot(2, 2, 4)
ax.plot(firstreapreds_mse.T, biases_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(firstreapreds_avg.T, biases_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(firstreapreds_msrb.T, biases_msrb.T, marker=".", color="green", ls="None", ms=2)
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)
ax.set_xlim(-1.7, 2.5)
ax.set_ylim(-0.4, 0.4)




plt.tight_layout()
plt.show()	

