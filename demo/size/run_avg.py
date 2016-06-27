
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


(obs_normer, params_normer, normobs, normparams, normuniparams, normuniobs, normtestobs) = tenbilac.utils.readpickle("data.pkl")

# We average the training observations over the realizations:
normobs = np.mean(normobs, axis=0).reshape(1, 1, normobs.shape[2])

traindata = tenbilac.data.Traindata(normobs, normparams, valfrac=0.5, shuffle=True)

net = tenbilac.net.Net(1, [7])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)

train = tenbilac.train.Training(net, traindata, errfctname="mse")

train.minibatch_bfgs(mbsize=100, mbloops=5, maxiter=50)

#tenbilac.plot.sumevo(train)

train.save("train_avg.pkl", keepdata=True)



