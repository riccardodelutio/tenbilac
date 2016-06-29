"""
Training a network happens here
"""

import numpy as np
import scipy.optimize
from datetime import datetime
import os
import copy

import logging
logger = logging.getLogger(__name__)

from . import layer
from . import utils
from . import err
from . import act
from . import plot



class Training:
    """
    Holds together everthing related to the process of training a Net (or a WNet): the training data and the network.
    """


    def __init__(self, net, dat, errfctname="msrb", itersavepath=None, autoplotdirpath=".", autoplot=False, verbose=False, name=None, regularization=0.0):
        """

        Sets up
        - housekeeping lists
        - error function

        """

        self.name = name

        self.set_net(net)
        self.set_dat(dat)

        # Let's check compatibility between those two!
        assert net.ni == self.dat.getni()
        if errfctname in ["mse", "msb", "msrb", "msre", "msbw"]:
            assert net.no == self.dat.getno()
        elif errfctname in ["msbwnet"]:
            assert net.no == 2*self.dat.getno()
        else:
            logger.warning("Unknown error function, will blindly go ahead...")

        # Setting up the cost function
        self.errfctname = errfctname
        self.errfct = eval("err.{0}".format(self.errfctname))

        # We initialize some counters for the optimization:
        self.optit = 0 # The iteration counter
        self.optcall = 0 # The cost function call counter
        self.optitcall = 0 # Idem, but gets reset at each new iteration
        self.opterr = np.inf # The current cost function value on the training set

        # And some lists describing the optimization:
        self.opterrs = [] # The cost function value on the training set at each (!) call

        self.optitparams = [] # A copy of the network parameters at each iteration
        self.optiterrs_train = [] # The cost function value on the training set at each iteration
        self.optiterrs_val = [] # The cost function value on the validation set at each iteration

        self.optitcalls = [] # The cost function call counter at each iteration
        self.optittimes = [] # Time taken for iteration, in seconds
        self.optbatchchangeits = [] # Iteration counter (in fact indices) when the batche gets changed

        self.regularization = regularization

        self.verbose = verbose
        self.itersavepath = itersavepath

        self.autoplotdirpath = autoplotdirpath
        self.autoplot = autoplot

        self.regularization = regularization

        logger.info("Done with setup of {self}".format(self=self))

        # And let's test this out before we start, so that it fails fast in case of a problem.
        # But we only do so if the file does not yet exist, to avoid overwriting an existing training.
        if self.itersavepath is not None:
            if not os.path.exists(self.itersavepath):
                self.save(self.itersavepath)


    def set_dat(self, dat):
        """
        Allows to add or replace training data (e.g. when reading a self.save()...)
        """
        self.dat = dat
        self.set_datstr()


    def set_datstr(self):
        """
        We do this so that we can still display this string on plots if dat has been removed.
        """
        self.datstr = str(self.dat)


    def set_net(self, net):
        """
        Replaces the network object
        """
        self.net = net
        self.params = self.net.get_params_ref() # Fast connection to the network parameters
        self.paramslice = slice(None) # By default, all params are free to be optimized

    def set_paramslice(self, mode=None):
        """
        The paramslice allows to specify which params you want to be optimized.
        This is relevant for instance when training a WNet.
        We use a slice of this. Indexing with a boolean array ("mask") would seem nicer, but fancy indexing does not preserve
        the references. Hence using a slice is a good compromise for speed.
        """
        #self.paramslice[0:self.net.neto.nparams()] = False #= self.net.get_params_ref(mode=mode)

        if mode == "o": # the slice selects only the params of the "ouputs"
            self.paramslice = slice(0, self.net.neto.nparams())
        elif mode == "w": # Idem but for the weights
            self.paramslice = slice(self.net.neto.nparams(), self.net.nparams())
        elif mode == None: # Empty slice, use all params
            self.paramslice = slice(None)
        else:
            raise ValueError("Unknown mode!")

        logger.info("Set paramslice to mode '{}' : {}/{} params are free to be optimized.".format(
                mode, len(self.params[self.paramslice]), self.net.nparams())
                )


    def __str__(self):
        """
        A short spaceless automatic description
        """
        if self.dat is not None: # If available, we use the live dat.
            datstr = str(self.dat)
        else: # Otherwise, we use datstr as backup solution.
            datstr = getattr(self, "datstr", "Ouch!") # To ensure that it works also if datstr is missing (backwards compatibility).
        autotxt = "{self.errfctname}({self.net}, {datstr})".format(self=self, datstr=datstr)
        return autotxt

    def title(self):
        """
        Returns the name and string, typically nicer for plots.
        """


        if self.name is not None:
            return "Training '{name}': {auto} ({it} it, {tmin:.1f} min)".format(name=self.name, auto=str(self), it=self.optit, tmin=np.sum(self.optittimes)/60.0)
        else:
            return "{auto} ({it} it, {tmin:.1f} min)".format(auto=str(self), it=self.optit, tmin=np.sum(self.optittimes)/60.0)


    def takeover(self, othertrain):
        """
        This copies the net and all the progress counters and logs from another train object into the current one.
        Useful if you want to carry on a training with a new train object, typically with different settings and
        maybe on different data.
        """

        logger.info("Setting up training '{}' to take over the work from '{}'...".format(self.name, othertrain.name))

        if not self.net.nparams() == othertrain.net.nparams():
            raise RuntimeError("Other network is not compatible, this is fishy!")

        self.set_net(othertrain.net)

        # Copying the counter positions:
        self.optit = othertrain.optit
        self.optcall = othertrain.optcall
        self.optitcall = 0 # Gets reset at each new iteration
        self.opterr = othertrain.opterr

        # And some lists describing the optimization:
        self.opterrs = othertrain.opterrs[:]

        self.optitparams = othertrain.optitparams[:]
        self.optiterrs_train = othertrain.optiterrs_train[:]
        self.optiterrs_val = othertrain.optiterrs_val[:]

        self.optitcalls = othertrain.optitcalls[:]
        self.optittimes = othertrain.optittimes[:]
        try:
            self.optbatchchangeits = othertrain.optbatchchangeits[:]
        except AttributeError:
            self.optbatchchangeits = []


        logger.info("Done with the takeover")



    def save(self, filepath, keepdata=False):
        """
        Saves the training progress into a pkl file
        As the training data is so massive, by default we do not save it!
        Note that this might be done at each iteration!
        """

        if keepdata is True:
            logger.info("Writing training to disk and keeping the data...")
            utils.writepickle(self, filepath)
        else:
            tmptraindata = self.dat
            self.set_datstr()
            self.dat = None
            utils.writepickle(self, filepath)
            self.dat = tmptraindata



    def makeplots(self, suffix="_optitXXXXX", dirpath=None):
        """
        Saves a bunch of default checkplots into the specified directory.
        Can typically be called at the end of training, or after iterations.

        """

        if dirpath is None:
            dirpath = self.autoplotdirpath

        if suffix == "_optitXXXXX":
            suffix = "_optit{0:05d}".format(self.optit)

        logger.info("Making and writing plots, with suffix '{}'...".format(suffix))
        plot.sumevo(self, os.path.join(dirpath, "sumevo"+suffix+".png"))
        plot.outdistribs(self, os.path.join(dirpath, "outdistribs"+suffix+".png"))
        plot.errorinputs(self, os.path.join(dirpath, "errorinputs"+suffix+".png"))
        logger.info("Done with plots")


    def start(self):
        """
        Called a the beginning of a training
        """
        self.testcost()
        self.iterationstarttime = datetime.now()
        self.optitcall = 0


    def end(self):
        """
        Called at the end of a training (each minibatch) depening on the algo.
        """
        self.optitcall = 0
        logger.info("Cumulated training time: {0:.2f} s".format(np.sum(self.optittimes)))
        if self.autoplot:
            self.makeplots()


    def callback(self, *args):
        """
        Function called by the optimizer after each "iteration".
        Print out some info about the training progress,
        saves status of the counters,
        and optionally writes the network itself to disk.
        """
        #print args
        #exit()

        self.optit += 1
        now = datetime.now()
        secondstaken = (now - self.iterationstarttime).total_seconds()
        callstaken = self.optitcall

        # Not sure if it is needed to update the params (of if the optimizer already did it), but it cannot harm and is fast:
        self.params[self.paramslice] = args[0] # Updates the network parameters

        self.optittimes.append(secondstaken)
        self.optiterrs_train.append(self.opterr)
        self.optitcalls.append(self.optcall)
        self.optitparams.append(copy.deepcopy(self.params)) # We add a copy of the current params

        # Now we evaluate the cost on the validation set:
        valerr = self.valcost()
        self.optiterrs_val.append(valerr)

        valerrratio = valerr / self.opterr

        mscallcase = 1000.0 * float(secondstaken) / (float(callstaken) * self.dat.getntrain()) # Time per call and training case

        logger.info("Iter. {self.optit:4d}, {self.errfctname} train = {self.opterr:.6e}, val = {valerr:.6e} ({valerrratio:4.1f}), {time:.4f} s for {calls} calls ({mscallcase:.4f} ms/cc)".format(
                self=self, time=secondstaken, valerr=valerr, valerrratio=valerrratio, calls=callstaken, mscallcase=mscallcase))

        if self.itersavepath != None:
            self.save(self.itersavepath)

        # We reset the iteration counters:
        self.iterationstarttime = datetime.now()
        self.optitcall = 0

        # And now we take care of getting a new batch
        #self.randombatch()



    def cost(self, p):
        """
        The "as-fast-as-possible" function to compute the training error based on parameters p.
        This gets called repeatedly by the optimizers.
        """

        self.params[self.paramslice] = p # Updates the network parameters

        # Compute the outputs
        outputs = self.net.run(self.dat.traininputs) # This is not a masked array!

        # And now evaluate the error (cost) function.
        if self.dat.trainoutputsmask is not None:
            outputs = np.ma.array(outputs, mask=self.dat.trainoutputsmask)

        err = self.errfct(outputs, self.dat.traintargets, auxinputs=self.dat.trainauxinputs) + self.regularization * np.sum(np.square(self.net.get_weights()))

        self.opterr = err
        self.optcall += 1
        self.optitcall += 1
        self.opterrs.append(err)

        if self.verbose:
            logger.debug("Iteration {self.optit:4d}, call number {self.optcall:8d}: cost = {self.opterr:.8e}".format(self=self))
            logger.debug("\n" + self.net.report())

        return err


    def currentcost(self):
        return self.cost(p=self.params[self.paramslice])

    def testcost(self):
        """
        Calls the cost function and logs some info.
        """

        logger.info("Testing cost function calls...")
        starttime = datetime.now()
        err = self.currentcost()
        endtime = datetime.now()
        took = (endtime - starttime).total_seconds()
        logger.info("On the training set:   {took:.4f} seconds, {self.errfctname} = {self.opterr:.8e}".format(self=self, took=took))
        starttime = datetime.now()
        err = self.valcost()
        endtime = datetime.now()
        took = (endtime - starttime).total_seconds()
        logger.info("On the validation set: {took:.4f} seconds, {self.errfctname} = {err:.8e}".format(self=self, took=took, err=err))



    def valcost(self):
        """
        Evaluates the cost function on the validation set.
        """
        outputs = self.net.run(self.dat.valinputs) # This is not a masked array!
        if self.dat.valoutputsmask is not None:
            outputs = np.ma.array(outputs, mask=self.dat.valoutputsmask)

        err = self.errfct(outputs, self.dat.valtargets, auxinputs=self.dat.valauxinputs) + self.regularization * np.sum(np.square(self.net.get_weights()))

        return err



    def minibatch_bfgs(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs):

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.bfgs(**kwargs)

    def bfgs(self, maxiter=100, gtol=1e-8):

        self.start()
        logger.info("Starting BFGS for {} iterations (maximum) with gtol={}...".format(maxiter, gtol))

        optres = scipy.optimize.fmin_bfgs(
                self.cost, self.params[self.paramslice],
                fprime=None,
                maxiter=maxiter, gtol=gtol,
                full_output=True, disp=True, retall=False, callback=self.callback)

        if len(optres) == 7:
            (xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag) = optres
            self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
            logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
        else:
            logger.warning("Optimization output is fishy")

        self.end()



    def minibatch_l_bfgs_b(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs): # Less effective than BFGS.. Usefull?

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.l_bfgs_b(**kwargs)

    def l_bfgs_b(self, maxiter=100, gtol=1e-8):

        self.start()
        logger.info("Starting L_BFGS_B for {} iterations (maximum) with gtol={}...".format(maxiter, gtol))

        optres = scipy.optimize.fmin_l_bfgs_b(
                self.cost, self.params[self.paramslice],
                fprime=None, args=(), approx_grad=1, bounds=None, m=10, 
                factr=10000000.0, pgtol=gtol, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=maxiter,
                disp=False, callback=self.callback, maxls=20)

        if len(optres) == 3:
            (xopt, fopt, d) = optres
            self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
            logger.info("Done with optimization, {0} func_calls".format(d['funcalls']))
        else:
            logger.warning("Optimization output is fishy")

        self.end()



    def minibatch_cg(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs): # Less effective than BFGS.. Usefull?

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.cg(**kwargs)

    def cg(self, maxiter):

        self.start()
        logger.info("Starting CG for {0} iterations (maximum)...".format(maxiter))

        optres = scipy.optimize.fmin_cg(
                self.cost, self.params,
                fprime=None, gtol=1e-05,
                maxiter=maxiter, full_output=True, disp=True, retall=False, callback=self.callback)

        if len(optres) == 5:
            (xopt, fopt, func_calls, grad_calls, warnflag) = optres
            self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
            logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
        else:
            logger.warning("Optimization output is fishy")

        self.end()



    def minibatch_powell(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs): # Less effective than BFGS.. Usefull?

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.powell(**kwargs)


    def powell(self, maxiter=100, gtol=1e-8):

        self.start()
        logger.info("Starting POWELL for {} iterations (maximum) with gtol={}...".format(maxiter, gtol))

        optres = scipy.optimize.fmin_powell(
                self.cost, self.params[self.paramslice],
                args=(), xtol=0.0001, ftol=0.0001,
                maxiter=maxiter, maxfun=None,
                full_output=True, disp=True, retall=False, callback=self.callback, direc=None)

        if len(optres) == 7:
            (xopt, fopt, direc, iter, func_calls, warnflag, allvecs) = optres
            self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
            logger.info("Done with optimization, {0} func_calls".format(func_calls))
        else:
            logger.warning("Optimization output is fishy")

        self.end()






    def minibatch_slsqp(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs): # Less effective than BFGS.. Usefull?

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.slsqp(**kwargs)

    def slsqp(self, maxiter=100, gtol=1e-8):

        self.start()
        logger.info("Starting SLSQP for {} iterations (maximum) with gtol={}...".format(maxiter, gtol))

        optres = scipy.optimize.fmin_slsqp(self.cost, self.params[self.paramslice], eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=maxiter, acc=gtol, iprint=1, disp=None, full_output=1, epsilon=1.4901161193847656e-08, callback=self.callback)

        if len(optres) == 5:
            (xopt, fopt, func_calls, imode, smode) = optres
            self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
            logger.info("Done with optimization, {0} func_calls".format(func_calls))
        else:
            logger.warning("Optimization output is fishy")

        self.end()
        
        
        
    def minibatch_backprop(self, mbsize=None, mbfrac=0.1, mbloops=10, **kwargs):

        for loopi in range(mbloops):
            if mbloops > 1:
                logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
            self.dat.random_minibatch(mbsize=mbsize, mbfrac=mbfrac)
            self.optbatchchangeits.append(self.optit) # We record this minibatch change
            self.backprop(**kwargs)



    def backprop(self, maxiter=100, eta=0.0001):

        self.start()
        logger.info("Starting BACKPROP for {} iterations (maximum) ...".format(maxiter))
        
        cost = self.currentcost()
        
        tmpnet = self.net

        for iter in range(maxiter):
                 	
            logger.info("Starting iteration number {}, current cost {}, cost difference {}".format(iter+1, self.currentcost(), self.currentcost()-cost))
            cost = self.currentcost()
            
            outputs = self.net.run(self.dat.traininputs)
            targets = self.dat.traintargets
            
            deltas = [1]*len(self.net.layers) #Declaring the list that will contain all the deltas (or "errors")
            
            deltas[-1] = np.broadcast_to(np.mean(outputs,axis=0) - targets, np.shape(outputs)) #Last layer delta, note that activation function for this layer is the identity function
            
            for i in range(-2,-len(self.net.layers)-1,-1):
                deltas[i] = self.net.derivative_run(self.dat.traininputs,len(self.net.layers)+i) * np.rollaxis(np.dot(np.rollaxis(self.net.layers[i+1].weights,1),deltas[i+1]),1)
                #THE SHAPES ARE CORRECT
                logger.info("i {}, shape der_run {}, shape rollaxis {}".format(i,np.shape(self.net.derivative_run(self.dat.traininputs,len(self.net.layers)+i)),np.shape(np.rollaxis(np.dot(np.rollaxis(self.net.layers[i+1].weights,1),deltas[i+1]),1))))

            for li in range(len(self.net.layers)):
                #I THINK THE PROBLEM MUST COME FROM HERE
                
                logger.info("shape delta {} and shape parrun {}".format(np.shape(deltas[li]),np.shape(self.net.par_run(self.dat.traininputs,li))))
                tmpnet.layers[li].weights -= eta * np.tensordot(deltas[li],self.net.par_run(self.dat.traininputs,li),((0,2),(0,2))) #Best take at vectorization so far.. Note that numpy's tensordot function doesn't work with masked array
                tmpnet.layers[li].biases -= eta * np.sum(deltas[li],(0,2))
                
                #logger.info("GRADIENT IN LAYER {} IS \n{}".format(li+1,np.tensordot(deltas[li],self.net.par_run(self.dat.traininputs,li),((0,2),(0,2)))))
                #logger.info("NUMERICAL GRADIENT IN LAYER {} IS \n{}".format(li+1,self.numgrad(li,epsilon = 0.00000001)))
                logger.info("\n{}".format(np.tensordot(deltas[li],self.net.par_run(self.dat.traininputs,li),((0,2),(0,2)))-self.numgrad(li,epsilon = 0.00000000001)))

                self.net = tmpnet
                	
       

        self.optit += 1
        now = datetime.now()
        secondstaken = (now - self.iterationstarttime).total_seconds()
        callstaken = self.optitcall

        self.optittimes.append(secondstaken)
        self.optiterrs_train.append(self.opterr)
        self.optitcalls.append(self.optcall)
        self.optitparams.append(copy.deepcopy(self.params)) # We add a copy of the current params

        # Now we evaluate the cost on the validation set:
        valerr = self.valcost()
        self.optiterrs_val.append(valerr)

        valerrratio = valerr / self.opterr

        mscallcase = 1000.0 * float(secondstaken) / (float(callstaken) * self.dat.getntrain()) # Time per call and training case

        logger.info("Iter. {self.optit:4d}, {self.errfctname} train = {self.opterr:.6e}, val = {valerr:.6e} ({valerrratio:4.1f}), {time:.4f} s for {calls} calls ({mscallcase:.4f} ms/cc)".format(
                self=self, time=secondstaken, valerr=valerr, valerrratio=valerrratio, calls=callstaken, mscallcase=mscallcase))

        if self.itersavepath != None:
            self.save(self.itersavepath)

        # We reset the iteration counters:
        self.iterationstarttime = datetime.now()
        self.optitcall = 0

        self.end()        
        

    def numgrad(self,li,epsilon = 0.0001):
		
		tmpnet = self.net
		grad = np.ones(np.shape(tmpnet.layers[li].weights))
		
		for i in range(np.shape(tmpnet.layers[li].weights)[0]):
		    for j in range(np.shape(tmpnet.layers[li].weights)[1]):
		        tmpnet.layers[li].weights[i,j] += epsilon 
		        plus = tmpnet.run(self.dat.traininputs)
		        tmpnet.layers[li].weights[i,j] -= 2.0 *epsilon
		        minus = tmpnet.run(self.dat.traininputs)
		        tmpnet.layers[li].weights[i,j] += epsilon # Reestablishing the correct value of the weight
		        grad[i,j] = (err.ssb(plus, self.dat.traintargets) - err.ssb(minus, self.dat.traintargets)) / (2.0 * epsilon)	        
		
		return grad
#       def anneal(self, maxiter=100):
#
#               self.testcost()
#               logger.info("Starting annealing for {0} iterations (maximum)...".format(maxiter))
#
#               optres = scipy.optimize.basinhopping(
#                       self.cost, self.params,
#                       niter=maxiter, T=0.001, stepsize=0.1, minimizer_kwargs=None, take_step=None, accept_test=None,
#                       callback=self.callback, interval=100, disp=True, niter_success=None)
#
#                       # Warning : interval is not the callback interval, but the step size update interval.
#
#               print optres
#
#               print len(optres)

#       def fmin(self, maxiter=100):    # One iteration per call
#               self.testcost()
#               logger.info("Starting fmin for {0} iterations (maximum)...".format(maxiter))
#
#               optres = scipy.optimize.fmin(
#                       self.cost, self.params,
#                       xtol=0.0001, ftol=0.0001, maxiter=maxiter, maxfun=None,
#                       full_output=True, disp=True, retall=True, callback=self.callback)
#
#               print optres


#               """
#               optres = scipy.optimize.fmin_powell(
#                       cost, params,
#                       maxiter=maxiter, ftol=1e-06,
#                       full_output=True, disp=True, retall=True, callback=self.optcallback)
#               """
#               """
#               optres = scipy.optimize.fmin(
#                       cost, params,
#                       xtol=0.0001, ftol=0.0001, maxiter=maxiter, maxfun=None,
#                       full_output=True, disp=True, retall=True, callback=self.optcallback)
#               """
#               """
#               optres = scipy.optimize.minimize(
#                       cost, params, method="Anneal",
#                       jac=None, hess=None, hessp=None, bounds=None, constraints=(),
#                       tol=None, callback=self.optcallback, options={"maxiter":maxiter, "disp":True})
#               """
#
#               """
#               optres = scipy.optimize.basinhopping(
#                       cost, params,
#                       niter=maxiter, T=0.001, stepsize=1.0, minimizer_kwargs=None, take_step=None, accept_test=None,
#                       callback=self.optcallback, interval=50, disp=True, niter_success=None)
#               """
