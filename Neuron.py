import numpy as np
import matplotlib.pyplot as plt

restingPotential = -0.065
LIFMembraneResistance = 10000000
LIFThreshold = 0


class InputNeuron:
    def __init__(self, framePixel):
        self.framePixel = framePixel

    def __str__(self):
        return 'Input Neuron #'+str(self.framePixel)

    def __repr__(self):
        return 'Input #'+str(self.framePixel)


class OutputNeuron:
    def __init__(self, name, inputNeuronIds):
        self.name = name
        self.inputNeuronIds = inputNeuronIds
        self.data = None

    def __str__(self):
        dat = 'no data' if self.data is None else \
            str(self.data.shape[0])+' datapoints'
        return 'Output '+self.name+' with inputs '+str(self.inputNeuronIds)+' and '+dat

    def __repr__(self):
        return 'Output "'+self.name+'" inputs:'+str(self.inputNeuronIds)+' dat:'+(str(None) if self.data is None else (self.data.shape[0]))

    def init_stepcount(self, tstepcount):
        self.data = np.ndarray((tstepcount, 1))

    def add_datapoint(self, i, v):
        self.data[i] = sum(v[self.inputNeuronIds])

    def plot(self, tsteps):
        plt.plot(tsteps, self.data)
        plt.title(self.name)
        plt.show()


class ModelNeuron:
    def __init__(self, inputNeuronIds, nV, hasNonlinear=False):
        self.inputNeuronIds = inputNeuronIds
        self.nV = nV
        self.hasNonlinear = hasNonlinear

    # def stampLinearY(self,Y)
    # def stampLinearJ(self,J)
    # def stampCompanionJ(self,J,slast)
    # def stampNonlinearY(self,Y,vguess)
    # def stampNonlinearJ(self,J,vguess)


class LIFNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV):
        super().__init__(inputNeuronIds, nV)
        # TODO: Add weight vector to self.something

    def __str__(self):
        return 'LIF neuron voltage on '+str(self.nV)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'LIF nV:'+str(self.nV)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        Y[self.nV, self.nV] = LIFMembraneResistance
        # TODO: Capacitor
        return Y

    def stampLinearJ(self, J):
        return J

    def stampCompanionJ(self, J, slast, stdpSpiked):
        if stdpSpiked:
            pass  # TODO: STDP Learning weight vector update

        # TODO: Current source: J[self.nV] += np.dot(slast[self.inputNeurons,0],self.whateverWeightsAreCalled)
        # TODO: Capacitor, what-have-you.
        return J


class FNNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV, nW):
        super().__init__(inputNeuronIds, nV, True)
        self.nV = nV
        self.nW = nW

    def __str__(self):
        return 'FN neuron spiking on '+str(self.nV)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'FN nV:'+str(self.nV)+' nW:'+str(self.nW)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        raise NotImplementedError()

    def stampLinearJ(self, J):
        raise NotImplementedError()

    def stampCompanionJ(self, J, slast):
        raise NotImplementedError()

    def stampNonlinearY(self, Y, vguess):
        raise NotImplementedError()

    def stampNonlinearJ(self, J, vguess):
        raise NotImplementedError()
