import numpy as np
import matplotlib.pyplot as plt

LIFrestingPotential = -70e-3
LIFMembraneResistance = 10e6
LIFMembraneCapacitance = 200e-12
LIFThreshold = -60e-3
tstep = 10e-3


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
        self.data = np.ndarray(tstepcount)

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
    # def stampCompanionJ(self,J,vlast,slast,stdpSpiked)
    # def stampNonlinearY(self,Y,vguess)
    # def stampNonlinearJ(self,J,vguess)


class LIFNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV):
        super().__init__(inputNeuronIds, nV)
        self.w = np.ones((len(inputNeuronIds), 1))

    def __str__(self):
        return 'LIF neuron voltage on '+str(self.nV)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'LIF nV:'+str(self.nV)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        Y[self.nV, self.nV] += 1/LIFMembraneResistance  # Leak Resistor
        Y[self.nV, self.nV] += LIFMembraneCapacitance/tstep  # Capacitor
        return Y

    def stampLinearJ(self, J):
        J[self.nV] += LIFrestingPotential/LIFMembraneResistance
        return J

    def stampCompanionJ(self, J, vlast, slast, stdpSpiked):
        if stdpSpiked:
            pass  # TODO: STDP Learning weight vector update

        if vlast > LIFThreshold:
            vlast = LIFrestingPotential  # Spike reset

        J[self.nV] += np.dot(slast[self.inputNeuronIds, 0], self.w)  # STDP
        J[self.nV] += LIFMembraneCapacitance/tstep * vlast  # Capacitor
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
