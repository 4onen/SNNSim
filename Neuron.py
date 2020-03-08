import numpy as np
import matplotlib.pyplot as plt

LIFRealrestingPotential = -70e-3
LIFRealMembraneResistance = 10e6
LIFRealMembraneCapacitance = 200e-12
LIFRealThreshold = -60e-3

LIFDecayTC = 0.40455715765
LIFrestingPotential = 0
LIFMembraneResistance = 1e6
LIFMembraneCapacitance = LIFDecayTC/LIFMembraneResistance
LIFThreshold = 64

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


def LIF_weight_update(w, ispike, ospike):
    # W-current weight of a neuron and the function returns the updated weight
    # ispike is an array of size 5 with input data from time steps t-4 to t
    if(ispike[0] == 1):
        if (ospike[1] == 1):
            w -= 2
        if (ospike[2] == 1):
            w -= 1
    if(ospike[0] == 1):
        if (ispike[1] == 1):
            w += 2
        if (ispike[2] == 1):
            w += 1
    return np.clip(w, 0, 3)


class LIFNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV, w=None):
        super().__init__(inputNeuronIds, nV)
        if w is None:
            self.w = np.ones((len(inputNeuronIds),))
        else:
            self.w = np.array(w, dtype=float)
        self.spiked = np.zeros((3,), bool)

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

    def stampCompanionJ(self, J, vlast, slast, training):
        self.spiked[1:3] = self.spiked[0:2]
        self.spiked[0] = vlast > LIFThreshold
        if self.spiked[0]:
            vlast = LIFrestingPotential  # Spike reset

        if training:
            for i in range(len(self.w)):
                self.w[i] = LIF_weight_update(
                    self.w[i], slast[i, :], self.spiked)

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
