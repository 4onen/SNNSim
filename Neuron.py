import numpy as np
import matplotlib.pyplot as plt

LIFRealrestingPotential = -70e-3
LIFRealMembraneResistance = 10e6
LIFRealMembraneCapacitance = 200e-12
LIFRealThreshold = -60e-3

LIFDecayTC = 0.01
LIFrestingPotential = 0
LIFMembraneResistance = 1
LIFMembraneCapacitance = LIFDecayTC/LIFMembraneResistance
LIFThreshold = 10

LIFLearningRate = 0.1

tstep = 1e-3


class InputNeuron:
    def __init__(self, framePixel):
        self.framePixel = framePixel

    def __str__(self):
        return 'Input Neuron #'+str(self.framePixel)

    def __repr__(self):
        return 'Input #'+str(self.framePixel)


class OutputNeuron:
    def __init__(self, name, inputNeuronIds, modelOffset):
        self.name = name
        self.inputNeuronIds = inputNeuronIds
        self.modelNeuronIds = list(map(lambda x: x-modelOffset,
                                       filter(lambda x: x >= modelOffset, inputNeuronIds)))
        self.data = None
        self.vdata = None

    def __str__(self):
        dat = 'no data' if self.data is None else \
            str(self.data.shape[0])+' datapoints'
        return 'Output '+self.name+' with inputs '+str(self.inputNeuronIds)+' and '+dat

    def __repr__(self):
        return 'Output "'+self.name+'" inputs:'+str(self.inputNeuronIds)+' dat:'+(str(None) if self.data is None else (self.data.shape[0]))

    def init_stepcount(self, tstepcount):
        self.data = np.ndarray(tstepcount)
        if len(self.modelNeuronIds) > 0:
            self.vdata = np.ndarray(tstepcount)

    def add_datapoint(self, i, s, v, inhib):
        self.data[i] = sum(s[self.inputNeuronIds])\
            - sum(inhib[self.modelNeuronIds])
        if len(self.modelNeuronIds) > 0:
            self.vdata[i] = sum(v[self.modelNeuronIds])

    def plot(self, tsteps):
        if len(self.modelNeuronIds) > 0:
            plt.plot(tsteps, self.vdata, 'y-')
        plt.plot(tsteps, self.data, 'b-')
        plt.title(self.name)
        plt.show()


class ModelNeuron:
    def __init__(self, inputNeuronIds, nV, nSpike, latInhib=[], w=None, hasNonlinear=False):
        self.inputNeuronIds = inputNeuronIds
        self.nV = nV
        self.nSpike = nSpike
        self.hasNonlinear = hasNonlinear
        if w is None:
            self.w = np.ones((len(inputNeuronIds),))
        else:
            self.w = np.array(w, dtype=float)
        self.latInhib = latInhib

    # def stampLinearY(self,Y)
    # def stampLinearJ(self,J)
    # def stampCompanionJ(self,J,vlast,slast,training,inhibited)


def LIF_weight_update(w, ispike, ospike):
    # W-current weight of a neuron and the function returns the updated weight
    # ispike is an array of size 5 with input data from time steps t-4 to t
    if(ispike[0] == 1):
        if (ospike[1] == 1):
            w -= 2*LIFLearningRate
        if (ospike[2] == 1):
            w -= 1*LIFLearningRate
    if(ospike[0] == 1):
        if (ispike[0] == 1):
            w += 2*LIFLearningRate
        if (ispike[1] == 1):
            w += 1*LIFLearningRate
        # if (ispike[2] == 1):
        #    w += 0.25*LIFLearningRate
    return np.clip(w, 0, 6)


class LIFNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV, nSpike, latInhib=[], w=None):
        super().__init__(inputNeuronIds, nV, nSpike, latInhib, w)

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

    def stampCompanionJ(self, J, vlast, slast, training, inhibited):
        if inhibited or slast[self.nSpike, 0]:
            vlast = LIFrestingPotential  # Spike reset

        if training:
            for i in range(len(self.w)):
                self.w[i] = LIF_weight_update(
                    self.w[i], slast[i, :], slast[self.nSpike, :])

        J[self.nV] += np.dot(slast[self.inputNeuronIds, 0], self.w)  # STDP
        J[self.nV] += LIFMembraneCapacitance/tstep * vlast  # Capacitor
        return J


def negativeXOR(Iin):
    V_I = LIFThreshold*1.3
    return (Iin-V_I*np.exp(0.1*(V_I-Iin)))*np.heaviside(Iin-V_I, 1)


class LIFXorNeuron(LIFNeuron):
    def __init__(self, inputNeuronIds, nV, nSpike, latInhib=[], w=None):
        super().__init__(inputNeuronIds, nV, nSpike, latInhib, w)

    def __str__(self):
        return 'LIF neuron voltage on '+str(self.nV)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'LIF nV:'+str(self.nV)+' inputs:'+str(self.inputNeuronIds)

    def stampCompanionJ(self, J, vlast, slast, training, inhibited):
        super().stampCompanionJ(J, vlast, slast, training, inhibited)
        J[self.nV] -= negativeXOR(np.dot(slast[self.inputNeuronIds, 0], self.w))
        return J


class FNNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV, nSpike, w=None):
        super().__init__(inputNeuronIds, nV, nSpike, [], w)
        self.W = 0.8739065233685703

    def __str__(self):
        return 'FN neuron spiking on '+str(self.nV)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'FN nV:'+str(self.nV)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        Y[self.nV, self.nV] = 1
        return Y

    def stampLinearJ(self, J):
        return J

    def stampCompanionJ(self, J, vlast, slast, training, inhibited):
        I = np.dot(slast[self.inputNeuronIds, 0], self.w)  # STDP Weights
        dV = vlast-vlast*vlast*vlast/3-self.W+I
        dW = 0.08*(vlast+0.7-0.8*self.W)
        J[self.nV] += tstep*dV
        self.W += tstep*dW
        # print(self.W)
        return J
