import numpy as np


class InputNeuron:
    def __init__(self, framePixel, nS):
        self.framePixel = framePixel
        self.nS = nS

    def __str__(self):
        return 'Input Neuron #'+str(self.framePixel)+' spiking at '+str(self.ns)

    def __repr__(self):
        return 'Input #'+str(self.framePixel)+' nS:'+str(self.nS)


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
        self.data = np.zeros(tstepcount, 1)


class ModelNeuron:
    def __init__(self, inputNeuronIds, nS, hasNonlinear=False):
        self.inputNeuronIds = inputNeuronIds
        self.nS = nS
        self.hasNonlinear = hasNonlinear

    # def stampLinearY(self,Y)
    # def stampLinearJ(self,J)
    # def stampCompanionY(self,Y,vlast)
    # def stampCompanionJ(self,J,vlast)
    # def stampNonlinearY(self,Y,vguess)
    # def stampNonlinearJ(self,J,vguess)


class LIFNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nS, nV):
        super().__init__(inputNeuronIds, nS)
        self.nV = nV

    def __str__(self):
        return 'LIF neuron spiking on '+str(self.nS)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'LIF nS:'+str(self.nS)+' nV:'+str(self.nV)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        raise NotImplementedError()

    def stampLinearJ(self, J):
        raise NotImplementedError()

    def stampCompanionY(self, Y, vlast):
        raise NotImplementedError()

    def stampCompanionJ(self, J, vlast):
        raise NotImplementedError()


class FNNeuron(ModelNeuron):
    def __init__(self, inputNeuronIds, nV, nW):
        super().__init__(inputNeuronIds, nV, True)
        self.nV = nV
        self.nW = nW

    def __str__(self):
        return 'FN neuron spiking on '+str(self.nS)+' and inputs '+str(self.inputNeuronIds)

    def __repr__(self):
        return 'FN nS:'+str(self.nS)+' nW:'+str(self.nW)+' inputs:'+str(self.inputNeuronIds)

    def stampLinearY(self, Y):
        raise NotImplementedError()

    def stampLinearJ(self, J):
        raise NotImplementedError()

    def stampCompanionY(self, Y, vlast):
        raise NotImplementedError()

    def stampCompanionJ(self, J, vlast):
        raise NotImplementedError()

    def stampNonlinearY(self, Y, vguess):
        raise NotImplementedError()

    def stampNonlinearJ(self, J, vguess):
        raise NotImplementedError()
