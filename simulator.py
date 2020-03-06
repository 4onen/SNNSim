#!/usr/bin/env -S python3
import InputParser
import NetworkParser
from Neuron import *
from scipy.sparse.linalg import splu
import scipy.sparse
import math


def simulator(modelFile, inputFile):
    (inputNeurons, modelNeurons, outputNeurons,
     matSize, spikerCnt) = NetworkParser.readNetworkFile(modelFile)
    (tstop, frameTimes, frameFreqs, frameData) = InputParser.readInputFile(inputFile)
    print((inputNeurons, modelNeurons, outputNeurons),
          (frameTimes, frameFreqs, frameData))

    # Check whether we have any nonlinear neuron models
    hasNonlinear = any(map(lambda n: n.hasNonlinear, modelNeurons))

    # Allocate storage vectors
    spikes = np.zeros((spikerCnt, 5), dtype=bool)
    Y0 = scipy.sparse.lil_matrix((matSize, matSize))
    J0 = np.zeros((matSize, 1))

    # Stamp linear (constant) behaviours
    for n in modelNeurons:
        Y0 = n.stampLinearY(Y0)
        J0 = n.stampLinearJ(J0)

    # If we have no nonlinear neuron models, do LU factorization with SciPy SuperLU
    if not hasNonlinear:
        Ylin = splu(Y0.tocsc())

    # Initialize output neurons with number of steps
    for n in outputNeurons:
        n.init_stepcount(math.ceil(tstop/tstep))

    # Prepare state variables
    v = np.zeros((matSize, 1))+LIFrestingPotential
    tidx = 0
    t = 0
    while t < tstop:
        # Stamp new excitation vector
        J = J0.copy()
        for i, n in enumerate(modelNeurons):
            stdpSpiked = spikes[i+len(inputNeurons), 2]
            J = n.stampCompanionJ(J, v[n.nV], spikes, stdpSpiked)

        # Simulation step
        if hasNonlinear:
            raise NotImplementedError()
        else:
            v = Ylin.solve(J)

        # Calculate step spikes
        for i, input in enumerate(inputNeurons):
            # TODO: I don't know the best way to decide when input cells spike
            pass  # raise NotImplementedError()
        # TODO: No FN support here.
        spikes[:, 1:5] = spikes[:, 0:4]
        spikes[len(inputNeurons):spikes.shape[0], 0] = v > LIFThreshold

        # Write spikes to outputs
        for n in outputNeurons:
            n.add_datapoint(tidx, spikes[:, 0])

        # Advance time
        t += tstep
        tidx += 1

    tsteps = np.linspace(0, tstop, tidx)
    for n in outputNeurons:
        n.plot(tsteps)


def weight_update(self, w, ispike, ospike):
    # W-current weight of a neuron and the function returns the updated weight
    # ispike is an array of size 5 with input data from time steps t-4 to t
    # ospike is the output data at time step t-2
    if (ospike == 1):
        if (ispike[1] == 1):
            w+= 2
        if (ispike[0] == 1):
            w+= 1
        if (ispike[3] == 1):
            w-= 2
        if (ispike[4] == 1):
            w-= 1
        return w
    else:
        return w


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} MODELFILE INPUTFILE')
        sys.exit(len(sys.argv) < 2)

    simulator(sys.argv[1], sys.argv[2])
