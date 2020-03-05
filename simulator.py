#!/usr/bin/env -S python3
import InputParser
import NetworkParser
from Neuron import *
from scipy.sparse.linalg import splu
import scipy.sparse

tstep = 0.001


def simulator(modelFile, inputFile):
    (inputNeurons, modelNeurons, outputNeurons,
     matSize, spikerCnt) = NetworkParser.readNetworkFile(modelFile)
    (tstop, frameTimes, frameFreqs, frameData) = InputParser.readInputFile(inputFile)
    print((inputNeurons, modelNeurons, outputNeurons),
          (frameTimes, frameFreqs, frameData))

    # Check whether we have any nonlinear neuron models
    hasNonlinear = any(map(lambda n: n.hasNonlinear, modelNeurons))

    # Allocate storage vectors
    spikes = np.zeros((spikerCnt, 1), dtype=bool)
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
        n.init_stepcount(tstop//tstep)

    # Prepare state variables
    v = np.zeros((matSize, 1))+restingPotential
    tidx = -1
    t = 0
    while t < tstop:
        tidx += 1

        # Stamp new excitation vector
        J = J0
        for n in modelNeurons:
            J = n.stampCompanionJ(J, spikes)

        # Simulation step
        if hasNonlinear:
            raise NotImplementedError()
        else:
            v = Ylin.solve(J)

        # Calculate step spikes
        for i, input in enumerate(inputNeurons):
            # TODO: I don't know the best way to decide when input cells spike
            raise NotImplementedError()
        # TODO: No FN support here.
        spikes[len(inputNeurons):end] = v > LIFThreshold

        # Write spikes to outputs
        for n in outputNeurons:
            n.add_datapoint(tidx, spikes)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} MODELFILE INPUTFILE')
        sys.exit(len(sys.argv) < 2)

    simulator(sys.argv[1], sys.argv[2])
