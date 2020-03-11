#!/usr/bin/env -S python3
import InputParser
import NetworkParser
from Neuron import *
from scipy.sparse.linalg import splu
import scipy.sparse
import math
import matplotlib.pyplot as plt


def oneSim(modelFile, inputFile):
    model = NetworkParser.readNetworkFile(modelFile)
    inputs = InputParser.readInputFile(inputFile)
    print(model[0], model[1], model[2])
    simulator(model, inputs, True, True)


def trainer(modelFile, trainingFile, testingFile, trainingEpochs, plotTraining=False):
    model = NetworkParser.readNetworkFile(modelFile)
    trainingInput = InputParser.readInputFile(trainingFile)

    print(model[0], model[1], model[2])

    for i in range(trainingEpochs):
        simulator(model, trainingInput, True, plotTraining)
        if (i+1) % 5 == 0:
            print(f'Epoch {i+1} completed.')

    testingInput = InputParser.readInputFile(testingFile)
    simulator(model, testingInput, False, True)
    for i, n in enumerate(model[1]):
        plt.imshow(np.reshape(n.w, (5, 5)))
        plt.title(i+1)
        plt.show()


def simulator(model, inputs, training, output):
    inputNeurons, modelNeurons, outputNeurons, matSize, spikerCnt = model
    num_tsteps, frameData = inputs
    # Check whether we have any nonlinear neuron models
    hasNonlinear = any(map(lambda n: n.hasNonlinear, modelNeurons))

    # Allocate storage vectors
    spikes = np.zeros((spikerCnt, 5), dtype=bool)
    inhibitions = np.zeros(len(modelNeurons), dtype=bool)
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
    if output:
        for n in outputNeurons:
            n.init_stepcount(num_tsteps)

    # Prepare state variables
    v = np.zeros(matSize)+LIFrestingPotential
    for tidx in range(num_tsteps):
        # Calculate lateral inhibitions
        inhibitions[:] = False
        for i, n in enumerate(modelNeurons):
            if spikes[len(inputNeurons)+i, 0] and len(n.latInhib) > 0:
                for inhib in n.latInhib:
                    inhibitions[inhib] = True

        # Stamp new excitation vector
        J = J0.copy()
        for i, n in enumerate(modelNeurons):
            J = n.stampCompanionJ(
                J, v[n.nV], spikes, training, inhibitions[i])

        # Simulation step
        if hasNonlinear:
            raise NotImplementedError()
        else:
            v = Ylin.solve(J)
            v.shape = (v.shape[0],)

        # Advance spike time
        spikes[:, 1:] = spikes[:, :(spikes.shape[1]-1)]
        # Calculate step spikes
        for i, input in enumerate(inputNeurons):
            spikes[i, 0] = frameData[tidx, i]
        # TODO: No FN support here. Calculate model spikes
        spikes[len(inputNeurons):spikes.shape[0], 0] = v > LIFThreshold

        # Write spikes to outputs
        if output:
            for n in outputNeurons:
                n.add_datapoint(tidx, spikes[:, 0], v, inhibitions)

        # Advance time
        tidx += 1

    if output:
        tsteps = np.linspace(0, num_tsteps*tstep, num_tsteps)
        for n in outputNeurons:
            n.plot(tsteps)


if __name__ == "__main__":
    import sys

    if len(sys.argv) is 3:
        oneSim(sys.argv[1], sys.argv[2])
    elif len(sys.argv) is 5:
        trainer(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) is 6:
        trainer(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), True)
    else:
        print(
            f'Usage: {sys.argv[0]} MODELFILE INPUTFILE\n\tor: {sys.argv[0]} MODELFILE TRAININGFILE TESTINGFILE EPOCHS')
        sys.exit(1)
