#!/usr/bin/env -S python3
import InputParser
import NetworkParser
from Neuron import *


def simulator(modelFile, inputFile):
    (inputNeurons, modelNeurons, outputNeurons) = NetworkParser.readNetworkFile(modelFile)
    (frameTimes, frameFreqs, frameData) = InputParser.readInputFile(inputFile)
    print((inputNeurons, modelNeurons, outputNeurons),
          (frameTimes, frameFreqs, frameData))
    raise NotImplementedError()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} MODELFILE INPUTFILE')
        sys.exit(len(sys.argv) < 2)

    simulator(sys.argv[1], sys.argv[2])
