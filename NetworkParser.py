import yaml
import Neuron
import numpy as np


def readNetwork(data):
    def readInputs(inputs, nameNums):
        inputIds = list()
        for input in inputs:
            try:
                inputIds.append(int(input))
            except ValueError:
                inputIds.append(nameNums[input])

        return inputIds

    inlist = yaml.safe_load(data)

    # Extract IDs for various names
    nameNums = dict()
    for i, neuron in enumerate(inlist):
        if 'name' in neuron.keys():
            nameNums[neuron['name']] = i

    # Extract neuron kinds
    inputNeurons = list()
    modelNeurons = list()
    outputNeurons = list()
    assignedSpikeIDs = 0
    assignedMatrixIDs = 0
    for i, neuron in enumerate(inlist):
        if neuron['model'] == 'Input':
            inputNeurons.append(Neuron.InputNeuron(i))
            assignedSpikeIDs += 1
        elif neuron['model'] == 'LIF':
            inputs = readInputs(neuron['inputs'], nameNums)
            modelNeurons.append(Neuron.LIFNeuron(
                inputs, assignedMatrixIDs, np.array(neuron['weights'])))
            assignedMatrixIDs += 1
            assignedSpikeIDs += 1
        elif neuron['model'] == 'FN':
            modelNeurons.append(Neuron.FNNeuron(
                readInputs(neuron['inputs'], nameNums), assignedMatrixIDs, assignedMatrixIDs+1))
            assignedMatrixIDs += 2
            assignedSpikeIDs += 1
        elif neuron['model'] == 'Output':
            name = neuron['name'] if 'name' in neuron.keys()\
                else 'Neuron #'+str(i)
            voltageIds = np.array(readInputs(neuron['vinputs'], nameNums))-len(inputNeurons) \
                if 'vinputs' in neuron.keys() else None
            outputNeurons.append(Neuron.OutputNeuron(
                name, readInputs(neuron['inputs'], nameNums), voltageIds))
        else:
            raise KeyError()

    return (inputNeurons, modelNeurons, outputNeurons, assignedMatrixIDs, assignedSpikeIDs)


def readNetworkFile(filename):
    with open(filename, 'r') as infile:
        return readNetwork(infile)
