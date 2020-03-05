import yaml
import Neuron


def readNetwork(data):
    def readInputs(neuron, nameNums):
        inputs = list()
        for input in neuron['inputs']:
            try:
                inputs.append(int(input))
            except ValueError:
                inputs.append(nameNums[input])

        return inputs

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
            inputs = readInputs(neuron, nameNums)
            modelNeurons.append(Neuron.LIFNeuron(
                readInputs(neuron, nameNums), assignedMatrixIDs))
            assignedMatrixIDs += 1
            assignedSpikeIDs += 1
        elif neuron['model'] == 'FN':
            modelNeurons.append(Neuron.FNNeuron(
                readInputs(neuron, nameNums), assignedMatrixIDs, assignedMatrixIDs+1))
            assignedMatrixIDs += 2
            assignedSpikeIDs += 1
        elif neuron['model'] == 'Output':
            if 'name' in neuron:
                outputNeurons.append(Neuron.OutputNeuron(
                    neuron['name'], readInputs(neuron, nameNums)))
            else:
                outputNeurons.append(Neuron.OutputNeuron(
                    'Neuron #'+str(i), readInputs(neuron, nameNums)))
        else:
            raise KeyError()

    return (inputNeurons, modelNeurons, outputNeurons, assignedMatrixIDs, assignedSpikeIDs)


def readNetworkFile(filename):
    with open(filename, 'r') as infile:
        return readNetwork(infile)
