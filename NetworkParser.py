import yaml
import Neuron


def readNetwork(data):
    def readInputs(neuron, nameNums, spikeIDs):
        inputs = list()
        for input in neuron['inputs']:
            try:
                inputs.append(spikeIDs[int(input)])
            except ValueError:
                inputs.append(spikeIDs[nameNums[input]])

        return inputs

    inlist = yaml.safe_load(data)

    # Extract IDs for various names
    nameNums = dict()
    for i, neuron in enumerate(inlist):
        if 'name' in neuron.keys():
            nameNums[neuron['name']] = i

    # Assign Matrix positions for different models
    assignedMatrixIDs = 0
    spikeIDs = list()
    for neuron in inlist:
        if neuron['model'] == 'Input':
            spikeIDs.append(assignedMatrixIDs)
            assignedMatrixIDs += 1
        elif neuron['model'] == 'LIF':
            spikeIDs.append(assignedMatrixIDs)
            assignedMatrixIDs += 2
        elif neuron['model'] == 'FN':
            spikeIDs.append(assignedMatrixIDs)
            assignedMatrixIDs += 2
        elif neuron['model'] == 'Output':
            pass
        else:
            raise KeyError()

    # Extract neuron kinds
    inputNeurons = list()
    modelNeurons = list()
    outputNeurons = list()
    for i, neuron in enumerate(inlist):
        if neuron['model'] == 'Input':
            inputNeurons.append(Neuron.InputNeuron(i, spikeIDs[i]))
        elif neuron['model'] == 'LIF':
            inputs = readInputs(neuron, nameNums, spikeIDs)
            modelNeurons.append(Neuron.LIFNeuron(
                readInputs(neuron, nameNums, spikeIDs),
                spikeIDs[i],
                spikeIDs[i]+1))
        elif neuron['model'] == 'FN':
            modelNeurons.append(Neuron.FNNeuron(
                readInputs(neuron, nameNums, spikeIDs),
                spikeIDs[i],
                spikeIDs[i]+1))
        elif neuron['model'] == 'Output':
            if 'name' in neuron:
                outputNeurons.append(Neuron.OutputNeuron(
                    neuron['name'], readInputs(neuron, nameNums, spikeIDs)))
            else:
                outputNeurons.append(Neuron.OutputNeuron(
                    'Neuron #'+str(i), readInputs(neuron, nameNums, spikeIDs)))

    return (inputNeurons, modelNeurons, outputNeurons)


def readNetworkFile(filename):
    with open(filename, 'r') as infile:
        return readNetwork(infile)
