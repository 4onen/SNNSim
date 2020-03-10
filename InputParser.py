import yaml
import numpy as np


def readNeuronSeq(s, i):
    if isinstance(s, int):
        return bool((s >> i) & 1)
    else:
        return s[i] == '1'


def readInput(data):
    indat = yaml.safe_load(data)
    num_images = len(indat)
    num_frames = sum((im['inputs'] for im in indat))
    num_neurons = len(indat[0]['data'])
    frameDat = np.ndarray((num_frames, num_neurons), dtype=bool)
    for neuronIdx in range(num_neurons):
        frameDat[:, neuronIdx] = [readNeuronSeq(im['data'][neuronIdx], im['inputs']-tidx)
                                  for im in indat for tidx in range(im['inputs'])]
    return (num_frames, frameDat)


def readInputFile(filename):
    with open(filename, 'r') as infile:
        return readInput(infile)
