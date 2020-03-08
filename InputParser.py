import yaml
import numpy as np


def readNeuronSeq(s):
    if isinstance(s, int):
        return s
    else:
        return int(s, 2)


def readInput(data):
    indat = yaml.safe_load(data)
    num_images = len(indat)
    num_frames = sum((im['inputs'] for im in indat))
    num_neurons = len(indat[0]['data'])
    frameDat = np.ndarray((num_frames, num_neurons), dtype=bool)
    for neuronIdx in range(num_neurons):
        frameDat[:, neuronIdx] = [bool((readNeuronSeq(im['data'][neuronIdx]) >> (im['inputs']-tidx)) & 1)
                                  for im in indat for tidx in range(im['inputs'])]
    return (num_frames, frameDat)


def readInputFile(filename):
    with open(filename, 'r') as infile:
        return readInput(infile)
