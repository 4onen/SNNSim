import yaml
import numpy as np


def readInput(data):
    indat = yaml.safe_load(data)
    num_frames = indat['frames']
    return (num_frames, indat['data'])


def readInputFile(filename):
    with open(filename, 'r') as infile:
        return readInput(infile)
