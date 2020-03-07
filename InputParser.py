import yaml
import numpy as np


def readInput(data):
    indat = yaml.safe_load(data)
    num_images = len(indat)
    num_frames = sum((im['inputs'] for im in indat))
    return (num_frames, [frame for im in indat for frame in im['data']])


def readInputFile(filename):
    with open(filename, 'r') as infile:
        return readInput(infile)
