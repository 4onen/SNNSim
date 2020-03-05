import yaml
import numpy as np


def readInput(data):
    inlist = yaml.safe_load(data)
    num_frames = len(inlist)
    frameTimes = np.zeros((num_frames, 1))
    frameFreqs = np.zeros((num_frames, 2))
    frame_size = len(inlist[0]['data'])
    frameData = np.zeros((frame_size, num_frames))

    frameTimeSum = 0
    for i, frame in enumerate(inlist):
        frameTimes[i] = frameTimeSum
        frameTimeSum += frame['displayTime']
        frameData[:, i] = frame['data']
        frameFreqs[i, 0] = frame['onFreq']
        frameFreqs[i, 1] = frame['offFreq']

    return (frameTimeSum, frameTimes, frameFreqs, frameData)


def readInputFile(filename):
    with open(filename, 'r') as infile:
        return readInput(infile)
