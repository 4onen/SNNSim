#!/usr/bin/env -S python3
from Neuron import *
from scipy.sparse.linalg import splu
import scipy.sparse
import matplotlib.pyplot as plt
import math


def simulator(model, outputs, num_tsteps):
    modelNeurons, matSize = model
    # Check whether we have any nonlinear neuron models
    hasNonlinear = any(map(lambda n: n.hasNonlinear, modelNeurons))

    # Allocate storage vectors
    spikes = np.zeros((1, 1), dtype=bool)
    inhibitions = np.zeros(len(modelNeurons), dtype=bool)
    Y0 = scipy.sparse.lil_matrix((matSize, matSize))
    J0 = np.zeros((matSize, 1))

    # Stamp linear (constant) behaviours
    for n in modelNeurons:
        Y0 = n.stampLinearY(Y0)
        J0 = n.stampLinearJ(J0)

    # Do LU factorization with SciPy SuperLU
    Ylin = splu(Y0.tocsc())

    vdata = np.ndarray((num_tsteps, len(outputs)))\
        if outputs is not None and len(outputs) > 0\
        else None

    # Prepare state variables
    v = np.zeros(matSize)-0.0087281773402
    for tidx in range(num_tsteps):
        # Stamp new excitation vector
        J = J0.copy()
        for i, n in enumerate(modelNeurons):
            J = n.stampCompanionJ(J, v[n.nV], spikes, False, None)

        # Simulation step
        v = Ylin.solve(J)
        v.shape = (v.shape[0],)

        # Calculate step spikes
        spikes[0, 0] = math.sin(10*tidx*tstep) > 0

        # Write outputs
        if vdata is not None:
            vdata[tidx, :] = v[outputs]

    if vdata is not None:
        tsteps = np.linspace(0, num_tsteps*tstep, num_tsteps)
        for i, o in enumerate(outputs):
            plt.figure
            plt.plot(tsteps, vdata[:, o])
            plt.title(str(o))
        plt.show()


if __name__ == "__main__":
    model = ([FNNeuron([0], 0)], 1)
    simulator(model, [0], 1000)
