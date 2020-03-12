## Installation

Open a terminal or command prompt to the directory containing the simulation code. A terminal window will be used to enter all commands given in this file.

To check whether the terminal is pointed to the right location, the command
```
pdb
```
will tell you the path the terminal is pointed at. If this path is to the root directory of this project, the terminal is in the right place. Otherwise, use `cd` (Change Directory) and `ls` (List) to navigate to this project's root directory.

First, ensure sure Python 3.7 or above is installed:
```
python3 --version
```
If this command does not return a name and version number, check your Python installation.

Next, install this project's dependencies:
```
python3 -m pip install matplotlib numpy scipy pyyaml
```

Finally, test that this project works by plotting the output of an LIFXOR neuron for increasing current:
```
./voltageSim
```

### Whenever a plot appears, close the plot to continue execution!

## Usage
Standard usage is as follows:
```
./simulator NETWORK TRAIN_INPUT TEST_INPUT NUM_EPOCHS [PLOT_TRAINING]
```
+ NETWORK is any `.yaml` file containing a description of a spiking neural network that our simulator can understand. The folder `networks` provides three examples of such, from which the format should hopefully be clear.
+ TRAIN_INPUT is any `.yaml` file containing input data that our simulator can understand. The folder `benchmarks` provides eight such files.
+ TEST_INPUT is any `.yaml` file containing input data that our simulator can understand.
+ NUM_EPOCHS must be a number, which the simulator will use as the number of times to learn from TRAIN_INPUT before testing on TEST_INPUT.
+ PLOT_TRAINING is an optional option. If any text is present there, plots will be output of each epoch.

voltageSim usage is as follows:
```
./voltageSim TIMESTEPS MODELNAME
```
+ TIMESTEPS a number, which the simulator will use as the number of timesteps to simulate for.
+ MODELNAME is one of "LIF", "LIFXOR", or "FN". (FN has not been tested on recent edits to the software!)

## Replication:
Our STDP learning results can be replicated with:
```
./simulator networks/mnist_learner_inhib.yaml benchmarks/generatedInput.yaml benchmarks/oneTestingNewSeq.yaml 5
```
Our LIF voltage graph can be replicated with:
```
./voltageSim 1200 LIF
```
Our LIFXOR voltage graph can be replicated with:
```
./voltageSim 1200 LIFXOR
```