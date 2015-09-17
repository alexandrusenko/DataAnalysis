#Then, make a simple time series:
data = [1] * 3 + [2] * 3
data *= 3
print(data)
#Now put this timeseries into a supervised dataset, where the target for each sample is the next sample:from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.datasets.sequential import SequentialDataSet

ds = SequentialDataSet(1, 1)
for sample, next_sample in zip(data, cycle(data[1:])):
    ds.addSample(sample, next_sample)
print ds

#Build a simple LSTM network with 1 input node, 5 LSTM cells and 1 output node:
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

net = buildNetwork(1, 5, 1, 
                   hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

#Train the network:
from pybrain.supervised import RPropMinusTrainer
from sys import stdout

trainer = RPropMinusTrainer(net, dataset=ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 100
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    print("epoch: "+str(epoch) + "/" + str(EPOCHS))
    stdout.flush()

print()
print("final error =", train_errors[-1])

#Plot the errors (note that in this simple toy example, we are testing and training on the same dataset, 
#which is of course not what you'd do for a real project!):
import matplotlib.pyplot as plt

plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

#Now ask the network to predict the next sample:
for sample, target in ds.getSequenceIterator(0):
    print("               sample = %4.1f" % sample)
    print("predicted next sample = %4.1f" % net.activate(sample))
    print("   actual next sample = %4.1f" % target)
    print()