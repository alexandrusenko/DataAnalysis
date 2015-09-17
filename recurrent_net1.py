from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer
ds = SupervisedDataSet(10, 11)
z = map(int,'1 2 4 6 2 3 4 5 1 3 5 6 7 1 4 7 1 2 3 5 6 1 2 5 6 2 4 4 5 1 2 5 6 7 1 4 6 1 2 3 3 6 1 3 5 7 2 4 6 7 1 3 5 6 7 1 4 6 1 2 2 3 7'.split())
obsLen = 10
predLen = 11
for i in xrange(len(z)):
  if i+(obsLen-1)+predLen < len(z):
    ds.addSample([z[d] for d in range(i,i+obsLen)],[z[d] for d in range(i+1,i+1+predLen)])

net = buildNetwork(10, 20, 11, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)
ts = UnsupervisedDataSet(10,)
ts.addSample(map(int,'1 3 5 7 2 4 6 7 1 3'.split()))
print [ int(round(i)) for i in net.activateOnDataset(ts)[0]]