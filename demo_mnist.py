import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from os.path import join

from simple_convnet import convnet as cn

####################################################################################################
# LOAD DATA

# Download at http://deeplearning.net/data/mnist/mnist.pkl.gz
with open('data/mnist.pkl', 'rb') as f:
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = pickle.load(f)

train_x = train_x.reshape((-1,28,28,1)).astype('float32')
test_x = test_x.reshape((-1,28,28,1)).astype('float32')
val_x = val_x.reshape((-1,28,28,1)).astype('float32')

####################################################################################################
# SET UP PARAMETERS

### simpler net with much fewer params
# layer_args = [
#         (cn.ConvLayer, dict(num_filters=20, filter_shape=(9,9))),
#         (cn.BiasLayer, dict(init_val=0.1)),
#         (cn.ReluLayer, dict()),
#         (cn.MeanPoolingLayer, dict(pool_size=2)),
#         (cn.DenseLayer, dict(num_nodes=10)),
#         (cn.BiasLayer, dict())
#         ] 

### closer to LeNet5
layer_args = [
        (cn.ConvLayer, dict(num_filters=8, filter_shape=(5,5))),
        (cn.BiasLayer, dict(init_val=0.1)),
        (cn.ReluLayer, dict()),
        (cn.MeanPoolingLayer, dict(pool_size=2)),
        (cn.ConvLayer, dict(num_filters=16, filter_shape=(5,5))),
        (cn.BiasLayer, dict(init_val=0.1)),
        (cn.ReluLayer, dict()),
        (cn.MeanPoolingLayer, dict(pool_size=2)),
        (cn.DenseLayer, dict(num_nodes=128)),
        (cn.BiasLayer, dict(init_val=0.1)),
        (cn.ReluLayer, dict()),
        (cn.DenseLayer, dict(num_nodes=10)),
        (cn.BiasLayer, dict())
        ]

fit_args = dict(
        val_freq=10,
        batch_size=64, 
        num_epoch=3, 
        weight_decay=0.0005,
        momentum=0.9,
        learn_rate=1e-1)

####################################################################################################
# TRAIN AND TEST

net = cn.SoftmaxNet(layer_args=layer_args, 
                    input_shape=train_x.shape[1:],
                    rand_state=np.random.RandomState(0))
net.fit(train_x, 
        train_y, 
        val_x=val_x[::10,:],
        val_y=val_y[::10],
        verbose=True,
        **fit_args)

with open(join('data', 'mnist_model_deep.pkl'), 'wb') as f:
    pickle.dump((net, layer_args, fit_args), f)

yp = net.predict(test_x, batch_size=128)
print 'test accuracy: %f' % np.mean(yp == test_y)

conf = confusion_matrix(test_y, yp)
plt.matshow(conf)
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
