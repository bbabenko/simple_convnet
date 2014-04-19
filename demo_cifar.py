import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from sklearn.metrics import confusion_matrix

from simple_convnet import convnet as cn

####################################################################################################
# LOAD DATA

def load_batch(fname):
    with open('data/cifar-10-batches-py/%s'%fname, 'rb') as f:
        data = pickle.load(f)
    x = data['data'].reshape((-1,3,32,32)).astype('float32')/255
    x =  np.rollaxis(x, 1, 4)
    y = np.array(data['labels'])
    return x, y

train_x = np.zeros((30000,32,32,3), dtype='float32')
train_y = np.zeros(30000, dtype='float32')
for b in xrange(3):
    train_x[b*10000:(b+1)*10000,...], train_y[b*10000:(b+1)*10000] = \
        load_batch('data_batch_%d' % (b+1))
mean_x = train_x[::10,...].mean(0)[np.newaxis,...]

val_x, val_y = load_batch('data_batch_5')
test_x, test_y = load_batch('test_batch')

####################################################################################################
# SET UP PARAMETERS

layer_args = [
        (cn.ConvLayer, dict(num_filters=32, filter_shape=(5,5), init_from=None)),
        (cn.BiasLayer, dict(init_val=1)),
        (cn.ReluLayer, dict()),
        (cn.MeanPoolingLayer, dict(pool_size=2)),
        (cn.ConvLayer, dict(num_filters=32, filter_shape=(5,5))),
        (cn.BiasLayer, dict(init_val=1)),
        (cn.ReluLayer, dict()),
        (cn.MeanPoolingLayer, dict(pool_size=2)),
        (cn.DenseLayer, dict(num_nodes=64)),
        (cn.BiasLayer, dict(init_val=1)),
        (cn.ReluLayer, dict()),
        (cn.DenseLayer, dict(num_nodes=10)),
        (cn.BiasLayer, dict())
        ]

fit_args = dict(
        val_freq=20,
        batch_size=32, 
        num_epoch=30, 
        weight_decay=0.0005,
        learn_rate_decay=.00005,
        chill_out_iters=100,
        momentum=0.9,
        learn_rate=.01)

####################################################################################################
# TRAIN AND TEST

net = cn.SoftmaxNet(layer_args=layer_args, 
                    input_shape=train_x.shape[1:], 
                    rand_state=np.random.RandomState(0))

with open('kmeans_filters.pkl', 'rb') as f:
    filters = pickle.load(f)
net.layers_[0].filters_ = filters/5.0

net.fit(train_x, 
        train_y, 
        val_x=val_x[::30,:],
        val_y=val_y[::30],
        verbose=True,
        **fit_args)

with open(join('data', 'cifar_model.pkl'), 'wb') as f:
    pickle.dump((net, layer_args, fit_args), f)
        
yp = net.predict(test_x, batch_size=128)
print 'test accuracy: %f' % np.mean(yp == test_y)

conf = confusion_matrix(test_y, yp)
plt.matshow(conf)
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
