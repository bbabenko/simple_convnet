import numpy as np
from simple_convnet import convnet as cn

from scipy.optimize import approx_fprime

def _check_gradients(layer_args, input_shape):
    rand = np.random.RandomState(0)
    net = cn.SoftmaxNet(layer_args=layer_args, input_shape=input_shape, rand_state=rand)
    x = rand.randn(*(10,)+net.input_shape)/100
    y = rand.randn(10) > 0
    by = net.binarize_labels(y)

    g1 = approx_fprime(net.get_params(), net.cost_for_params, 1e-5, x, by)
    g2 = net.param_grad(x, by)
    err = np.max(np.abs(g1-g2))/np.abs(g1).max()
    print err
    assert err < 1e-3, 'incorrect gradient!'

def test_dense_layer():
    layer_args = [(cn.DenseLayer, dict(num_nodes=20)), 
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (10,))

def test_relu_layer():
    layer_args = [(cn.ReluLayer, dict()),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (10,))

def test_sigmoid_layer():
    layer_args = [(cn.SigmoidLayer, dict()),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (10,))

def test_conv_layer():
    layer_args = [(cn.ConvLayer, dict(num_filters=5, filter_shape=(3,3))),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (8,8,3))

def test_convbias_layer():
    layer_args = [(cn.ConvLayer, dict(num_filters=5, filter_shape=(3,3))),
                  (cn.BiasLayer, dict()),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (8,8,3))

def test_pool_layer():
    layer_args = [(cn.ConvLayer, dict(num_filters=5, filter_shape=(3,3))),
                  (cn.MeanPoolingLayer, dict(pool_size=2)),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (8,8,3))

def test_deep():
    layer_args = [(cn.ConvLayer, dict(num_filters=5, filter_shape=(3,3))),
                  (cn.BiasLayer, dict()),
                  (cn.ReluLayer, dict()),
                  (cn.MeanPoolingLayer, dict(pool_size=2)),
                  (cn.ConvLayer, dict(num_filters=5, filter_shape=(3,3))),
                  (cn.BiasLayer, dict()),
                  (cn.SigmoidLayer, dict()),
                  (cn.MeanPoolingLayer, dict(pool_size=2)),
                  (cn.DenseLayer, dict(num_nodes=10)),
                  (cn.BiasLayer, dict()),
                  (cn.DenseLayer, dict(num_nodes=2))]
    _check_gradients(layer_args, (18,18,3))

def test_fit():
    layer_args = [(cn.DenseLayer, dict(num_nodes=4)),
                  (cn.DenseLayer, dict(num_nodes=2))]
    net = cn.SoftmaxNet(layer_args=layer_args, input_shape=(2,))

    num = 1000
    rand = np.random.RandomState(0)
    x = rand.rand(num,2)
    y = np.zeros(num)
    y[x[:,0]>0.5] = 1

    net.fit(x, y, batch_size=16, learn_rate=1, num_epoch=100, verbose=True)
    yp = net.predict(x)
    acc = np.mean(y==yp)
    assert acc > 0.7
