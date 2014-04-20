import numpy as np

from simple_convnet.helpers import (
    filter2D, batch_filter3D, padarray, atleast, safe_exp, safe_log, choice, imshow
    )
from matplotlib import pyplot as plt
from time import time
from skimage.transform import downscale_local_mean

class Layer(object):
    def __init__(self, input_shape, rand_state=np.random):
        """
        Layer constructor (abstract).

        Parameters
        ----------
        input_shape : tuple of ints specifying shape of a single input
        rand_state : a RandomState object

        """
        self.input_shape = np.array(input_shape)
        self.output_shape = self.input_shape

    def forward(self, input_act):
        """
        Forward propagation.  This class is mostly wraps around _forward and does some extra
        asserts.  Child classes should overwrite _forward rather than this method.

        Parameters
        ----------
        input_act : numpy array, activations from the layer below; shape must either be the same as
            self.input_shape, or (NUMBER_OF_EXAMPLES,) + self.input_shape
        
        Returns
        -------
        output_act : numpy array, output activations from this layer; shape will be
            self.output_shape or (NUMBER_OF_EXAMPLES,) + self.output_shape, depending on the input

        """
        input_ndim = len(self.input_shape)
        assert input_act.shape[-input_ndim:] == tuple(self.input_shape), 'wrong input shape'
        many = (input_act.ndim > input_ndim)
        input_act = atleast(input_act, input_ndim+1)

        act = self._forward(input_act)

        assert act.shape[1:] == tuple(self.output_shape), 'wrong output shape'
        return act if many else act[0,...]

    def backward(self, grad_act, input_act):
        """
        Backward propagation.  This class is mostly wraps around _backward and does some extra
        asserts.  Child classes should overwrite _backward rather than this method.

        Parameters
        ----------
        grad_act : nump array, gradient of cost function with respect to the activations from this
            layer (usually calculated in the layer above and passed down during backward
            propagation), shape is self.output_shape or (NUMBER_OF_EXAMPLES,) + self.output_shape
        input_act : numpy array, activations from the layer below; shape must either be the same as
            self.input_shape, or (NUMBER_OF_EXAMPLES,) + self.input_shape

        Returns
        -------
        grad_input_act : numpy array, gradient of cost function with respect to the input
            activations this layer received, which is to be passed down to the layer below; shape
            will be self.input_shape or (NUMBER_OF_EXAMPLES,) + self.input_shape, depending on the
            input
        grad_params : 1D  numpy array of length self.num_params() (or None if self.num_params()==0),
            gradient of cost function with respect to the params of this layer
            
        """
        input_ndim = len(self.input_shape)
        output_ndim = len(self.output_shape)

        assert grad_act.shape[-output_ndim:] == tuple(self.output_shape), 'wrong grad input shape'
        assert input_act.shape[-input_ndim:] == tuple(self.input_shape), 'wrong input shape'
        assert ((grad_act.ndim==output_ndim and input_act.ndim==input_ndim)
                or grad_act.shape[0] == input_act.shape[0]), 'wrong number of samples'
        many = (input_act.ndim > input_ndim)
        input_act = atleast(input_act, input_ndim+1)
        grad_act = atleast(grad_act, output_ndim+1)

        grad_input_act, grad_params = self._backward(grad_act, input_act)

        assert grad_input_act.shape[1:] == tuple(self.input_shape), \
            'wrong input act grad shape'
        if self.num_params() > 0:
            grad_params = grad_params.ravel()
            assert grad_params.size == self.num_params(), 'wrong param grad shape'

        return (grad_input_act if many else grad_input_act[0,...], grad_params)

    ################################################################################################
    ### METHODS TO OVERWRITE IN CHILD CLASSES
    def num_params(self):
        """
        Returns the number of parameters in this layer
        """
        return 0

    def get_params(self):
        """
        Returns a 1D numpy array, length self.num_params(), with the parameters of this layer.
        """
        return None

    def set_params(self, params):
        """
        Sets the parameters of this layer
        
        Parameters
        ----------
        params : 1D numpy array, length self.num_params(), with the parameters of this layer

        """
        pass

    def _forward(self, input_act):
        """
        Forward propagation.

        Parameters
        ----------
        input_act : numpy array, activations from the layer below; shape is 
            (NUMBER_OF_EXAMPLES,) + self.input_shape
        
        Returns
        -------
        output_act : numpy array, output activations from this layer; shape will be
            (NUMBER_OF_EXAMPLES,) + self.output_shape

        """
        raise NotImplemented

    def _backward(self, grad_act, input_act):
        """
        Backward propagation.

        Parameters
        ----------
        grad_act : nump array, gradient of cost function with respect to the activations from this
            layer (usually calculated in the layer above and passed down during backward
            propagation), shape is (NUMBER_OF_EXAMPLES,) + self.output_shape
        input_act : numpy array, activations from the layer below; shape must either be the same as
            (NUMBER_OF_EXAMPLES,) + self.input_shape

        Returns
        -------
        grad_input_act : numpy array, gradient of cost function with respect to the input
            activations this layer received, which is to be passed down to the layer below; shape
            will be (NUMBER_OF_EXAMPLES,) + self.input_shape
        grad_params : 1D  numpy array of length self.num_params() (or None if self.num_params()==0),
            gradient of cost function with respect to the params of this layer
            
        """
        # returns next grad_act (layer below), and grad_params for this layer
        raise NotImplemented

class ConvLayer(Layer):
    def __init__(self, 
                 input_shape, 
                 num_filters=1, 
                 filter_shape=(3,3), 
                 init_from=None, 
                 rand_state=np.random):
        """
        Convolutional layer.

        Parameters
        ----------
        input_shape : tuple of ints specifying shape of a single input; this particular layer
            expects the input shape to be 3D (height x width x channels)
        num_filters : int, number of filters in this layer
        filter_shape : tuple specifying height and width of the filters (current implementation
            only square filters)
        init_from : (experimental feature) a dataset to use in initializing filters
        rand_state : a RandomState object

        """
        super(ConvLayer, self).__init__(input_shape)
        assert filter_shape[0]%2 == 1 and filter_shape[1]%2 ==1
        assert filter_shape[0] == filter_shape[1], 'Only square filters currently supported'
        if init_from is not None:
            # a bit of a hack to try out...
            assert init_from.shape[3] == input_shape[2]
            assert init_from.shape[0] > 5
            self.filters_ = np.zeros(filter_shape + (input_shape[2], num_filters), dtype='float32')
            for i in xrange(num_filters):
                sample = init_from[choice(15, init_from.shape[0]),...].mean(0)
                r_start = rand_state.randint(init_from.shape[1] - filter_shape[0])
                c_start = rand_state.randint(init_from.shape[2] - filter_shape[1])
                self.filters_[...,i] = sample[r_start:r_start+filter_shape[0], 
                                              c_start:c_start+filter_shape[1],
                                              ...]/10
        else:
            self.filters_ = rand_state.randn(*(filter_shape + (input_shape[2], num_filters)))
            self.filters_ /= np.sqrt(np.prod(self.filters_.shape[:-1]))
            self.filters_ = self.filters_.astype('float32')
        self.filter_shape = filter_shape
        self.filter_pad = (filter_shape[0]/2, filter_shape[1]/2)
        self.output_shape = np.array([self.input_shape[0] - filter_shape[0] + 1,
                                      self.input_shape[1] - filter_shape[1] + 1,
                                      num_filters])

    def viz(self, num_row=1):
        """
        Displays the filters in this layer (only makes sense for the first layer of a network)
        """
        num_filters = self.filters_.shape[-1]
        fig = plt.figure()
        num_col = int(np.ceil(float(num_filters)/num_row))
        
        for i in xrange(num_filters):
            ax = fig.add_subplot(num_row, num_col, i)
            imshow(self.filters_[...,i], ax=ax)

    def num_params(self):
        return np.prod(self.filters_.shape)

    def get_params(self):
        return self.filters_.ravel()

    def set_params(self, params):
        self.filters_ = params.reshape(self.filters_.shape)

    def _forward(self, input_act):
        fp = self.filter_pad
        act = batch_filter3D(input_act, self.filters_)
        act = act[:,fp[0]:-fp[0],fp[1]:-fp[1],:]
        return act

    def _backward(self, grad_act, input_act):
        # this is probably the trickiest method in this entire module...

        # input activation gradient -- notice that we have to flip the filters horizontally and
        # vertically
        rev_filters = np.fliplr(np.flipud(self.filters_))

        # note: opencv doesn't like arbitrary slices of numpy arrays, so we need to shuffle the
        # dimensions around a little bit

        # rev_filters will now be NUM_FILTERS x NUM_CHANNELS x ...
        rev_filters = np.rollaxis(np.rollaxis(rev_filters, 2, 0), 3, 0).copy()
        padded_grad_act = padarray(grad_act, self.filter_pad)
        # padded_grad_act will now be NUM_FILTERS x NUM_EXAMPLES x ...
        padded_grad_act = np.rollaxis(padded_grad_act, 3, 0).copy()
        grad_input_act = np.zeros(input_act.shape, dtype='float32')
        for z in xrange(input_act.shape[0]):
            for c in xrange(input_act.shape[-1]):
                for f in xrange(self.filters_.shape[-1]):
                    grad_input_act[z,:,:,c] +=  filter2D(padded_grad_act[f,z], rev_filters[f,c])

        # grad_input_act = grad_input_act.sum(-1)

        # params gradient
        grad_params = np.zeros((input_act.shape[1:4] + (grad_act.shape[-1],)), dtype='float32')
        # grad_act_ will now be NUM_FILTERS x NUM_EXAMPLES x ...
        grad_act_ = np.rollaxis(grad_act, 3, 0).copy()
        # padded_grad_act will now be NUM_CHANNELS x NUM_EXAMPLES x ...
        input_act = np.rollaxis(input_act, 3, 0).copy()
        for n in xrange(input_act.shape[1]):
            for c in xrange(input_act.shape[0]):
                for f in xrange(grad_act.shape[-1]):
                    grad_params[:,:,c,f] +=  filter2D(input_act[c,n], grad_act_[f,n])
        grad_params /= input_act.shape[1]

        r_border, c_border = grad_act.shape[1]/2, grad_act.shape[2]/2
        if grad_act.shape[1] %2 == 0:
            grad_params = grad_params[r_border:-r_border+1, c_border:-c_border+1,...]
        else:
            grad_params = grad_params[r_border:-r_border, c_border:-c_border,...]
        assert grad_params.shape == self.filters_.shape, 'wrong param grad shape'

        return grad_input_act, grad_params.ravel()

class MeanPoolingLayer(Layer):
    def __init__(self, input_shape, pool_size=2, rand_state=np.random):
        """
        Mean pooling layer.  There are no learnable parameters in this layer type.

        Parameters
        ----------
        input_shape : tuple of ints specifying shape of a single input
        pool_size : int, size of the pooling window (stride will be the same as this size, in other 
            words no overlap in the pooling)
        rand_state : a RandomState object

        """
        super(MeanPoolingLayer, self).__init__(input_shape)
        self.output_shape = self.input_shape / np.array([pool_size, pool_size, 1])
        self.pool_size = pool_size

    def _forward(self, input_act):
        act = downscale_local_mean(np.rollaxis(input_act, 0, 4),
                                   (self.pool_size, self.pool_size, 1, 1))
        return np.rollaxis(act, 3, 0)

    def _backward(self, grad_act, input_act):
        kron_kernel = np.ones((self.pool_size,self.pool_size))[np.newaxis,...,np.newaxis]
        grad_input_act = np.kron(grad_act, kron_kernel)/self.pool_size/self.pool_size
        return grad_input_act, None

class ReluLayer(Layer):
    """
    Rectified linear unit layer.  There are no learnable parameters in this layer type.
    """
    def _forward(self, input_act):
        return input_act * (input_act>0)

    def _backward(self, grad_act, input_act):
        return (input_act>0).astype('float')*grad_act, None

class SigmoidLayer(Layer):
    """
    Sigmoid unit layer.  There are no learnable parameters in this layer type.
    """
    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

    def _forward(self, input_act):
        return SigmoidLayer._sigmoid(input_act)

    def _backward(self, grad_act, input_act):
        out = SigmoidLayer._sigmoid(input_act)
        return out*(1.0-out)*grad_act, None

class DenseLayer(Layer):
    def __init__(self, input_shape, num_nodes=1, rand_state=np.random):
        """
        Dense/fully connected layer.

        Parameters
        ----------
        input_shape : tuple of ints specifying shape of a single input
        num_nodes : int, number of nodes in the layer
        rand_state : a RandomState object

        """
        super(DenseLayer, self).__init__(input_shape)
        self.output_shape = np.array([num_nodes])
        self.weights_ = rand_state.randn(np.prod(self.input_shape), num_nodes).astype('float32')
        self.weights_ /= np.sqrt(np.prod(self.weights_.shape))

    def num_params(self):
        return self.weights_.size

    def get_params(self):
        return self.weights_.ravel()

    def set_params(self, params):
        self.weights_ = params.reshape(self.weights_.shape)

    def _forward(self, input_act):
        input_act = input_act.reshape((-1,self.weights_.shape[0]))
        return np.dot(input_act, self.weights_)

    def _backward(self, grad_act, input_act):
        input_act = input_act.reshape((-1,self.weights_.shape[0]))

        grad_input_act = np.dot(grad_act, self.weights_.T)
        grad_input_act = grad_input_act.reshape((-1,) + tuple(self.input_shape))

        grad_params = np.array([np.outer(act, grad) for act, grad in zip(input_act, grad_act)])
        grad_params = grad_params.mean(0)

        return grad_input_act, grad_params

class BiasLayer(Layer):
    def __init__(self, input_shape, init_val=0, rand_state=np.random):
        """
        Bias layer.  For an input shape of [...] x N, this layer adds N bias terms.  E.g., for a 
        convolutional layer with an output of shape WxHxC where C is the number of channels/filters,
        this layer will contain C bias terms, one for each filter.

        Parameters
        ----------
        input_shape : tuple of ints specifying shape of a single input
        init_val : float, value to initialize all weights with
        rand_state : a RandomState object

        """
        super(BiasLayer, self).__init__(input_shape)
        # assert len(input_shape) == 3
        self.output_shape = np.array(input_shape)
        self.weights_ = np.ones(input_shape[-1]) * init_val

    def num_params(self):
        return self.weights_.size

    def get_params(self):
        return self.weights_.ravel()

    def set_params(self, params):
        self.weights_ = params.reshape(self.weights_.shape)

    def _forward(self, input_act):
        return input_act + self.weights_

    def _backward(self, grad_act, input_act):
        grad_input_act = grad_act
        # sum over the width and height dimensions (if any), average over all input examples
        grad_params = grad_act.mean(0)
        while grad_params.ndim > 1:
            grad_params = grad_params.sum(0)

        return grad_input_act, grad_params

class NNet(object):
    def __init__(self, layer_args, input_shape, rand_state=np.random):
        """
        Abstract neural net class.

        Parameters
        ----------
        layer_args : list of (LayerClass, kwargs) tuples where LayerClass is a class that inherits
            from the Layer class, and kwargs are to be passed into the constructor of that class.
            layer_args[0] is the first layer, closest to the input, and layer_args[-1] is the
            top-most layer.  The kwargs need not include the input_shape argument -- this will be
            determined automatically starting with the input_shape for the network (see below).
        input_shape : tuple of ints specifying shape of a single input to the network
        rand_state : a RandomState object
        
        """
        # layer_args is a list of (layer_class, layer_init_args) for first through last layer
        self.layers_ = []
        self.input_shape = input_shape
        for args in layer_args:
            layer_class, args = args
            args['rand_state'] = rand_state
            layer = layer_class(input_shape, **args)
            self.layers_.append(layer)
            # get input shape for the next layer
            input_shape = layer.output_shape

        self._rand_state = rand_state
        self._cache_acts = None

        # this will keep track of how many batches and epochs have been trained
        self.num_batch = 0
        self.num_epoch = 0

    def set_params(self, params):
        """
        Set parameters to the network (i.e. all the layer parameters).

        Parameters
        ----------
        params : numpy array of length self.num_params()

        """
        ind = 0
        for layer in self.layers_:
            num_params = layer.num_params()
            if num_params:
                layer.set_params(params[ind:ind+num_params])
            ind += num_params

    def get_params(self):
        """
        Returns a single numpy array of length self.num_params() with all the parameters (i.e. all
            the layer parameters concatenated into one vector).
        """
        return np.concatenate([layer.get_params() 
            for layer in self.layers_ if layer.get_params() is not None])

    def num_params(self):
        """
        Returns the number of (learnable) parameters in the entire network.
        """
        return np.sum([layer.num_params() for layer in self.layers_])

    def num_nodes(self):
        """
        Returns the number of nodes/neurons in the network.
        """
        return (np.sum(np.prod(layer.output_shape) for layer in self.layers_) + 
                np.prod(self.input_shape))

    def cost_for_params(self, params, x, y=None):
        """
        Calculates the cost of the network for the specified inputs and the specified network
        params.
        
        Parameters
        ----------
        params : numpy array of length self.num_params() specified network parameters
        x : input examples
        y : labels of the examples
        
        Returns
        -------
        cost : float

        """
        curr_params = self.get_params()
        self.set_params(params)
        cost = self.cost(x, y=y)
        # revert params
        self.set_params(curr_params)
        return cost
        
    def cost(self, x, y=None, final_acts=None):
        """
        Calculates the cost of the network for the specified inputs.  Child classes should
        implement _cost rather than this method.
        
        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        y : numpy array, training labels, shape should be (NUMBER_OF_EXAMPLES, shape of labels)
        final_acts : (optional) output of top-most layer in the network for the set of examples
        
        Returns
        -------
        cost : float

        """
        if final_acts is None:
            final_acts = self.forward(x)[-1]
        return self._cost(final_acts, y)

    def forward(self, x, batch_size=None):
        """
        Forward propagation through the whole network.

        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        
        Returns
        -------
        acts : list that contains a numpy array for each layer in the network; the first element in 
            the list is the array x itself, and each following array is the output of that layer for
            the given examples x

        """
        acts = [x]
        for layer in self.layers_:
            act = layer.forward(acts[-1])
            acts.append(act)
        return acts

    def forward_final(self, x, batch_size=None):
        """
        Forward propagation through the whole network; returns only output of final layer.

        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        batch_size : number of samples to process at a time (conserves memory)
        
        Returns
        -------
        acts : activations of the final layer

        """
        if batch_size is None or batch_size > x.shape[0]:
            batch_size = x.shape[0]

        ind = 0
        res = []
        while ind < x.shape[0]:
            acts = x[ind:ind+batch_size,...]
            for layer in self.layers_:
                acts = layer.forward(acts)
            res.append(acts)
            ind += batch_size
        return np.concatenate(res) if len(res)>1 else res[0]
        
    def param_grad(self, x, y=None, acts=None):
        """
        Calculate the gradient of the cost function with respect to all learnable parameters of this
        network.
        
        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        y : numpy array, training labels, shape should be (NUMBER_OF_EXAMPLES, shape of labels)
        acts : (optional) list that contains a numpy array for each layer in the network; the first
            element in the list is the array x itself, and each following array is the output of
            that layer for the given examples x
        
        Returns
        -------
        param_grad : numpy array of length self.num_params()

        """
        if acts is None:
            acts = self.forward(x)

        curr_act_grad = self.cost_grad(final_acts=acts[-1], y=y)
        param_grad = []

        for ind_from_end, layer in enumerate(reversed(self.layers_)):
            curr_act_grad, curr_param_grad = layer.backward(curr_act_grad, acts[-2-ind_from_end])
            if curr_param_grad is not None:
                param_grad.append(curr_param_grad)

        param_grad.reverse()
        return np.concatenate(param_grad)

    @staticmethod
    def get_batch(x, y=None, batch_size=128, batch_ind=0, inds=None):
        """
        Calculate the gradient of the cost function with respect to all learnable parameters of this
        network.
        
        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        y : numpy array, training labels, shape should be (NUMBER_OF_EXAMPLES, shape of labels)
        batch_size : number of examples to use in each batch
        batch_ind : which batch to return
        inds : a permuation of indexes for this dataset (numpy array of length x.shape[0])
        
        Returns
        -------
        batch_x : subset of at most batch_size examples in x
        batch_y : corresponding labels for this batch

        """
        if inds is None:
            inds = np.arange(x.shape[0])
        batch_x = x[inds[batch_ind*batch_size:(batch_ind+1)*batch_size],...]
        batch_y = None
        if y is not None:
            batch_y = y[inds[batch_ind*batch_size:(batch_ind+1)*batch_size],...]

        return batch_x, batch_y

    @staticmethod
    def get_num_batch(num_examples, batch_size):
        """
        Returns the number of batches for a given number of examples and given batch size.
        """
        return int(np.ceil(num_examples/float(batch_size)))

    def split_per_layer(self, vec):
        """ Given a vector with entries for each learnable parameter in the net, this method sums
        up the entries for each layer and returns a vector of such sums.  E.g., can be used to
        calculate absolute mean of weights in each layer."""
        split = []
        ind = 0
        for layer in self.layers_:
            split.append(vec[ind:layer.num_params()])
            ind += layer.num_params()

        return split

    def fit(self,
            x,
            y=None,
            val_x=None,
            val_y=None,
            val_freq=10,
            batch_size=128,
            num_epoch=10,
            momentum=0.9,
            learn_rate=0.01,
            learn_rate_decay=0.05,
            chill_out_iters=10,
            weight_decay=.0005,
            verbose=False):
        """
        Train the neural network via mini-batch gradient descent.

        Parameters
        ----------
        x : numpy array, training examples; shape should be (NUMBER_OF_EXAMPLES,) + self.input_shape
        y : numpy array, training labels, shape should be (NUMBER_OF_EXAMPLES, shape of labels)
        val_x : validation examples, similar shape as x
        val_y : validation labels, similar shape as y
        val_freq : validation will be performed every val_freq iterations
        batch_size : number of examples to use in each batch
        num_epoch : number of epochs to train for (maximum, may terminate earlier)
        learn_rate : initial learning rate, will decay as learning proceeds
        learn_rate_decay : at each iteration i the learning rate will be
            learn_rate/(i*learn_rate_decay+1)
        chill_out_iters : if there is no improvement in validation error after this many iterations
            of validation, the learning rate will be cut in half and the network will go back to
            the set of parameters that achieved the lower cost so far
        weight_decay : amount of weight decay to apply
        verbose : whether to print debug messages during training or not

        """

        if verbose:
            print '='*80
            print 'training net on %d samples' % x.shape[0]
            if val_x is not None:
                print 'using %d validation samples' % val_x.shape[0]
            print '='*80

        min_cost = 1e8
        velocity = np.zeros(self.num_params(), dtype='float32')
        best_params = self.get_params()
        stop = False
        no_improvement_iters = 0
        num_train = x.shape[0]
        num_batch = NNet.get_num_batch(num_train, batch_size)
        init_learn_rate = learn_rate

        start_time = time()
        for epoch in xrange(self.num_epoch, self.num_epoch + num_epoch):
            inds = self._rand_state.permutation(x.shape[0])
            if stop: 
                break
            for batch in xrange(num_batch):
                batch_x, batch_y = self.get_batch(
                        x, y=y, batch_size=batch_size, inds=inds, batch_ind=batch)
                assert batch_x.shape[0] > 0
                param_grad = self.param_grad(batch_x, y=batch_y)
                params = self.get_params()
                learn_rate = init_learn_rate/((epoch*num_batch + batch)*learn_rate_decay+1)
                velocity = (
                        momentum*velocity - 
                        learn_rate*param_grad - 
                        learn_rate*weight_decay*params)
                self.set_params(params + velocity)

                # check validation error every once in a while
                if (batch%val_freq == 0 and batch>0) or batch == num_batch-1:
                    if val_x is None:
                        val_x, val_y = batch_x, batch_y
                    val_acts = self.forward_final(val_x, batch_size=batch_size)
                    cost = self.cost(val_x, val_y, final_acts=val_acts)
                    # child classes don't necessarily have a concept of "accuracy" and might not
                    # implement the accuracy method
                    try:
                        acc = self.accuracy(val_acts, val_y)
                    except NotImplemented:
                        acc = np.nan

                    cost_diff = cost - min_cost

                    # if there has been significant regression in cost, chill out
                    if ((cost_diff > 0 and cost_diff/min_cost > 1) or 
                        no_improvement_iters>chill_out_iters):
                        self.set_params(best_params)
                        no_improvement_iters = 0
                        init_learn_rate /= 2
                        velocity = np.zeros_like(velocity, dtype='float32')
                        print 'cost was %.3e, chilling out...' % cost
                        cost = min_cost
                    elif cost < min_cost:
                        best_params = self.get_params()
                        min_cost = cost
                        no_improvement_iters = 0
                    else:
                        no_improvement_iters += 1

                    if verbose:
                        print 'epoch %03d, batch=%04d/%04d' % (epoch, batch+1, num_batch)
                        print 'cost=%.3e, min_cost=%.3e, acc=%.2f' % (cost, min_cost, acc)
                        print 'learn_rate=%.3e, velocity L1 norm %f' % (learn_rate, 
                                                                        np.abs(velocity).sum(0))
                        print '-'*80

        self.num_epoch = epoch
        end_time = time()
        print 'training complete [%.2f min]' % ((end_time-start_time)/60)

    def predict(self, x, batch_size=None):
        """
        Retruns the output of the final layer of this network.
        """
        return self.forward_final(x, batch_size)

    ################################################################################################
    ### METHODS TO OVERWRITE IN CHILD CLASSES
    def _cost(self, final_acts, y):
        """
        Calculates the cost of the network for the specified inputs.
        
        Parameters
        ----------
        final_acts : output of top-most layer in the network for a set of examples
        y : labels of the examples
        
        Returns
        -------
        cost : float

        """
        raise NotImplemented

    def accuracy(self, final_acts, y):
        """
        Child class can optionally implement this in case there is a notion of accuracy that is
        separate from cost (e.g. cross entropy cost versus classifcation accuracy).

        Parameters
        ----------
        final_acts : output of top-most layer in the network for a set of examples
        y : labels of the examples

        Returns
        -------
        accuracy : float

        """
        raise NotImplemented

    def cost_grad(self, final_acts, y):
        """
        Calculates the gradient of the cost function with respect to the top-most layer activations.
        
        Parameters
        ----------
        final_acts : output of top-most layer in the network for a set of examples
        y : labels of the examples
        
        Returns
        -------
        cost_grad : numpy array, same shape as the output_shape of the top-most layer.

        """
        raise NotImplemented

class SoftmaxNet(NNet):
    def __init__(self, layer_args, input_shape, rand_state=np.random):
        """
        Softmax (cross entropy) cost neural net.
        """
        super(SoftmaxNet, self).__init__(layer_args, input_shape, rand_state=rand_state)
        self.num_classes = self.layers_[-1].output_shape[0]

    def fit(self, x, y=None, **kwargs):
        assert y is not None, 'Labels must be passed in'
        assert tuple(np.unique(y)) == tuple(range(self.num_classes)), \
            'Labels should range from 0 to C-1 where C is the number of nodes in the last layer'
        binary_y = self.binarize_labels(y)
        if 'val_y' in kwargs:
            kwargs['val_y'] = self.binarize_labels(kwargs['val_y'])
        super(SoftmaxNet, self).fit(x, binary_y, **kwargs)

    def binarize_labels(self, y):
        """
        Turns discrete labels into binary vector labels.

        Parameters
        ----------
        y : numpy array of N integers from 0 to C-1
        
        Returns
        -------
        b : numpy array of shape Nx(C-1) s.t. b[i,j]=1 if y[i]==j, and b[i,k] for all k!=j
        """
        binary_y = np.zeros((len(y), self.num_classes))
        for c in xrange(self.num_classes):
            binary_y[y==c,c] = 1

        return binary_y

    def predict(self, x, batch_size=None):
        acts = self.forward_final(x, batch_size)
        return np.argmax(acts, axis=1)

    def _cost(self, final_acts, y):
        exp_act = safe_exp(final_acts)
        lse_act = safe_log(np.sum(exp_act, axis=1))
        return -np.mean(np.sum((y * (final_acts - lse_act[:,np.newaxis])), axis=1))

    def accuracy(self, final_acts, y):
        yp = np.argmax(final_acts, axis=1)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        return np.mean(yp == y)*100

    def cost_grad(self, final_acts, y):
        exp_act = safe_exp(final_acts)
        sum_exp = np.sum(exp_act, axis=1)
        return exp_act/sum_exp[:,np.newaxis] - y

