SimpleConvnet
==============

This is a basic implementation of a convolutional neural net.  It is meant primarily for pedagogical
purposes -- if you are looking for a fully featured, efficient implementation, there are a few other
options I'd recommend:

* [cuda-convnet](https://code.google.com/p/cuda-convnet/)
* [caffe](http://caffe.berkeleyvision.org/)

### Installing
To install, run:

```bash
python setup.py install
```

### Dependencies
* matplotlib 1.1
* numpy 1.6
* scipy 0.10
* scikit-image 0.9
* scikit-learn 0.14
* opencv 2.4

### Running unit tests
To run unit tests you will need nosetests installed.  You can run all unit tests with this:

```
nosetests -v
```
