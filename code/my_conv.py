"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class MyConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.input = input

        # initialize weights to temporary values until we know the
        # shape of the output feature maps
        W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # replace weight values with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W.set_value(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class NLPNet(object):

    def __init__(self, 
        batch_size=500,
        layers=1,
        ishape=(28, 28), 
        conv_filter_shape=(5, 5), 
        maxpool_filter_shape=(2, 2),
        nkerns=[1, 50, 50, 10]):
        self.ishape = ishape
        # allocate symbolic variables for the data
        self.x = T.matrix('x')   # the data is presented batches of rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.batch_size = batch_size
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        rng = numpy.random.RandomState(23455)

        layer0_input = self.x.reshape((batch_size, layers, self.ishape[0], self.ishape[1]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = MyConvPoolLayer(rng, input=layer0_input,
                                 filter_shape=(nkerns[0], 1, 
                                               conv_filter_shape[0], 
                                               conv_filter_shape[1]), 
                                 poolsize=maxpool_filter_shape)
        layer0_output_x = (ishape[0] - conv_filter_shape[0] + 1)/maxpool_filter_shape[0]
        layer0_output_y = (ishape[1] - conv_filter_shape[1] + 1)/maxpool_filter_shape[1]

        layer1_input = layer0.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer1 = HiddenLayer(rng, input=layer1_input, 
                             n_in=nkerns[0] * layer0_output_x * layer0_output_y,
                             n_out=nkerns[1], activation=T.tanh)

        layer2 = HiddenLayer(rng, input=layer1.output, 
                             n_in=nkerns[1], n_out=nkerns[2],
                             activation=T.tanh)

        layer3 = LogisticRegression(input=layer2.output,
                                    n_in=nkerns[2], n_out=nkerns[3])

        # classify the values of the fully-connected sigmoidal layer

        # the cost we minimize during training is the NLL of the model
        self.cost = layer3.negative_log_likelihood(self.y)
        self.errors = layer3.errors
        # create a list of all model parameters to be fit by gradient descent
        self.params = layer3.params + layer2.params + layer1.params\
                      + layer0.params


    def train(self, datasets, learning_rate=0.1, n_epochs=200):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size
        n_valid_batches /= self.batch_size
        n_test_batches /= self.batch_size

        index = T.lscalar()  # index to a [mini]batch
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function([index], self.errors(self.y),
                 givens={
                    self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        validate_model = theano.function([index], self.errors(self.y),
                givens={
                    self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates dictionary by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = {}
        for param_i, grad_i in zip(self.params, grads):
            updates[param_i] = param_i - learning_rate * grad_i

        train_model = theano.function([index], self.cost, updates=updates,
              givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]})   


        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = epoch * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_train_batches, \
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                               'model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i,'\
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    net = NLPNet()
    datasets = load_data("../data/mnist.pkl.gz")      
    net.train(datasets)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
