import re
import csv
import numpy as N

import theano
import theano.tensor as T

from my_conv import NLPNet
from embeddings import Embeddings
from sklearn import cross_validation

WORD_DIM = 50
MAX_WORDS = 50

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(N.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(N.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_datasets():
    print "Loading embeddings..."
    embeddings = Embeddings("Sentiment140/words.lst", 
    						"Sentiment140/embeddings/embeddings.txt")

    print "Loading datasets..."
    ftrain = csv.reader(open("Sentiment140/training.2000.processed.noemoticon.csv"),
    					delimiter=",", quotechar="\"")
    ftest = csv.reader(open("Sentiment140/testdata.manual.2009.06.14.csv"),
      				   delimiter=",", quotechar="\"")
    token_pattern = re.compile(r"([a-z0-9]+|[\.\!\?\,\-\'])", re.I)

    train, valid, test = [], [], []
    for x, fi in [(train, ftrain), (test, ftest)]:
        sents, labels = [], []
        for i, row in enumerate(fi):
            label, text = int(row[0]), row[-1]
            if x==test and label==2: continue
            if label==4: label = 1

            words = token_pattern.findall(text)
            sent, l = [], 0
            for w in words:
                if embeddings.has_word(w):
                    l += 1
                    sent.extend(embeddings.vector(w))
            for i in xrange(MAX_WORDS - l):
                sent.extend([0]*WORD_DIM)
            sents.append(sent)
            labels.append(label)
        
        x.extend([sents, labels])

    (x_train, x_valid, y_train, y_valid)\
        = cross_validation.train_test_split(train[0], train[1],
                                            test_size=0.2,
                                            random_state=1)
    valid.extend([x_valid, y_valid])
    train = [x_train, y_train]

    return [shared_dataset(train), shared_dataset(valid), shared_dataset(test)]


def main():
    net = NLPNet(ishape=(WORD_DIM, MAX_WORDS), 
                 conv_filter_shape=(5, WORD_DIM),
                 maxpool_filter_shape=(1, WORD_DIM), 
                 nkerns=[1, 50, 50, 2],
                 batch_size=50)
    datasets = load_datasets()
    net.train(datasets)


if __name__ == "__main__":
	main()
