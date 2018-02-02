import time
from .layer import *


def nn_forward_propagate(network_layers, inp=None, predicted=False):
    num_layers = len(network_layers)
    if predicted:
        num_layers -= 1
    for i in range(num_layers):  # for each layer
        inp = network_layers[i].forward(inp)
        # print 'FP at layer:', i, ' output:', layer['output']
    return inp


def nn_backward_propagate(network_layers):
    err = None
    num_layers = len(network_layers)
    for i in reversed(range(0, num_layers)):
        err = network_layers[i].backward_error(err)


class NeuralNetwork:
    def __init__(self, name, learning_rate=0.1, momentum_rate=0.1):
        self.name = name
        self.layers = []

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def add(self, *args):
        for layer in args:
            if not isinstance(layer, Layer):
                raise TypeError('The added layer must be '
                                'an instance of class Layer. '
                                'Found: ' + str(layer))

            self.layers.append(layer)

    def save(self, fmt=None):
        if not fmt:
            fmt = 'nn-%s.txt'
        filename = fmt % self.name
        ofs = open(filename, 'wb+')
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].save(ofs)

    def load(self, fmt=None):
        if not fmt:
            fmt = 'nn-%s.txt'
        filename = fmt % self.name
        import os.path

        if os.path.isfile(filename):
            ofs = open(filename, 'rb')
            num_layers = len(self.layers)
            for i in range(num_layers):
                self.layers[i].load(ofs)
            print ('>>>> load saved parameters for %s successfully!' % self.name)
            return True
        return False

    def build(self):
        assert isinstance(self.layers[-1], LossLayer)
        assert isinstance(self.layers[0], DataLayer)
        for layer in self.layers:
            layer.set_config(learning_rate=self.learning_rate, momentum_rate=self.momentum_rate)

    def fit(self, x, y, n_epoch, batch_size=1, verbose=0):
        if len(x.shape) == 1:
            x = np.array([x]).transpose()
        if len(y.shape) == 1:
            y = np.array([y]).transpose()
        assert x.shape[1] == self.layers[0].input_size
        assert y.shape[1] == self.layers[-2].output_size

        """ x.shape: samples, input_size of DataLayer
            y.shape: samples, output_shape of last layer
            assert x.shape[0] == y.shape[0]
        """
        assert x.shape[0] == y.shape[0] and 'rows of input and output must match!'
        r = x.shape[0]
        start = time.time()
        for epoch in range(n_epoch + 1):
            sum_error = 0
            r_perm = np.random.permutation(r)

            for i in range(0, r, batch_size):  # for each pattern
                ie = i + batch_size
                if ie > r:
                    ie = r
                batch = r_perm[i:ie]

                self.layers[0].set_data(x[batch, :])
                self.layers[-1].set_labels(y[batch, :])

                sum_error += nn_forward_propagate(self.layers)

                nn_backward_propagate(self.layers)

            if verbose == 1 and epoch % 100 == 0:
                print('>>> epoch=%d, error=%.9f' % (epoch, sum_error))
        end = time.time()
        print ('>>>> NN (%s) fitting time is %.4fs.' % (self.name, (end - start)))

    def train(self, x, y, n_epoch, batch_size=1, verbose=0):
        return self.fit(x, y, n_epoch, batch_size, verbose)

    def predict(self, x):
        """ x: [samples, input_size of DataLayer] or input_size of DataLayer
        """
        if x.shape == self.layers[0].input_size:
            x = x.reshape((1, ) + self.layers[0].input_size)
        elif x.shape[1] == self.layers[0].input_size:
            pass
        else:
            print (TypeError('The parameter should be compatible with input_size of DataLayer!'))
        return nn_forward_propagate(self.layers, x, predicted=True)

    def score(self, x, y):
        pass

    def output(self):
        print ('\n=============================================================')
        for i in range(1, len(self.layers) - 1):
            layer = self.layers[i]
            deb_msg = layer.debug()
            if deb_msg:
                print ('layer:', i, deb_msg)
        print ('=============================================================\n')


def linear_test():
    np.random.seed(1000)
    np.random.RandomState()

    inp = DataLayer(2)
    out = DenseLayer(inp, 1)
    loss = L2LossLayer(out)

    # Example a linear function
    batches = 10
    x1 = np.random.random(batches) * 3
    x2 = np.random.random(batches)
    y0 = - 2 * x1 + x2
    data = np.array([x1, x2]).transpose()

    nn_train = NeuralNetwork('linear(-2x_1 + x_2)', learning_rate=.1, momentum_rate=.1)
    nn_train.add(inp, out, loss)
    nn_train.build()

    nn_train.fit(data, y0, 1000, batch_size=8, verbose=0)
    nn_train.output()


def xor_test():
    ones, zeros = np.ones(500, dtype=np.int32), np.zeros(500, dtype=np.int32)
    x1 = np.concatenate([ones, ones, zeros, zeros], axis=0)
    x2 = np.concatenate([ones, zeros, ones, zeros], axis=0)
    indexes = np.random.permutation(x1.shape[0])
    x1, x2 = x1[indexes], x2[indexes]

    y = np.logical_xor(x1, x2)

    data = np.array([x1, x2]).transpose().astype(np.float64)

    inp = DataLayer(2)
    dense1 = DenseLayer(inp, 3)
    dense2 = SigmoidLayer(dense1)

    # out = SigmoidLayer(dense3)
    dense3 = DenseLayer(dense2, 1)
    loss = SigmoidCrossEntropyLossLayer(dense3)
    y0 = y.astype(np.float32).reshape((-1, 1))

    # dense3 = DenseLayer(dense2, 2)
    # loss = SoftmaxCrossEntropyLossLayer(dense3)
    # y0 = np.stack([y, np.logical_not(y)], axis=1).astype(np.int32).astype(np.float64)

    nn_train = NeuralNetwork('xor_test', learning_rate=.1, momentum_rate=0.1)
    nn_train.add(inp, dense1, dense2, dense3, loss)

    nn_train.build()

    nn_train.fit(data,  y0, 2000, batch_size=128, verbose=1)
    nn_train.output()

    test_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    test_y = nn_train.predict(test_data)
    print (test_data, sigmoid(test_y))

if __name__ == '__main__':
    linear_test()
    xor_test()
