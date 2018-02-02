import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import xlogy


class Layer:
    def __init__(self, input_layers, input_size, output_size):
        self.num_input = input_size
        self.num_output = output_size

        self.input_layers = []
        if isinstance(input_layers, Layer):
            self.input_layers.append(input_layers)
        elif isinstance(input_layers, list):
            self.input_layers += input_layers

        self.output_layers = []

        for input_layer in self.input_layers:
            input_layer.add_output_layer(self)

        self.inp = None
        self.rng = np.random.RandomState(1234)

        self.conf = dict({'learning_rate': .1,
                          'momentum_rate': .1})

    @property
    def input_size(self):
        return self.num_input

    @property
    def output_size(self):
        return self.num_output

    def set_config(self, **kwargs):
        self.conf.update(kwargs)

    def add_output_layer(self, output_layer):
        assert isinstance(output_layer, Layer)
        self.output_layers.append(output_layer)

    def debug(self):
        pass

    def save(self, ofs):
        pass

    def load(self, ofs):
        pass

    def forward(self, inp=None):
        pass

    def backward_error(self, err=None):
        pass

    def backward_partial_gradient(self, out=None):
        pass


class DataLayer(Layer):
    def __init__(self, num_input):
        Layer.__init__(self, None, num_input, num_input)

    def set_data(self, input_data):
        """
        :param inp: np.1darray [batch, input_size]
        """
        self.inp = input_data

    def forward(self, inp=None):
        """
        :param inp: np.1darray [batch, input_size]
        """
        if isinstance(inp, np.ndarray):
            self.inp = inp
        return self.inp


class LossLayer(Layer):
    def __init__(self, input_layer):
        Layer.__init__(self, input_layer, input_layer.output_size, 1)

        self.labels = None
        self.err = None  # [batch, num_input] calculated in forward

    def set_labels(self, labels):
        """
        :param inp: np.2darray [batch, input_size]
        """
        self.labels = labels

    def backward_error(self, err=None):
        """
        :param err: np.2darray (num_output, batch); for LossLayer, this is None
        :return: np.2darray (num_input, batch)
        """
        return np.transpose(self.err)



class L2LossLayer(LossLayer):
    def __init__(self, input_layer):
        LossLayer.__init__(self, input_layer)

    def forward(self, inp=None):
        """
        :param inp: np.2darray [batch, input_size]
        """
        batch, _ = inp.shape
        self.inp = inp
        total_loss = np.sum(np.square(inp - self.labels)) / (2 * batch)

        self.err = (self.inp - self.labels) / batch
        return np.sqrt(total_loss)

    def backward_partial_gradient(self, out=None):
        pass


class SigmoidCrossEntropyLossLayer(LossLayer):
    def __init__(self, input_layer):
        assert input_layer.output_size == 1 and ">>> classification with two labels (0, 1)!!"
        LossLayer.__init__(self, input_layer)

    def forward(self, inp=None):
        """
        :param inp: np.2darray [batch, input_size]
        """
        batch, _ = inp.shape
        self.inp = inp
        out = sigmoid(inp)
        self.err = - (self.labels * (1 - out) - (1 - self.labels) * out) / batch

        total_loss = np.sum(xlogy(self.labels, out) + xlogy(1 - self.labels, 1 - out))  # = np.sum(self.labels * np.log(out))
        return - total_loss / batch

    def backward_partial_gradient(self, out=None):
        pass


class SoftmaxCrossEntropyLossLayer(LossLayer):
    # # to be test
    def __init__(self, input_layer):
        assert isinstance(input_layer, Layer)
        LossLayer.__init__(self, input_layer)

    def forward(self, inp=None):
        """
        :param inp: np.2darray [batch, input_size]
        """
        batch, _ = inp.shape
        self.inp = inp
        e_x = np.exp(np.transpose(inp) - np.max(inp, axis=1))
        o = np.transpose(e_x / e_x.sum(axis=0))
        # print (np.max(inp))

        labels_sum = np.sum(self.labels, axis=1).reshape((-1, 1))
        self.err = - (self.labels - labels_sum * o) / batch

        # # for loop for computing self.err
        # batch, c = o.shape
        # self.err_ = np.zeros((batch, c))
        # for i in range(batch):
        #     d_sum = np.sum(self.labels[i, :])
        #     for j in range(c):
        #         self.err_[i, j] = - (self.labels[i, j] - d_sum * o[i, j])

        total_loss = np.sum(xlogy(self.labels, o))  # = np.sum(self.labels * np.log(out))
        return - total_loss / batch

    def backward_partial_gradient(self, out=None):
        pass


class DenseLayer(Layer):
    def __init__(self, input_layer, num_output, use_bias=False):
        assert isinstance(input_layer, Layer)
        Layer.__init__(self, input_layer, input_layer.output_size, num_output)

        self.w = np.array(self.rng.uniform(low=-0.5, high=0.5, size=(self.num_input, self.num_output)))
        self.dw = np.zeros(self.w.shape)
        self.b = np.zeros((self.num_output,))
        self.db = np.zeros(self.b.shape)

        self.use_bias = use_bias
        if use_bias:
            self.b = np.array(self.rng.uniform(low=-0.1, high=0.1, size=(self.num_output,)))

    def save(self, ofs):
        np.savetxt(ofs, self.w, delimiter=',')
        if self.use_bias:
            np.savetxt(ofs, self.b, delimiter=',')

    def load(self, ofs):
        w = np.genfromtxt(ofs, delimiter=',', max_rows=self.w.shape[0])
        self.w = np.reshape(w, self.w.shape)

        if self.use_bias:
            b = np.genfromtxt(ofs, delimiter=',', max_rows=self.b.shape[0])
            self.b = np.reshape(b, self.b.shape)

    def forward(self, inp=None):
        """
        :param inp: np.1darray [batch, input_size]
        """
        self.inp = inp
        out = np.dot(inp, self.w)
        if self.use_bias:
            out += self.b
        return out

    def backward_error(self, err=None):
        """
        :param err: np.2darray (num_output, batch)
        :return: np.2darray (num_input, batch)
        """
        w_grad = np.transpose(np.dot(err, self.inp))
        if self.use_bias:
            b_grad = np.sum(err, axis=1)
        else:
            b_grad = None
        err = np.dot(self.w, err)

        self._update_weights(w_grad, b_grad)
        return err

    def _update_weights(self, w_grad, b_grad):
        # # update weights
        self.dw *= self.conf['momentum_rate']
        self.dw += self.conf['learning_rate'] * w_grad
        self.w -= self.dw

        if self.use_bias and isinstance(b_grad, np.ndarray):
            self.db *= self.conf['momentum_rate']
            self.db += self.conf['learning_rate'] * b_grad
            self.b -= self.db

    def backward_partial_gradient(self, partial_grad=None):
        """
        :param partial_grad: np.3darray (batch, num_output, final_output_size)
        :return np.3darray (batch, num_input, final_output_size)
        """
        partial_w_grad = np.tensordot(partial_grad, self.inp, axes=0)

        return np.dot(self.w, partial_grad).transpose((1, 0, 2))

    def debug(self):
        return 'weights: %s' % str(self.w)


class SigmoidLayer(Layer):
    def __init__(self, input_layer):
        output_size = input_layer.output_size
        Layer.__init__(self, input_layer, output_size, output_size)

        self.y_p = None

    def forward(self, inp=None):
        """
        :param inp: np.1darray [batch, input_size]
        """
        out = sigmoid(inp)
        # # derivative
        self.y_p = out * (1 - out)  # point-wise multiplication
        return out

    def backward_error(self, err=None):
        """
        :param err: np.2darray (num_output, batch)
        :return: np.2darray (num_input, batch)
        """
        return np.transpose(self.y_p) * err  # point-wise multiplication

    def backward_partial_gradient(self, partial_grad=None):
        return partial_grad * self.y_p


class ReLULayer(Layer):
    def __init__(self, input_layer):
        output_size = input_layer.output_size
        Layer.__init__(self, input_layer, output_size, output_size)

        self.y_p = None

    def forward(self, inp=None):
        """
        :param inp: np.1darray [batch, input_size]
        """
        out = np.array(inp)
        self.y_p = np.ones(inp.shape)
        indexes = np.where(inp < 0)
        out[indexes] = 0
        self.y_p[indexes] = 0
        return out

    def backward_error(self, err=None):
        """
        :param err: np.2darray (num_output, batch)
        :return: np.2darray (num_input, batch)
        """
        return np.transpose(self.y_p) * err  # point-wise multiplication

    def backward_partial_gradient(self, partial_grad=None):
        pass


class SoftmaxLayer(Layer):
    # # to be test
    def __init__(self, input_layer):
        output_size = input_layer.output_size
        Layer.__init__(self, input_layer, output_size, output_size)

        self.y_p = None

    def forward(self, inp=None):
        """
        :param inp: np.1darray [batch, input_size]
        """
        self.inp = inp
        e_x = np.exp(inp - np.max(inp, axis=1))
        o = np.transpose(np.transpose(e_x) / e_x.sum(axis=1))

        batch, c = o.shape
        res = np.zeros((batch, c, c))
        for i in range(batch):
            res[i, :, :] = -np.tensordot(o[i, :], o[i, :], axes=0)
            np.fill_diagonal(res[i, :, :], o[i, :] * (1 - o[i, :]))
        self.y_p = res
        return o

    def backward_error(self, err=None):
        """
        :param err: np.2darray (num_output, batch)
        :return: np.2darray (num_input, batch)
        """
        c, batch = err.shape
        res = np.zeros(self.input_size, batch)
        for i in range(batch):
            res[:, i] = np.dot(self.y_p[i, :, :], err[:, i])
        return res
