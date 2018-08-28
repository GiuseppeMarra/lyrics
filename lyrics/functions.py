import tensorflow as tf
import abc
import math


FUNCTIONS = {}

eps = 1e-12

def reno(x):
    return tf.maximum(1 - abs(2 * x - 1), 0)

class AbstractFunction(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.precomputed = None

    @abc.abstractmethod
    def __call__(self, *a):
        raise NotImplementedError('You must define "__call__" function to use this base class')

class Learner(AbstractFunction):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Learner, self).__init__()

    def cost(self, labels, *inputs):
        raise NotImplementedError('users must define "cost" function to use this base class')


class RegularizedFunction(AbstractFunction):
    def __init__(self):
        super(RegularizedFunction, self).__init__()

    def regularization_cost(self):
        raise NotImplementedError('users must define "regularization_cost" function to use this base class')



class Slice(Learner):
    #Wrapper for slicing the output of a function. Commonly used in conjuction with share.
    def __init__(self, function, axis):
        super(Slice, self).__init__()
        self.function = function
        self.axis = axis

    def __call__(self, input=None):
        output = self.function(input)
        if len(output.get_shape())==2:
            return output[:,self.axis]
        else:
            return output[:,:,self.axis]


class FFNClassifier(Learner):

    def __init__(self, name, input_size, n_classes, hidden_layers = (10,)):
        super(FFNClassifier, self).__init__()
        self.name = name
        self.output_size = n_classes
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self._reuse = False

    def _internal_(self,x):
        with tf.variable_scope(self.name, reuse = self._reuse):
            for hidden_size in self.hidden_layers:
                x = tf.layers.dense(x, hidden_size, activation=tf.nn.sigmoid)
            x = tf.layers.dense(x, self.output_size)
            activation = tf.nn.softmax if self.output_size > 1 else tf.sigmoid
            y = activation(x)
            return y,x

    def __call__(self,x):
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [-1, self.input_size])
        y,_ = self._internal_(x)
        self._reuse = True
        return y

    def cost(self, labels, *input):
        labels = tf.cast(labels, tf.float32)
        x = tf.cast(input[0], tf.float32)
        _, logits = self._internal_(x)
        self._reuse = True
        if self.output_size > 1:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)


class SupervisionFunction(AbstractFunction):

    def __init__(self, tensor):
        super(SupervisionFunction, self).__init__()
        self.tensor = tensor

    def __call__(self, *x):
        return self.tensor

class BinaryIndexFunction(AbstractFunction):

    def __init__(self, name, k1=None,k2=None, var = None):
        super(BinaryIndexFunction, self).__init__()
        if k1 is not None and k2 is not None:
            self.var = tf.get_variable(name, initializer= -4 * tf.ones([k1,k2]))
        elif var is not None:
            self.var = var
        else:
            raise Exception("You must provide both k1 and k2, or alternatively a single var")

    def __call__(self, idx1, idx0):
        idx0 = tf.cast(idx0, tf.int64)
        idx1 = tf.cast(idx1, tf.int64)
        idx = tf.concat((idx1, idx0), axis=1)
        res = tf.gather_nd(params=self.var, indices=idx)
        return tf.sigmoid(res)

class StaticBinaryIndexFunction(AbstractFunction):

    def __init__(self, var):
        super(StaticBinaryIndexFunction, self).__init__()
        self.var = var

    def __call__(self, idx1, idx0):
        idx0 = tf.cast(idx0, tf.int64)
        idx1 = tf.cast(idx1, tf.int64)
        idx = tf.concat((idx1, idx0), axis=1)
        res = tf.gather_nd(params=self.var, indices=idx)
        return res

class NotDifferentiableEqual():


    def __call__(self, a, b):

        return tf.cast(tf.equal(a,b), tf.float32)

class L2SimilarityFunction(AbstractFunction):

    def __call__(self, x, y):
        # return 1 - tf.tanh(tf.reduce_mean(tf.squared_difference(x,y),axis=1))
        return 1/(1+tf.reduce_sum(tf.squared_difference(x,y), axis=1))

class CosineSimilarity(AbstractFunction):

    def __call__(self,x,mu):

        xnorm = tf.norm(x, axis=1, keep_dims=True, name="xnorm")
        mnorm = tf.norm(mu, axis=1, keep_dims=True, name="mnorm")
        a = tf.reduce_sum((xnorm * mnorm), axis=1, keep_dims=True) / (xnorm * mnorm + eps)
        return a

class NaryFFNClassifier(Learner):

    def __init__(self, name, input_size, n_classes, hidden_layers = (10,)):
        super(NaryFFNClassifier, self).__init__()
        self.name = name
        self.output_size = n_classes
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self._reuse = False

    def _internal_(self,x):
        with tf.variable_scope(self.name, reuse = self._reuse):
            for hidden_size in self.hidden_layers:
                x = tf.layers.dense(x, hidden_size, activation=tf.nn.sigmoid)
            x = tf.layers.dense(x, self.output_size)
            activation = tf.nn.softmax if self.output_size > 1 else tf.sigmoid
            y = activation(x)
            return y,x

    def __call__(self,*x):
        x = tf.concat(x,1)
        x = tf.reshape(x, [-1, self.input_size])
        y,_ = self._internal_(x)
        self._reuse = True
        return y

    def cost(self, labels, *input):
        input = tf.concat(input,1)
        _, logits = self._internal_(input)
        self._reuse = True
        if self.output_size > 1:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)




def _RBFLayer(x, input_size, output_size, sigma=1):


    dims_x = tf.shape(x)
    batch_size = dims_x[0]

    #Centers
    mu = tf.get_variable(name="centers", shape=[output_size, input_size], initializer=tf.truncated_normal_initializer)

    #Cosine Distance
    xnorm = tf.norm(x, axis=1, keep_dims=True, name="xnorm")
    mnorm = tf.norm(mu, axis=1, keep_dims=True, name="mnorm")
    similarity = tf.matmul(x / (xnorm + eps), tf.transpose(tf.div(mu,(mnorm + eps))))
    dist = tf.square(1 - similarity)

    return (1/(math.sqrt(2*math.pi*sigma)))*tf.exp(-0.5*dist/sigma)
    # return tf.exp(-0.5*dist/sigma)



class RBFClassifier(Learner):

        def __init__(self, label, input_size, hidden_size, output_size=10, sigma=1):
            super(RBFClassifier, self).__init__()
            self.label = label
            self.hidden_size = hidden_size
            self.sigma = sigma
            self.output_size = output_size
            self.input_size = input_size

        def _internal_(self, input):
            with tf.variable_scope("RBFClassifier_of_" + self.label, reuse=tf.AUTO_REUSE) as scope:


                h1 = _RBFLayer(input, input_size=self.input_size, output_size=self.hidden_size, sigma=self.sigma)
                w2 = tf.get_variable(name="w2", shape=[self.hidden_size, self.output_size], initializer=tf.truncated_normal_initializer)
                b2 = tf.get_variable(name="b2", shape=[self.output_size], initializer=tf.zeros_initializer)
                a2 = tf.matmul(h1, w2) + b2
                # a2 = tf.reduce_min(h1, axis=1)
                return a2


        def __call__(self, input=None):

                logits = self._internal_(input=input)
                y =  tf.nn.softmax(logits) if self.output_size > 1 else tf.sigmoid(logits)

                return y

        def cost(self, labels, input=None):
            with tf.name_scope("CostFunction"):
                logits = self._internal_(input=input)
                if self.output_size>1:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=labels))
                else:
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                              labels=labels))
            return loss

        def evaluate(self, labels, input=None, reuse=True):
            with tf.name_scope("Accuracy"):
                logits = self._internal_(input=input)
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                return accuracy

class FFNGenerator(Learner):

    def __init__(self, name, input_sizes, output_size, hidden_layers = (10,)):
        super(FFNGenerator, self).__init__()
        self.name = name
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.input_size = sum(input_sizes)
        self._reuse = False

    def _internal_(self,x):
        with tf.variable_scope(self.name, reuse = self._reuse):
            for hidden_size in self.hidden_layers:
                x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            y = tf.layers.dense(x, self.output_size)
            return y

    def __call__(self,*x):
        x = tf.concat(x, axis=1)
        x = tf.reshape(x, [-1, self.input_size])
        y  = self._internal_(x)
        self._reuse = True
        return y

    def cost(self, labels, *input):
        y = self._internal_(input[0])
        self._reuse = True
        loss = tf.reduce_sum(tf.squared_difference(y,labels), axis=1)
        return tf.reduce_mean(loss)

