import tensorflow as tf
import abc


FUNCTIONS = {}


def reno(x):
    return tf.maximum(1 - abs(2 * x - 1), 0)

class Function(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def call(self, a=None):
        raise NotImplementedError('users must define "call" function to use this base class')

class Learner(Function):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def call(self, a=None):
        raise NotImplementedError('users must define "call" function to use this base class')

    @abc.abstractmethod
    def cost(self, labels, inputs):
        raise NotImplementedError('users must define "call" function to use this base class')

    @abc.abstractmethod
    def evaluate(self, labels, inputs):
        raise NotImplementedError('users must define "call" function to use this base class')


class Slice(Learner):
    #Wrapper for slicing the output of a function. Commonly used in conjuction with share.
    def __init__(self, function, axis):
        self.function = function
        self.axis = axis

    def call(self, input=None):
        return self.function.call(input)[:, self.axis]

    def cost(self, labels, input=None):
        raise Exception("Slice functions cost cannot be used. Use the cost on the sliced function")

    def evaluate(self, labels, input=None):
        raise Exception("Slice functions evaluate cannot be used. Use the evaluate on the sliced function")


class Share(Learner):
    #Wrapper for sharing a function between different predicates over the same domain.
    #Useful when predicates are outputs of a NN model in order to share all the hidden layers.

    def __init__(self, function, domain):
        self.function = function
        self.tensor = function.call(domain.tensor)

    def call(self, input=None):
        return self.tensor

    def cost(self, labels, input=None):
        return self.function.cost(labels)

    def evaluate(self, labels, input=None):
        return self.function.evaluate(labels)


class FFNClassifier(Learner):

    def __init__(self, label, input_size, num_hidden_layers=1, size_layers=(20,)):
        self.label = label
        self.num_hidden_layers = num_hidden_layers
        with tf.variable_scope("FFNClassifier_of_"+self.label, reuse=False) as scope:
            for i in range(self.num_hidden_layers):
                output_size = size_layers[i]
                w = tf.get_variable(name="w" + str(i) + "_" + self.label,
                                        shape=[input_size, output_size],
                                        initializer=tf.random_normal_initializer())
                b = tf.get_variable(name="b" + str(i) + "_" + self.label,
                                        shape=[output_size],
                                        initializer=tf.constant_initializer(0.0))
                input_size = output_size
            w = tf.get_variable(name="wlast_" + self.label, shape=[input_size,1], initializer=tf.random_normal_initializer())
            b = tf.get_variable(name="blast_"+ self.label, shape=[1],initializer=tf.constant_initializer(0.0) )

    def call(self, input=None):
        if input is None:
            raise Exception("No input provided for FFN Classifier call()")
        with tf.variable_scope("FFNClassifier_of_"+self.label, reuse=True) as scope:
            h = input
            for i in range(self.num_hidden_layers):
                w = tf.get_variable(name="w" + str(i) + "_" + self.label)
                b = tf.get_variable(name="b" + str(i) + "_" + self.label)
                h = tf.sigmoid(tf.matmul(h,w) +b)
            w = tf.get_variable(name="wlast_" + self.label)
            b = tf.get_variable(name="blast_"+ self.label)
            h = reno(tf.matmul(h, w) + b)
            self.output = h
            return h

    def cost(self, labels, input=None):
        if input is None:
            raise Exception("No input provided for FFN Classifier cost()")
        return tf.reduce_mean(tf.square(self.call(input) - labels))