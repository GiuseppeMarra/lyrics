import abc
import tensorflow as tf


class FuzzyLogic(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def weak_conj(args):
        raise NotImplementedError('users must define "weak_conj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def strong_disj(args):
        raise NotImplementedError('users must define "strong_disj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def forall(a, axis):
        raise NotImplementedError('users must define "forall" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def exists(a, axis):
        raise NotImplementedError('users must define "exists" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def exists_n(a, axis, n):
        raise NotImplementedError('users must define "exists_n" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def negation(a):
        raise NotImplementedError('users must define "negation" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def implication(a, b):
        raise NotImplementedError('users must define "implies" to use this base class')


class Lukasiewicz(FuzzyLogic):

    @staticmethod
    def weak_conj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.reduce_min(arg, axis=new_axis)

    @staticmethod
    def strong_disj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.minimum(1., tf.reduce_sum(arg, axis=new_axis))

    @staticmethod
    def forall(a, axis):
        # return tf.reduce_sum(a, axis=axis)
        return tf.reduce_min(a, axis=axis)
        # return tf.maximum(tf.reduce_sum(a-1, axis=axis)+1,0)

    @staticmethod
    def exists(a, axis):
        return tf.reduce_max(a, axis=axis)

    @staticmethod
    def negation(a):
        return 1 - a

    @staticmethod
    def implication(a, b):
        return tf.minimum(1., 1 - a + b)

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = range(max+1)
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)

class LogicFactory:

    @staticmethod
    def create(logic):
        if logic=="lukasiewicz":
            return Lukasiewicz
