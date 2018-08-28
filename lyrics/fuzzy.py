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
    def exclusive_disj(args):
        raise NotImplementedError('users must define "exclusive_disj" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def forall(a, axis):
        raise NotImplementedError('users must define "forall" to use this base class')

    @classmethod
    def forall_with_loss(cls, a, axis):
        return cls.forall(a,axis)

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

    @staticmethod
    @abc.abstractmethod
    def iff(a, b):
        raise NotImplementedError('users must define "iff" to use this base class')

    @staticmethod
    @abc.abstractmethod
    def loss(a):
        raise NotImplementedError('users must define "loss" to use this base class')


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
    def exclusive_disj(args):
        # New axis stacking truth values of arguments
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)

        # Repetition along a new axis
        multiplies = tf.concat([tf.ones([new_axis + 1], dtype=tf.int32), [len(args)]], axis=0)
        arg = tf.tile(tf.expand_dims(arg, -1), multiplies)

        # Diagonal matrix along the last two dimensions
        diag = tf.diag(tf.ones([len(args)]))
        diag = tf.reshape(diag, tf.concat([tf.ones([new_axis], dtype=tf.int32), [len(args)], [len(args)]], axis=0))
        multiplies = tf.concat((tf.shape(args[0]), [1, 1]), axis=0)
        diag = tf.tile(diag, multiplies)
        diag_inverse = 1 - diag

        # MIN(MAX(1-a,b,c),MAX(a,1-b,c),MAX(a,b,1-c))
        arg = diag * (diag - arg) + diag_inverse * arg

        return tf.reduce_min(tf.reduce_max(arg, axis=-2), axis=-1)

    @staticmethod
    def forall(a, axis=0):
        # return tf.reduce_mean(a, axis=axis)
        # return tf.reduce_sum(a, axis=axis)
        return tf.reduce_min(a, axis=axis)
        # return tf.reduce_sum(a-1, axis=axis)+1


    @staticmethod
    def exists(a, axis):

        return tf.reduce_max(a, axis=axis)
        # return tf.log(tf.reduce_sum(tf.exp(a), axis=axis))
        # return tf.minimum(1., tf.reduce_sum(a, axis=axis))

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.minimum(1., 1 - a + b)

    @staticmethod
    def iff(a, b):
        return 1 - tf.abs(a-b)

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = list(range(max+1))
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)

    @staticmethod
    def loss(phi):
        return 1 - phi

class LukasiewiczStrong(FuzzyLogic):

    @staticmethod
    def weak_conj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.reduce_min(arg, axis=new_axis)
        # return tf.maximum(tf.reduce_sum(arg-1, axis=new_axis)+1,0)

    @staticmethod
    def strong_disj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.minimum(1., tf.reduce_sum(arg, axis=new_axis))

    @staticmethod
    def exclusive_disj(args):
        # New axis stacking truth values of arguments
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)

        # Repetition along a new axis
        multiplies = tf.concat([tf.ones([new_axis + 1], dtype=tf.int32), [len(args)]], axis=0)
        arg = tf.tile(tf.expand_dims(arg, -1), multiplies)

        # Diagonal matrix along the last two dimensions
        diag = tf.diag(tf.ones([len(args)]))
        diag = tf.reshape(diag, tf.concat([tf.ones([new_axis], dtype=tf.int32), [len(args)], [len(args)]], axis=0))
        multiplies = tf.concat((tf.shape(args[0]), [1, 1]), axis=0)
        diag = tf.tile(diag, multiplies)
        diag_inverse = 1 - diag

        # MIN(MAX(1-a,b,c),MAX(a,1-b,c),MAX(a,b,1-c))
        arg = diag * (diag - arg) + diag_inverse * arg

        return tf.reduce_min(tf.reduce_max(arg, axis=-2), axis=-1)

    @staticmethod
    def forall(a, axis):
        # return tf.reduce_mean(a, axis=axis)
        return tf.reduce_sum(a, axis=axis)
        # return tf.reduce_min(a, axis=axis)
        # return tf.reduce_sum(a, axis=axis)


    @staticmethod
    def exists(a, axis):

        return tf.reduce_max(a, axis=axis)
        # return tf.log(tf.reduce_sum(tf.exp(a), axis=axis))
        # return tf.minimum(1., tf.reduce_sum(a, axis=axis))

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.minimum(1., 1 - a + b)


    @staticmethod
    def iff(a, b):
        return 1 - tf.abs(a-b)

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = list(range(max+1))
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)

    @staticmethod
    def loss(phi):
        return 1 - phi

class Goedel(FuzzyLogic):

    @staticmethod
    def weak_conj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.reduce_min(arg, axis=new_axis)

    @staticmethod
    def strong_disj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.minimum(tf.reduce_max(arg, axis=new_axis), 1 - tf.reduce_min(arg, axis=new_axis))

    @staticmethod
    def exclusive_disj(args):
        #New axis stacking truth values of arguments
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)

        #Repetition along a new axis
        multiplies = tf.concat([tf.ones([new_axis+1], dtype=tf.int32), [len(args)]], axis=0)
        arg = tf.tile(tf.expand_dims(arg, -1), multiplies)

        #Diagonal matrix along the last two dimensions
        diag = tf.diag(tf.ones([len(args)]))
        diag = tf.reshape(diag, tf.concat([tf.ones([new_axis], dtype=tf.int32), [len(args)], [len(args)]], axis=0))
        multiplies = tf.concat((tf.shape(args[0]), [1,1]), axis=0)
        diag = tf.tile(diag, multiplies)
        diag_inverse = 1 - diag

        #MIN(MAX(1-a,b,c),MAX(a,1-b,c),MAX(a,b,1-c))
        arg = diag*(diag - arg) + diag_inverse*arg

        return tf.reduce_min(tf.reduce_max(arg, axis=-2), axis=-1)

    @staticmethod
    def forall(a, axis):
        # return tf.reduce_mean(a, axis=axis)
        # return tf.reduce_sum(a, axis=axis)
        return tf.reduce_min(a, axis=axis)
        # return tf.maximum(tf.reduce_sum(a-1, axis=axis)+1,0)

    @staticmethod
    def exists(a, axis):

        return tf.reduce_max(a, axis=axis)
        # return tf.log(tf.reduce_sum(tf.exp(a), axis=axis))
        # return tf.minimum(1., tf.reduce_sum(a, axis=axis))

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.where(tf.greater_equal(b,a), tf.ones_like(b), b)

    @staticmethod
    def iff(a, b):
        pass

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = list(range(max+1))
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)





class Custom(FuzzyLogic):

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
        return tf.reduce_mean(a, axis=axis)
        # return tf.reduce_min(a, axis=axis)
        # return tf.maximum(tf.reduce_sum(a-1, axis=axis)+1,0)

    @staticmethod
    def exists(a, axis):

        return tf.reduce_max(a, axis=axis)
        # return tf.log(tf.reduce_sum(tf.exp(a), axis=axis))
        # return tf.minimum(1., tf.reduce_sum(a, axis=axis))

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.minimum(a,b)

    @staticmethod
    def iff(a, b):
        return 1 - tf.abs(a-b)

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = list(range(max+1))
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)

class Product(FuzzyLogic):

    @staticmethod
    def weak_conj(args):
        new_axis = len(args[0].get_shape())
        arg = tf.stack(args, axis=new_axis)
        return tf.reduce_prod(arg, axis=new_axis)

    @staticmethod
    def strong_disj(args):
        a = args[0]
        b = args[1]
        return a + b - a * b

    @staticmethod
    def forall(a, axis=0):
        return tf.reduce_mean(a, axis=axis)
        # return tf.reduce_min(a, axis=axis)
        # return tf.maximum(tf.reduce_sum(a-1, axis=axis)+1,0)


    @staticmethod
    def exists(a, axis):

        return tf.reduce_max(a, axis=axis)

    @staticmethod
    def negation(a):
        return 1. - a

    @staticmethod
    def implication(a, b):
        return tf.where(a>b, b/(a+1e-12), tf.ones_like(a))

    @staticmethod
    def iff(a, b):
        # return 1 - tf.abs(a-b)
        return 1 - tf.abs(a-b)

    @staticmethod
    def exists_n(a, axis, n):
        #top_k sorts only on the last dimension, so we need to transpose the input
        max = len(a.get_shape()) -1
        r = list(range(max+1))
        r[axis] = max
        r[max]=axis
        a = tf.transpose(a,r)
        top,_ = tf.nn.top_k(a, n)
        red = tf.reduce_min(top,axis=max, keep_dims=True)
        return tf.transpose(red, r)
    @staticmethod
    def loss(phi):
        # return -tf.log(phi)
        return -tf.log(phi+1e-12)



class LogicFactory:

    @staticmethod
    def create(logic):
        if logic=="lukasiewicz":
            return Lukasiewicz
        if logic=="lukasiewicz-strong":
            return LukasiewiczStrong
        elif logic=="custom":
            return Custom
        elif logic =="Goedel":
            return Goedel
        elif logic =="product":
            return Product
        else:
            raise Exception("Logic %s unknown" % logic)
