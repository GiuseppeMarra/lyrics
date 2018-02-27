import tensorflow as tf
import select
import sys

def isIterableNotString(o):
    if isinstance(o, str):
        return False
    try:
        for e in o:
            return True
    except TypeError:
        return False


def cartesian_product(a, b):
    len_a = tf.shape(a)
    len_b = tf.shape(b)
    new_a = tf.reshape(tf.tile(a, [1, len_b[0]]), [-1, len_a[1]])
    new_b = tf.tile(b, [len_a[0],1])
    return tf.concat((new_a, new_b), axis=1)

import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func


def cartesian(tensors):
    try:
        if (len(tensors) < 2): raise Exception()
    except:
        raise Exception("The length of domains must be >= 2")

    tensor = tensors[0]
    for i in range(1, len(tensors)):
        tensor = cartesian_product(tensor, tensors[i])
    return tensor


def heardEnter():

    ''' Listen for the user pressing ENTER '''

    i,o,e = select.select([sys.stdin],[],[],0.0001)

    for s in i:

        if s == sys.stdin:
            input = sys.stdin.readline()
            return True

    return False
