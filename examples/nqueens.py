from CLARE.CLARE import *
import numpy as np

n = 7

data = np.reshape(range(n), [-1, 1])

class isAQueen(object):
    def __init__(self, n):
        with tf.variable_scope("isAQueen"):
            self.vars = tf.sigmoid(tf.Variable(initial_value=tf.random_uniform(shape=[n*n], minval=-10, maxval=10)))
            self.vars = tf.concat((self.vars, [0]),0)

    def call(self, a, b):
        a = tf.where(tf.logical_and(a >=0, a < n ), a, -float('inf') * tf.ones_like(a))
        b = tf.where(tf.logical_and(b >= 0, b< n),  b, -float('inf') * tf.ones_like(b))
        i = tf.cast(a*n+b, tf.int32)
        iplus = tf.where(i<0, (n*n)*tf.ones_like(i), i)
        return tf.gather(indices=iplus, params=self.vars)

class isEqual(object):
    def call(self,a,b):
        return tf.where(tf.equal(a,b), tf.ones_like(a), tf.zeros_like(a))
        # return tf.nn.relu(tf.square(a-b))

class lessThan(object):
    def call(self,a,b):
        return tf.where(tf.less_equal(a,b), tf.ones_like(a), tf.zeros_like(a))

class greaterThanZero(object):
    def call(self, a):
        return tf.where(tf.greater_equal(a,0), tf.ones_like(a), tf.zeros_like(a))

class lessThanN(object):
    def call(self, a):
        return tf.where(tf.less(a,n), tf.ones_like(a), tf.zeros_like(a))

class add(object):
    def call(self,a, b):
        return tf.add(a,b)

class minus(object):
    def call(self,a, b):
        return tf.subtract(a,b)


D = Domain(label="D", data=data)
K = Domain(label="K", data=data[1:])
Q= Relation(label="Queen", domains=("D", "D"), function=isAQueen(n))
Relation(label="greaterThanZero", domains="D", function=greaterThanZero())
Relation(label="lessThanN", domains="D", function=lessThanN())
Function(label="add", domains=("D", "K"), function=add())
Function(label="minus", domains=("D", "K"), function=minus())
Relation(label="isEqual", domains=("D", "D"), function=isEqual())
# #
# #
C = Constraint("forall x: (forall y: (forall z: Queen(x,y) -> (not Queen(x,z) or isEqual(y,z))))",3)
CC = Constraint("forall x: exists y: Queen(x,y)",3)

C1 = Constraint("forall y: (forall x: (forall z: Queen(x,y) -> (not Queen(z,y) or isEqual(x,z))))",3)
CC1 = Constraint("forall y: exists x: Queen(x,y)",3)


C2 = Constraint("forall x: forall y: forall k: (lessThanN(add(x,k)) and lessThanN(add(y,k)) and Queen(x,y)) -> (not Queen(add(x,k), add(y,k)))",2)
C3 = Constraint("forall x: forall y: forall k: (lessThanN(add(x,k)) and greaterThanZero(minus(y,k)) and Queen(x,y)) -> (not Queen(add(x,k), minus(y,k)))",2)

sess = tf.Session()
learn(learning_rate=0.01, sess=sess, num_epochs=2000)

Z = Constraint("Queen(x,y)")
# print(sess.run(C.tensor))
# print(sess.run(CC.tensor))
# print(sess.run(C1.tensor))
# print(sess.run(CC1.tensor))
# print(sess.run(C2.tensor))
# print(sess.run(C3.tensor))
# # print(sess.run(C4.tensor))
# # print(sess.run(C5.tensor))
# print(sess.run(Constraint("forall k: (lessThanN(add(x,k)) and lessThanN(add(y,k)) and Queen(x,y)) -> (not Queen(add(x,k), add(y,k)))").tensor))
# print(sess.run(Constraint("exists k: Queen(add(x,k), add(y,k))").tensor))
# print(sess.run(Constraint("forall k: (greaterThanZero(minus(x,k)) and lessThanN(add(y,k)) and Queen(x,y)) -> not Queen(minus(x,k), add(y,k))").tensor))
res = sess.run(tf.where(Z.tensor>=0.5, tf.ones_like(Z.tensor), tf.zeros_like(Z.tensor)))
#
print(res)
# print("col"+str(np.sum(res,axis=0)))
# print("row"+str(np.sum(res,axis=1)))
