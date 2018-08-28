import matplotlib as mpl
mpl.use('TkAgg')
import lyrics as lyr
import tensorflow as tf
from sklearn import datasets as data
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


data_size = 50



#Usupervised
X_np, y_np = data.make_moons(n_samples=data_size, shuffle=True, noise=0., random_state=None)


class IsClose(lyr.functions.AbstractFunction):

    def __call__(self, a, b):

        dist = tf.reduce_sum(tf.square(a - b), axis=1)
        res = 1. - tf.tanh(dist - 0.001)
        return res

class Is(lyr.functions.AbstractFunction):

    def __call__(self, a, b):

        return tf.cast(tf.equal(tf.reduce_sum(tf.square(a - b), axis=1), 0.), tf.float32)

is_ = Is()
is_close = IsClose()
is_A = lyr.functions.FFNClassifier(name="isA",
                                   input_size=2,
                                   n_classes=1,
                                   hidden_layers = (10,))
X = tf.cast(X_np, tf.float32)
y = tf.reshape(tf.cast(y_np, tf.float32), [-1 , 1])


lyr.current_world.lambda_p = tf.constant(1.)
lyr.current_world.lambda_c = tf.constant(1.)









Points = lyr.Domain(label="Points", data=X)

p0 = lyr.Individual(label="p0", domain="Points", value=tf.Variable([[-1,1.]]))
p1 = lyr.Individual(label="p1", domain="Points", value=tf.Variable([[2,0.4]]))

R1 = lyr.Relation("A", domains=("Points"), function=is_A)
R2 = lyr.Relation("isClose", domains=("Points", "Points"), function=is_close)
R3 = lyr.Relation("equal", domains=("Points", "Points"), function=is_)


lyr.Constraint("A(p0)")
lyr.Constraint("not A(p1)")
lyr.Constraint("exists q: not equal(q,p0) and A(q) and isClose(q,p0)")
lyr.Constraint("exists q: not equal(q,p1) and not A(q) and isClose(q,p1)")



lyr.PointwiseConstraint(is_A, y, X)
loss_pre_vincoli = lyr.current_world.lambda_p * lyr.current_world.pointwise_loss

activate_rules = tf.placeholder(dtype=tf.bool, shape=[])
lr = tf.placeholder(dtype=tf.float32, shape=[])
loss_post_vincoli = loss_pre_vincoli + lyr.current_world.lambda_c * lyr.current_world.constraint_loss
loss = tf.cond(activate_rules, lambda: loss_post_vincoli, lambda:loss_pre_vincoli)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

output = is_A(X)
predictions = tf.where(output > 0.5, tf.ones_like(output, dtype=tf.float32), tf.zeros_like(output, dtype=tf.float32))
accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, y), tf.float32)) / data_size






sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
sess.run(tf.global_variables_initializer())

pp0, pp1 = sess.run((p0.tensor, p1.tensor))
plt.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], c="blue", label="not A", marker="o")
plt.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], c="orange", label="A", marker="s")
plt.scatter(pp0[0,0], pp0[0,1], c="red", label="p0", marker="*", s=100)
plt.scatter(pp1[0,0], pp1[0,1], c="green", label="p1", marker="x", s=100)
plt.legend()
plt.show()




feed_dict = {activate_rules: False, lr:0.01}

switch=False
while True:
    _, acc, ll = sess.run((train_op,accuracy, loss), feed_dict)
    enter = lyr.utils.heardEnter()
    if not switch and enter:
        switch=True
        feed_dict = {activate_rules: True, lr: 0.01}
    elif switch and enter:
        break
    print(acc, ll)

pp0, pp1 = sess.run((p0.tensor, p1.tensor))
plt.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], c="blue", label="not A", marker="o")
plt.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], c="orange", label="A", marker="s")
plt.scatter(pp0[0,0], pp0[0,1], c="red", label="p0", marker="*",)
plt.scatter(pp1[0,0], pp1[0,1], c="green", label="p1", marker="x", s=100)
plt.legend()
plt.show()






