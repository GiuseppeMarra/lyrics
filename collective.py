##A=[-2,1]x[-2,2], B=[-1,2]x[-2,2], C=[-1,1]x[-2,2]


import lyrics as lyr
import tensorflow as tf
import numpy as np
np.random.seed(2)
tf.set_random_seed(2)
# np.random.seed(1)
# tf.set_random_seed(1)
import random
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

#current_world.logic = LogicFactory.create("Goedel")

epoch=2000
epochs=10000
soglia=0.5
n=16
nn=5   # number of neurons for the hidden layer
nSup=n/4

#Defining Supervisions

#A
Dom = np.linspace(-2, 2, num=n, endpoint=True, retstep=False, dtype=np.float32)
Test=np.array([[a, b] for a in Dom for b in Dom])
Test2=np.array([[[a, b]] for a in Dom for b in Dom])

trAp1=np.linspace(-2, 0.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
trAp2=np.linspace(-2, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupPosA=np.array([[a, b] for a in trAp1 for b in trAp2])
np.random.shuffle(SupPosA)
SupPosA=SupPosA[0:nSup]

trAn1=np.linspace(1, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
trAn2=np.linspace(-2, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupNegA=np.array([[a, b] for a in trAn1 for b in trAn2])
np.random.shuffle(SupNegA)
SupNegA=SupNegA[0:nSup]

SupA=np.concatenate((SupPosA,SupNegA), axis=0)



#B
trBp1=np.linspace(-1, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
trBp2=np.linspace(-2, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupPosB=np.array([[a, b] for a in trBp1 for b in trBp2])
np.random.shuffle(SupPosB)
SupPosB=SupPosB[0:nSup]

trBn1=np.linspace(-2, -1.1, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
trBn2=np.linspace(-2, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupNegB=np.array([[a, b] for a in trBn1 for b in trBn2])
np.random.shuffle(SupNegB)
SupNegB=SupNegB[0:nSup]

SupB=np.concatenate((SupPosB,SupNegB), axis=0)



#C
trCp1=np.linspace(-1, 0.9, num=n/4, endpoint=False, retstep=False, dtype=np.float32)+0.1
trCp2=np.linspace(-2, 1.9, num=n/4, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupPosC=np.array([[a, b] for a in trCp1 for b in trCp2])
np.random.shuffle(SupPosC)
SupPosC=SupPosC[0:nSup]

trCn11=np.linspace(-2, -0.8, num=n/8, endpoint=False, retstep=False, dtype=np.float32)+0.1
trCn12=np.linspace(1, 1.9, num=n/8, endpoint=False, retstep=False, dtype=np.float32)+0.1
trCn1=np.concatenate((trCn11,trCn12), axis=0)
trCn2=np.linspace(-2, 1.9, num=n/2, endpoint=False, retstep=False, dtype=np.float32)+0.1
SupNegC=np.array([[a, b] for a in trCn1 for b in trCn2])
np.random.shuffle(SupNegC)
SupNegC=SupNegC[0:nSup]

SupC=np.concatenate((SupPosC,SupNegC), axis=0)

labelA=np.concatenate((np.ones(shape=[SupPosA.shape[0], 1]),np.zeros(shape=[SupNegA.shape[0], 1])),axis=0)
labelB=np.concatenate((np.ones(shape=[SupPosB.shape[0], 1]),np.zeros(shape=[SupNegB.shape[0], 1])),axis=0)
labelC=np.concatenate((np.ones(shape=[SupPosC.shape[0], 1]),np.zeros(shape=[SupNegC.shape[0], 1])),axis=0)

# plt.scatter(SupC[labelC[:,0]>0.5,0], SupC[labelC[:,0]>0.5,1], color="green", marker="s", alpha=0.5, s=200, label="C")
# plt.scatter(SupC[labelC[:,0]<0.5,0], SupC[labelC[:,0]<0.5,1], color="red", marker="s", alpha=0.5, s=200, label="not C")
# plt.ylim((-3, 3))
# plt.xlim((-3, 3))
# plt.legend()
# plt.show()


A = Test[:,0]<=1
B = Test[:,0]>=-1
C = np.logical_and(Test[:,0]<=1, Test[:,0]>=-1)
plt.scatter(Test[A,0], Test[A,1], color="red", marker="s", alpha=0.5, s=200, label="A")
plt.scatter(Test[B,0], Test[B,1], color="blue", marker="s", alpha=0.5, s=200, label="B")
plt.scatter(Test[C,0], Test[C,1], color="black", marker="s", alpha=0.5, s=200, label="C")
plt.ylim((-3, 3))
plt.xlim((-3, 3))
plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()




is_A = lyr.functions.FFNClassifier("isA",2,1,(nn,))
is_B = lyr.functions.FFNClassifier("isB",2,1,(nn,))
is_C = lyr.functions.FFNClassifier("isC",2,1,(nn,))



#INcLARE





#CONSTRAINTS(Only Supervisions)
lyr.current_world.lambda_p = 1.5
lyr.PointwiseConstraint(is_A, labelA,  SupA)
lyr.PointwiseConstraint(is_B, labelB,  SupB)
lyr.PointwiseConstraint(is_C, labelC,  SupC)


#LEARNING PRIORS
loss = lyr.current_world.loss()
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    sess.run(train_op)
    # if i%100==0:
        # print(sess.run(loss))

# Evaluation1

bound = np.array([[[-2, 1], [-2, 2]], [[-1, 2], [-2, 2]], [[-1, 1], [-2, 2]]])

is_ABC = sess.run((is_A(Test), is_B(Test), is_C(Test)))



plt.scatter(Test2[is_ABC[0]>0.5,0], Test2[is_ABC[0]>0.5,1], color="red", marker="s", alpha=0.5, s=150, label="A")
plt.scatter(Test2[is_ABC[1]>0.5,0], Test2[is_ABC[1]>0.5,1], color="blue", marker="s", alpha=0.5, s=150, label="B")
plt.scatter(Test2[is_ABC[2]>0.5,0], Test2[is_ABC[2]>0.5,1], color="black", marker="s", alpha=0.5, s=150, label="C")
plt.ylim((-3, 3))
plt.xlim((-3, 3))
plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()


quanti = np.zeros(3)
preP = np.zeros(3)
preN = np.zeros(3)
trueP = np.zeros(3)
falseP = np.zeros(3)
falseN = np.zeros(3)
F1 = np.zeros(3)

for i in range(len(Test)):
    for j in range(3):
        if is_ABC[j][i] >= soglia:
            point = Test2[i]
            preP[j] += 1
            if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                    bound[j, 1, 1]:
                trueP[j] += 1
            else:
                falseP[j] += 1
        else:
            point = Test2[i]
            preN[j] += 1
            if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                    bound[j, 1, 1]:
                falseN[j] += 1
        if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                bound[j, 1, 1]:
            quanti[j] += 1

for j in range(3):
    F1[j] = 2 * trueP[j] / (2 * trueP[j] + falseN[j] + falseP[j])

print(quanti, preP, trueP, F1)



#Collective Classification with rules
priorA=sess.run(is_A(Test))
priorB=sess.run(is_B(Test))
priorC=sess.run(is_C(Test))



class IndexingFunction(lyr.functions.Learner):

    def __init__(self, k, initial_value = None):
        super(IndexingFunction, self).__init__()
        if initial_value is None:
            initial_value = tf.random_normal([k])
        self.k = k
        self.var = tf.Variable(initial_value=initial_value)

    def __call__(self, a):
        return tf.sigmoid(tf.gather(self.var, a))

    def cost(self, labels, input=None):
        return tf.reduce_mean(tf.square(self.__call__(input) - labels))


def invSigm(x):
    eps = 1e-12
    return np.log((x + eps) / (1 - x + eps))




lyr.current_world.pointwise_loss = tf.constant(0.)
lyr.current_world.constraint_loss = tf.constant(0.)

is_A = IndexingFunction(len(Test), initial_value=invSigm(priorA))
is_B = IndexingFunction(len(Test), initial_value=invSigm(priorB))
is_C = IndexingFunction(len(Test), initial_value=invSigm(priorC))


Test = np.reshape(range(len(Test)), [-1, 1])

lyr.Domain(label="Points", data=tf.constant(Test))

lyr.Relation(label="A", domains=("Points"), function=is_A)
lyr.Relation(label="B", domains=("Points"), function=is_B)
lyr.Relation(label="C", domains=("Points"), function=is_C)



#Constraints
lyr.Constraint("forall x: A(x) or B(x)")
lyr.Constraint("forall x: (A(x) and B(x)) <-> C(x)")
# Constraint("forall x: (A(x) and not B(x)) -> not C(x)")

lyr.PointwiseConstraint(is_A, priorA,  Test)
lyr.PointwiseConstraint(is_B, priorB,  Test)
lyr.PointwiseConstraint(is_C, priorC,  Test)



#Learning with Constraints
loss2 = lyr.current_world.loss()
train_op2 = tf.train.GradientDescentOptimizer(0.1).minimize(loss2)
sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())

for i in range(epochs):
    sess2.run(train_op2)
    # if i%100==0:
    #     print(sess2.run(loss2))



#Evaluation2
is_ABC = sess2.run((is_A(Test), is_B(Test), is_C(Test)))


quanti = np.zeros(3)
preP = np.zeros(3)
preN = np.zeros(3)
trueP = np.zeros(3)
falseP = np.zeros(3)
falseN = np.zeros(3)
F1 = np.zeros(3)

for i in range(len(Test)):
    for j in range(3):
        if is_ABC[j][i] >= soglia:
            point = Test2[i]
            preP[j] += 1
            if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                    bound[j, 1, 1]:
                trueP[j] += 1
            else:
                falseP[j] += 1
        else:
            point = Test2[i]
            preN[j] += 1
            if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                    bound[j, 1, 1]:
                falseN[j] += 1
        if point[0, 0] >= bound[j, 0, 0] and point[0, 0] <= bound[j, 0, 1] and point[0, 1] >= bound[j, 1, 0] and point[0, 1] <= \
                bound[j, 1, 1]:
            quanti[j] += 1

for j in range(3):
    F1[j] = 2 * trueP[j] / (2 * trueP[j] + falseN[j] + falseP[j])

print(quanti, preP, trueP, F1)

plt.scatter(Test2[is_ABC[0][:,0]>0.5,0], Test2[is_ABC[0][:,0]>0.5,1], color="red", marker="s", alpha=0.5, s=150, label="A")
plt.scatter(Test2[is_ABC[1][:,0]>0.5,0], Test2[is_ABC[1][:,0]>0.5,1], color="blue", marker="s", alpha=0.5, s=150, label="B")
plt.scatter(Test2[is_ABC[2][:,0]>0.5,0], Test2[is_ABC[2][:,0]>0.5,1], color="black", marker="s", alpha=0.5, s=150, label="C")
plt.ylim((-3, 3))
plt.xlim((-3, 3))
plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()