##A=[-2,1]x[-2,2], B=[-1,2]x[-2,2], C=[-1,1]x[-2,2]


import lyrics as lyr
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)
import random

#current_world.logic = LogicFactory.create("Goedel")

epoch=1000
epochs=10000
soglia=0.5
n=20
nn=10   # number of neurons for the hidden layer
nSup=n/4

#Defining Supervisions

r1 = np.linspace(-3, 3, num=n, endpoint=True, retstep=False, dtype=np.float32)
data=np.array([[a, b] for a in r1 for b in r1])

#A
dataApos=np.array([[a, b] for a,b in data if a <= 2 and a>=-2 and b <= 2 and b>=-2])
dataAneg = np.array([[a,b] for a,b in data if not (a <= 2 and a>=-2 and b <= 2 and b>=-2)])
dataA = np.concatenate((dataApos, dataAneg), axis=0)
dataA = tf.convert_to_tensor(dataA)
labelsApos = np.ones_like(dataApos[:,0:1])
labelsAneg = np.zeros_like(dataAneg[:,0:1])
labelsA = np.concatenate((labelsApos, labelsAneg), axis=0)
labelsA = tf.convert_to_tensor(labelsA)


#B
dataBpos= np.array([[a, b] for a,b in data if a <= 1 and a>=-1 and b <= 1 and b>=-1])
dataBneg = np.array([[a,b] for a,b in data if not (a <= 1 and a>=-1 and b <= 1 and b>=-1)])
dataB = np.concatenate((dataBpos, dataBneg), axis=0)
dataB = tf.convert_to_tensor(dataB)
labelsBpos = np.ones_like(dataBpos[:,0:1])
labelsBneg = np.zeros_like(dataBneg[:,0:1])
labelsB = np.concatenate((labelsBpos, labelsBneg), axis=0)
labelsB = tf.convert_to_tensor(labelsB)





is_A = lyr.functions.FFNClassifier("isA",2,1,(nn,))
is_B = lyr.functions.FFNClassifier("isB",2,1,(nn,))


#CONSTRAINTS(Only Supervisions)
lyr.current_world.lambda_p = 1.5
lyr.PointwiseConstraint(is_A, labelsA,  dataA)
lyr.PointwiseConstraint(is_B, labelsB,  dataB)


#LEARNING
loss = lyr.current_world.loss()
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    sess.run(train_op)
    if i%100==0:
        print(sess.run(loss))



lyr.Domain(label="Points", data=tf.concat((dataA, dataB), axis=0))


lyr.Relation(label="A", domains=("Points"), function=is_A)
lyr.Relation(label="B", domains=("Points"), function=is_B)


n=4
base =[[0,0], [1,0], [0,1], [1,1]]
formulas = []
for i in range(2**n):
    s = ("{0:0%db}" % n).format(i)
    s = [int(c) for c in s]
    atoms =[]
    for k in range(4):
        if s[k] == 1:
            if base[k][0] == 1:
                a = "A(x)"
            else:
                a = "not A(x)"
            if base[k][1] == 1:
                b = "B(x)"
            else:
                b = "not B(x)"
            subformula = "(%s and %s)" %(a,b)
            atoms.append(subformula)
    if len(atoms)>0:
        formula = " or ".join(atoms)
        formula = "forall x: " + formula
        formulas.append(formula)


for formula in formulas:
    print(formula)
    print(sess.run(lyr.Constraint(formula).tensor))



