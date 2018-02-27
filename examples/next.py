from builtins import input
from CLARE.CLARE_future import *

import matplotlib
matplotlib.use('Agg')
from mnist_functions import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt

images = mnist.train.images
labels = mnist.train.labels

with tf.Session() as sess:

    images_data = tf.convert_to_tensor(images, name="Images")
    labels_data = tf.convert_to_tensor(labels, name="Labels")

    Images = Domain("Images", data=images_data)
    classifier = MNISTClassifier(label="MNISTClassifier", hidden_size=1000, sigma=15.)
    PointwiseConstraint(classifier, labels_data, images_data)

    learn(learning_rate=0.001, sess=sess, num_epochs=50000, print_iters=100, eval=classifier.evaluate(labels, images_data))

    for i, name in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]):
        Relation(label=name, domains=Images, function=Slice(classifier, i))


    generative = ImageGenerator(label="RegressionNN", input_size=784, output_size=784)
    next = Function(label="next", domains=Images, function=generative)
    Constraint("forall x: zero(x) -> one(next(x))")
    Constraint("forall x: one(x) -> two(next(x))")
    Constraint("forall x: two(x) -> three(next(x))")
    Constraint("forall x: three(x) -> four(next(x))")
    Constraint("forall x: four(x) -> five(next(x))")
    Constraint("forall x: five(x) -> six(next(x))")
    Constraint("forall x: six(x) -> seven(next(x))")
    Constraint("forall x: seven(x) -> eight(next(x))")
    Constraint("forall x: eight(x) -> nine(next(x))")
    Constraint("forall x: nine(x) -> zero(next(x))")

    RegularizationConstraint(generative, 0.001)

    train_writer = tf.summary.FileWriter("./", sess.graph)
    learn(learning_rate=0.001, sess=sess, num_epochs=10000, print_iters=100, vars=generative.weights.values())

    y = generative.call(input=images)
    Y = sess.run(y)
    sess.close()

del sess

while True:
    a = int(input("image: "))
    img = images[a:a+1, :]
    gen = Y[a:a+1, :]


    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.imshow(np.reshape(img,[28,28]))

    ax3 = fig.add_subplot(212)
    ax3.imshow(np.reshape(gen,[28,28]))


    # plt.show()
    fig.savefig("n.png")
    plt.close(fig)