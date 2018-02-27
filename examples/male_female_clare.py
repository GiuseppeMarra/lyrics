from CLARE.CLARE_future import *
import numpy as np
import math
import scipy.misc as misc
from male_female_2.export_dataset import generator as gen

mode = "eval"
log_root= "log5"
n_filters = (3, 256, 256, 128, 128, 64, 64)
size_filters = (7, 7, 5, 5, 3, 3)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))



class encoder(object):

    def __init__(self,
                n_filters=(3, 64, 64),
                filter_sizes=(5, 5)):
        self.dimensions = n_filters
        self.filter_sizes = filter_sizes

        self.Ws = []
        self.bs = []
        for i in range(len(self.dimensions)-1):
            n_input = self.dimensions[i]
            n_output = self.dimensions[i+1]
            W = tf.Variable(initial_value=tf.random_uniform([filter_sizes[i],filter_sizes[i],n_input,n_output],
                            minval=-1.0 / math.sqrt(n_input),
                            maxval=1.0 / math.sqrt(n_input)),
                            name="W"+str(i))
            b = tf.Variable(tf.zeros([n_output]), name="b"+str(i))
            self.Ws.append(W)
            self.bs.append(b)



    def call(self, x):


            images = tf.reshape(x, [-1, 64,64,3])
            for i in range(len(self.dimensions)-1):
                with tf.variable_scope('conv'+str(i)) as scope:
                    conv = tf.nn.conv2d(images, self.Ws[i], [1, 1, 1, 1], padding='SAME')
                    pre_activation = tf.nn.bias_add(conv, self.bs[i])
                    conv = tf.nn.leaky_relu(pre_activation, name=scope.name)

                pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool'+str(i))
                norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                  name='norm'+str(i))

                images = norm


            self.z = tf.reshape(images, [-1, (64/(2**6)) * (64/(2**6)) * 64])

            return self.z



class decoder(object):
    def __init__(self,
                n_filters=(3, 64, 64),
                filter_sizes=(5, 5)):
        self.dimensions = list(n_filters)
        self.filter_sizes = filter_sizes

        self.Ws = []
        self.bs = []
        for i in range(len(self.dimensions)-1):
            n_input = self.dimensions[i]
            n_output = self.dimensions[i+1]
            W = tf.Variable(initial_value=tf.random_uniform([filter_sizes[i],filter_sizes[i],n_input,n_output],
                            minval=-1.0 / math.sqrt(n_input),
                            maxval=1.0 / math.sqrt(n_input)),
                            name="W"+str(i))
            b = tf.Variable(tf.zeros([n_input]), name="b"+str(i))
            self.Ws.append(W)
            self.bs.append(b)

        self.Ws.reverse()
        self.bs.reverse()



    def call(self, x):

        images = tf.reshape(x, [-1, (64/(2**6)), (64/(2**6)), 64])
        for i in range(len(self.dimensions) - 1):
            batch = tf.shape(images)[0]
            side = images.get_shape()[1]*2
            channels = self.bs[i].get_shape()[0]
            output_shape = [batch, side, side, channels]
            with tf.variable_scope('conv' + str(i)) as scope:
                conv = tf.nn.conv2d_transpose(images, self.Ws[i], output_shape, [1, 2, 2, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, self.bs[i])
                if i< len(self.dimensions) - 2:
                    conv = tf.nn.leaky_relu(pre_activation, name=scope.name)
                else:
                    conv = pre_activation
            images = conv


        self.z = tf.reshape(images, [-1, 64 * 64 * 3])

        return self.z


def cosine(a, b):
    eps = 1e-12
    batch = tf.shape(a)[0]
    a = tf.reshape(a, [batch, -1])
    b = tf.reshape(b, [batch, -1])

    xnorm = tf.norm(a, axis=1, keep_dims=True, name="xnorm")
    mnorm = tf.norm(b, axis=1, keep_dims=True, name="mnorm")
    similarity = tf.matmul(a / (xnorm + eps), tf.transpose(tf.div(b, (mnorm + eps))))
    dist = (1 - similarity) / 2

    return dist

def mse(a,b):
    batch = tf.shape(a)[0]
    a = tf.reshape(a, [batch, -1])
    b = tf.reshape(b, [batch, -1])
    return - tf.reduce_mean(tf.square(a -b), axis=1)


def loss_f(a,b):
    return mse(a,b)

class isEqual(object):

    def call(self, a,b):
        return mse(a,b)


class isMale(object):
    def __init__(self, labels):
        self.labels = labels

    def call(self, a):
        return self.labels

class isFemale(object):
    def __init__(self, labels):
        self.labels = labels

    def call(self, a):
        return 1 - self.labels


def train():


    current_world.logic = LogicFactory.create("implies_product")

    images = tf.placeholder(shape=[None, 64,64,3], dtype=tf.float32)
    images_rs = tf.reshape(images, [-1, 64*64*3])
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    I = Domain("Images", data=images_rs)
    Z = Domain("Latent", data=tf.zeros([1,(64/(2**6)) * (64/(2**6)) * 64]))

    scene = Function("scene", domains=I, function=encoder(n_filters, size_filters))
    male = Function("male", domains=Z, function=decoder(n_filters, size_filters))
    female = Function("female", domains=Z, function=decoder(n_filters, size_filters))
    equal = Relation("isEqual", domains=(I, I), function=isEqual())

    # Given
    ismale = Relation("isMale", domains=I, function=isMale(labels))
    isnfemale = Relation("isFemale", domains=I, function=isFemale(labels))

    ae1 = Constraint("forall x: isMale(x) -> isEqual(male(scene(x)), x)")
    ae2 = Constraint("forall x: isFemale(x) -> isEqual(female(scene(x)), x)")

    cons1 = Constraint("forall x: isMale(x) -> isEqual(male(scene(female(scene(x)))), x)")
    cons2 = Constraint("forall x: isFemale(x) -> isEqual(female(scene(male(scene(x)))), x)")

    global_step = tf.train.get_or_create_global_step()


    train_op = tf.train.AdamOptimizer(0.001).minimize(current_world.loss, global_step=global_step)

    batch_size = 16

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': global_step,
                 'loss': current_world.loss},
        every_n_iter=10)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=log_root,
            hooks=[logging_hook],
            save_checkpoint_secs=180,
            save_summaries_steps=0) as sess:

        annotation_path = "male_female_2/list_attr_celeba.txt"
        base = "male_female_2/img_align_celeba/"
        for e in range(100):
            generator = gen(annotation_path,base,batch_size)
            for img,lab in generator:
                sess.run(train_op, {labels: lab, images: img})


def evaluate():


    images = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)


    e = encoder(n_filters, size_filters)
    d1 = decoder(n_filters, size_filters)
    d2 = decoder(n_filters, size_filters)
    z = e.call(images)
    male = tf.reshape(d1.call(z), [-1,64,64,3])
    female = tf.reshape(d2.call(z), [-1,64,64,3])

    global_step = tf.train.get_or_create_global_step()


    saver = tf.train.Saver()

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    try:
        ckpt_state = tf.train.get_checkpoint_state(log_root)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
        return
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', log_root)
        return
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    annotation_path = "male_female_2/list_attr_celeba.txt"
    base = "male_female_2/img_align_celeba/"
    generator = gen(annotation_path, base, 20)
    img,lab = generator.next()
    imgmale, imgfemale = sess.run((male,female), feed_dict={images: img})
    print(lab)
    for i in range(20):

        misc.imsave("imgs3/img"+str(i)+"-original.jpg", np.squeeze(img[i]))
        misc.imsave("imgs3/img"+str(i)+"-male.jpg", np.squeeze(imgmale[i]))
        misc.imsave("imgs3/img"+str(i)+"-female.jpg", np.squeeze(imgfemale[i]))


def main(_):

    if mode == 'train':
        train()
    else:
        evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


