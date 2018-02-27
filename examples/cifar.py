import cifar_input
from resnet.resnet_model_uinque_learning import ResNet, HParams
import sys
import six
import time
from CLARE.CLARE_future import *
import tensorflow as tf
import hooks


train_dir ="tmp/4-cifar10-ST-withrules/train"
eval_dir = "tmp/4-cifar10-ST-withrules/eval"
log_root = "tmp/4-cifar10-ST-withrules"
mode = "train"
dataset='cifar10'
num_classes = 10
superv_batch_size = 128
test_batch_size = 64

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', mode, 'train or eval.')


def train():

    hps = HParams(batch_size=superv_batch_size+test_batch_size,
                  num_classes=num_classes,
                  num_classes_rules=5,
                  superv_batch=superv_batch_size,
                  unsuperv_batch=tf.constant(0.),
                  min_lrn_rate=0.0001,
                  lrn_rate=0.1,
                  num_residual_units=5,
                  use_bottleneck=False,
                  weight_decay_rate=0.0002,
                  relu_leakiness=0.1,
                  optimizer='mom')

    test_size = hps.unsuperv_batch
    images, labels = cifar_input.build_input('cifar10', "./cifar-10/data_bat*", hps.superv_batch, "test")
    images_test, _ = cifar_input.build_input('cifar10', "./cifar-10/test_bat*", test_batch_size, "test")

    images = tf.cond(test_size > 0.,
            true_fn= lambda: tf.concat((images, images_test), axis=0),
            false_fn= lambda: images)
    images = tf.reshape(images, [-1, 32 * 32 * 3])

    Images = Domain("Images", images)

    resnet = ResNet(hps=hps, mode="train")

    class_dict = {0: "airplane", 1: "automobile", 2: "bird", 3:"cat",
                  4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck",
                  10:"animal", 11:"transport", 12:"onroad", 13:"mammal", 14:"flies"}

    # Final classes
    airplane = Relation(label="airplane", domains=Images, function=Slice(resnet, 0))
    automobile = Relation(label="automobile", domains=Images, function=Slice(resnet, 1))
    bird = Relation(label="bird", domains=Images, function=Slice(resnet, 2))
    cat = Relation(label="cat", domains=Images, function=Slice(resnet, 3))
    deer = Relation(label="deer", domains=Images, function=Slice(resnet, 4))
    dog = Relation(label="dog", domains=Images, function=Slice(resnet, 5))
    frog = Relation(label="frog", domains=Images, function=Slice(resnet, 6))
    horse = Relation(label="horse", domains=Images, function=Slice(resnet, 7))
    ship = Relation(label="ship", domains=Images, function=Slice(resnet, 8))
    truck = Relation(label="truck", domains=Images, function=Slice(resnet, 9))

    # Middle Predicates

    animal = Relation(label="animal", domains=Images, function=Slice(resnet, 10))
    transport = Relation(label="transport", domains=Images, function=Slice(resnet, 11))

    onroad = Relation(label="onroad", domains=Images, function=Slice(resnet, 12))
    mammal = Relation(label="mammal", domains=Images, function=Slice(resnet, 13))
    flies = Relation(label="flies", domains=Images, function=Slice(resnet, 14))

    Constraint("forall x: airplane(x) -> transport(x)")
    Constraint("forall x: ship(x) -> transport(x)")
    Constraint("forall x: onroad(x) -> transport(x)")
    Constraint("forall x: automobile(x) -> onroad(x)")
    Constraint("forall x: truck(x) -> onroad(x)")
    Constraint("forall x: bird(x) -> animal(x)")
    Constraint("forall x: frog(x) -> animal(x)")
    Constraint("forall x: cat(x) -> mammal(x)")
    Constraint("forall x: dog(x) -> mammal(x)")
    Constraint("forall x: deer(x) -> mammal(x)")
    Constraint("forall x: horse(x) -> mammal(x)")
    Constraint("forall x: cat(x) -> not flies(x)")
    Constraint("forall x: dog(x) -> not flies(x)")
    Constraint("forall x: horse(x) -> not flies(x)")
    Constraint("forall x: deer(x) -> not flies(x)")
    Constraint("forall x: truck(x) -> not flies(x)")
    Constraint("forall x: ship(x) -> not flies(x)")#3
    Constraint("forall x: automobile(x) -> not flies(x)")
    Constraint("forall x: frog(x) -> not flies(x)") #2
    Constraint("forall x: bird(x) -> flies(x)")
    Constraint("forall x: airplane(x) -> flies(x)")

    #added by me
    Constraint("forall x: frog(x) -> (not mammal(x)) and (not onroad(x))")
    Constraint("forall x: bird(x) -> (not mammal(x)) and (not onroad(x))")
    Constraint("forall x: ship(x) -> (not mammal(x)) and (not onroad(x))")
    Constraint("forall x: airplane(x) -> (not mammal(x)) and (not onroad(x))")
    Constraint("forall x: mammal(x) -> animal(x)")




    #Chain animal->mammal->leaves
    Constraint("forall x: animal(x) -> bird(x) or frog(x) or mammal(x)")
    Constraint("forall x: mammal(x) -> cat(x) or deer(x) or dog(x) or horse(x)")


    #Chain [not] flies->leaves
    Constraint ("forall x: not flies(x) -> cat(x) or dog(x) or horse(x) or deer(x) or truck(x) or ship(x) or automobile(x) or frog(x)")
    Constraint("forall x: flies(x) -> bird(x) or airplane(x)")

    #Chain transport->onroad->leaves
    Constraint("forall x: transport(x) -> airplane(x) or ship(x) or onroad(x)")
    Constraint("forall x: onroad(x) -> automobile(x) or truck(x)")



    tf.summary.scalar("Costs/constraint_cost",  current_world.lambda_c * current_world.constraints_loss)


    PointwiseConstraint(function=resnet,
                        labels=labels,
                        inputs=images)

    tf.summary.scalar('Costs/supervision_cost', current_world.lambda_s * current_world.supervision_loss)

    cost = current_world.loss

    """Building TrainOp"""
    lrn_rate = tf.constant(hps.lrn_rate, tf.float32)
    tf.summary.scalar("LearningRate", lrn_rate)
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(cost, trainable_variables)

    optimizer = tf.train.MomentumOptimizer(lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=resnet.global_step, name='train_step')

    train_ops = [apply_op] + resnet._extra_train_ops
    train_op = tf.group(*train_ops)


    "Learning procedure"

    truth = tf.argmax(resnet.labels[0:hps.superv_batch,0:10], axis=1)
    predictions = tf.argmax(resnet.predictions[0:hps.superv_batch, 0:10], axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
    tf.summary.scalar("Precision", precision)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=train_dir,
        summary_op=tf.summary.merge_all())

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': resnet.global_step,
                 'loss': cost,
                 'precision': precision},
        every_n_iter=100)

    stop_hook = tf.train.StopAtStepHook(num_steps=220000)

    lr_hook = hooks.LearningRateSetterHook(lrn_rate, resnet)

    constraint_loss_hook = hooks.SetVariableAfterStepHook(var=current_world.lambda_c,
                                                          value_to_set=1.,
                                                          step=100000,
                                                          model=resnet)

    test_hook = hooks.SetVariableAfterStepHook(var=hps.unsuperv_batch,
                                                          value_to_set=test_batch_size,
                                                          step=160000,
                                                          model=resnet)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=log_root,
            save_checkpoint_secs=180,
            hooks=[logging_hook, lr_hook, constraint_loss_hook, test_hook],
            chief_only_hooks=[summary_hook, stop_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def evaluate():


    hps = HParams(batch_size=100,
                  num_classes=num_classes,
                  num_classes_rules=5,
                  superv_batch=100,
                  unsuperv_batch=0,
                  min_lrn_rate=0.0001,
                  lrn_rate=0.1,
                  num_residual_units=5,
                  use_bottleneck=False,
                  weight_decay_rate=0.0002,
                  relu_leakiness=0.1,
                  optimizer='mom')
    images, labels = cifar_input.build_input('cifar10', "./cifar-10/test_bat*", hps.batch_size, mode)
    images = tf.reshape(images, [-1, 32 * 32 * 3])

    Images = Domain("Images", images)

    resnet = ResNet(hps=hps, mode="test")
    cost = resnet.cost(labels, images)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(eval_dir)

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0},allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', log_root)
            break
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        tf.logging.info('Performing an entire step on the test set')
        for _ in six.moves.range(100):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [resnet.summaries, cost, resnet.predictions,
                 resnet.labels, resnet.global_step])

            truth = np.argmax(truth[:,0:10], axis=1)
            predictions = np.argmax(predictions[:,0:10], axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        time.sleep(60)



def main(_):

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'eval':
        evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
