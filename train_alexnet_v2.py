import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet_v2 import AlexNet
from datagenerator import ImageDataGenerator

train_file = './data/train.txt'
val_file = './data/val.txt'

learning_rate = 0.01
num_epoches = 5000
batch_size = 10

image_size = 227
num_channels = 3

dropout_rate = 0.5
num_classes = 10
train_layers = ['fc8']

display_step = 5

filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoint"

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, num_channels])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, batch_size, image_size, num_channels, train_layers)
score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

with tf.name_scope("cross_entropy"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

with tf.name_scope("train"):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

train_generator = ImageDataGenerator(train_file, horizontal_filp=True, shuffle=True, num_classes=10)
val_generator = ImageDataGenerator(val_file, shuffle=False, num_classes=10)

train_batches_per_epoch = np.floor(train_generator.data_size/batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size/batch_size).astype(np.int16)

format_str = ('epoch %d: step %d, loss=%.3f')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

    for epoch in range(num_epoches):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        for step in range(train_batches_per_epoch):
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})
            if step%display_step == 0:
                losses, s = sess.run([loss, merged_summary], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch+step+1)
                print(format_str % (epoch + 1, step, losses))

        print("{} Start validation".format(datetime.now()))

        val_acc = 0
        val_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
            val_acc += acc
            val_count += 1
        val_acc /= val_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), val_acc))

        val_generator.reset_pointer()
        train_generator.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model.ckpt')
        save_path = saver.save(sess, checkpoint_name)