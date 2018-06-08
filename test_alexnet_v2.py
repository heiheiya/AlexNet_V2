import tensorflow as tf
from alexnet_v2 import AlexNet
import matplotlib.pyplot as plt

image_size = 227
channels = 3
keep_prob = 0.5
num_classes = 10
batch_size = 10
class_name = ['cat', 'dog', 'airplane', 'car', 'bird', 'frog', 'deer', 'horse', 'ship', 'truck']

def test_image(filename, num_class, weights_path='Default'):
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string, channels=channels)
    img_resized = tf.image.resize_images(img_decoded, [image_size, image_size])
    img_reshape = tf.reshape(img_resized, shape=[1, image_size, image_size, channels])

    model = AlexNet(img_reshape, keep_prob, num_classes, batch_size, image_size, channels, skip_layer='')
    score = tf.nn.softmax(model.fc8)
    max = tf.argmax(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoint/model.ckpt")
        prob = sess.run(max)[0]
        print(prob)
        print(class_name[prob])
        #plt.imshow(img_decoded.eval())
        #plt.title("Class: " + class_name[prob])
        #plt.show()

test_image("./data/cifar-10_test/90004.jpg", num_class=num_classes)