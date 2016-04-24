from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

#placeholders for batches of images and their labels (one_hot in y_)
x = tf.placeholder("float", shape = [None, 784])
y_ = tf.placeholder("float", shape = [None, 10])

#reshaping the x tensor to a 4-D one. 1.d:image, 2.d:height, 3.d:width, 4.d:color(b/w->1/0)
x_image = tf.reshape(x, [-1, 28, 28, 1])

#functions for initializing weights/biases with random values
def weight_var(shape):
    init = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init)

def bias_var(shape):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)

#functions for initializing nodes in the convolutional/pooling layer
def conv(x, W):
    #on the input matrix x the weight matrix W is applied according the convolutional method
    #in every step the window is shifted strides[1] to the right and in the end strides[2]
    #to the bottom. zero padding is used. output matrix has the batches in the 1st dimenson,
    #the height in the second, the width in the 3rd and the for every kernel the weighted
    #sum in the 4th
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool22(x):
    #k_size defines the window in which the max is searched. 
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#initialize a weightmatrix and a biasvector
W_conv1 = weight_var([5, 5, 1, 32])
b_conv1 = bias_var([32])

#apply the convolution step (~ compute the input for the convolution neurons) + apply ReLU 
#activation for every conv neuron to get their output for the pooling
out_conv1 = tf.nn.relu(conv(x_image, W_conv1) + b_conv1)

#apply the pooling step (out_conv1 as input)
out_pool1 = max_pool22(out_conv1)
debug_out_pool1_shape = tf.shape(out_pool1)
#add a second conv/pooling layer with now 64 kernels (! from the firs layers we get a in_channel
#of 32)
#initialize a weightmatrix and a biasvector
W_conv2 = weight_var([5, 5, 32, 64])
b_conv2 = bias_var([64])

#apply the convolution step (~ compute the input for the convolution neurons) + apply ReLU 
#activation for every conv neuron to get their output for the pooling
out_conv2 = tf.nn.relu(conv(out_pool1, W_conv2) + b_conv2)

#apply the pooling step (out_conv1 as input)
out_pool2 = max_pool22(out_conv2)
debug_out_pool2_shape = tf.shape(out_pool2)


W_fc1 = weight_var([7*7*64, 1024])
b_fc1 = bias_var([1024])

out_pool2_res =tf.reshape(out_pool2, [-1, 7*7*64])
out_1024neurons = tf.nn.relu(tf.matmul(out_pool2_res, W_fc1) + b_fc1)

#add dropout
keep_prob = tf.placeholder("float")
out_1024neurons_drop = tf.nn.dropout(out_1024neurons, keep_prob)

#adding softmax regression as last "layer"
W_fc2 = weight_var([1024,10])
b_fc2 = bias_var([10])

y_conv = tf.nn.softmax(tf.matmul(out_1024neurons_drop, W_fc2) + b_fc2)

#training
cross_entropy = tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


for i in range(20):
    batch = mnist.train.next_batch(50)
    if i%10 == 0:
        train_accuracy, shape1, shape2 = sess.run([accuracy,debug_out_pool1_shape,debug_out_pool2_shape], feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('{}, {}'.format(shape1, shape2))
        print('step{0:d}, training accuracy: {1:g}'.format(i, train_accuracy))
    sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

#huge amount of data -> 4GiB neede
#print("test accuracy {0:g}".format(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))



