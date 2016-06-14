import tensorflow as tf
import db_clearer as c
import numpy as np


def setup_network(id_net, number_qualities=3, number_feat=None, eta=2.0):
    if number_feat is not None:
        global x
        global theta1
        global b1
        global y
        global y_
        global accuracy
        global sess
        global train_step
        global train_mean_step
        global logi
        global cross_entropy
        global pred
        global theta2
        global accuracy1
        global prediction
        x = tf.placeholder(tf.float32, [None, number_feat])
        theta = tf.Variable(tf.zeros([number_feat, number_qualities]))
        b  = tf.Variable(tf.zeros([number_qualities]))
        theta1 = tf.Variable(tf.zeros([number_feat, 15]))
        b1  = tf.Variable(tf.zeros([15]))
        theta2 = tf.Variable(tf.zeros([15, number_qualities]))
        b2  = tf.Variable(tf.zeros([number_qualities]))
        
        #y = tf.nn.softmax(tf.matmul(x, theta) + b)# evtl. ohne b (stanford)
        interim = tf.sigmoid(tf.matmul(x, theta1)+b1)
        prediction = tf.sigmoid(tf.matmul(interim, theta2)+b2)
        pred = tf.matmul(x, theta)+b
        #training
        y_ = tf.placeholder(tf.float32, [None, number_qualities])
        #do _not_ use class '0' as it would eval log(0) to nan'
        mean_squared_error = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(y_-prediction)))
        #cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(pred),reduction_indices = [1]))# evtl vorne auch reduce_sum(stanford)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,y_)
        train_step = tf.train.GradientDescentOptimizer(float(eta)).minimize(cross_entropy)
        train_mean_step = tf.train.GradientDescentOptimizer(float(eta)).minimize(mean_squared_error)
        
        #accuracy
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(pred),1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #accuracy
        correct_prediction1 = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        #init
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

def train_soft(n = 100):
    for i in range(n):
        train_x, train_y = c.next_training_batch('all')
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

def train_mean(n=100):
        for i in range(n):
            train_x, train_y = c.next_training_batch('all')
            print('epoch {}:'.format(i))
            print(sess.run(prediction, feed_dict={x:train_x}))
            print(sess.run(theta1))
            print(sess.run(b1))
            print('theta2:')
            print(sess.run(theta2))
            sess.run(train_mean_step, feed_dict={x: train_x, y_: train_y})

def test_soft():
    test_x, test_y = c.next_test_batch()
    return(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    
def test_mean():
    test_x, test_y = c.next_test_batch()
    print(sess.run(prediction, feed_dict={x: test_x, y_: test_y}))
    print(test_y)
    print(sess.run(accuracy1, feed_dict={x: test_x, y_: test_y}))

