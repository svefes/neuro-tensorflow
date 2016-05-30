import tensorflow as tf
import clearer as c

#extracting features and making training/test set
li = c.listify_postings('Doepke.csv')
c.get_features(li)
tr, te = c.train_test(li)

x = tf.placeholder(tf.float32, [None, 26])

theta = tf.Variable(tf.zeros([26, 3]))
b  = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, theta) + b)# evtl. ohne b (stanford)

#training
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))# evtl vorne auch reduce_sum(stanford)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#init
init = tf.initialize_all_variables()

#run
sess = tf.Session()
sess.run(init)

def train(n = 100):
    for i in range(n):
        train_x, train_y = c.next_batch(tr, 'all')
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

def test():
    test_x, test_y = c.next_batch(te, 'all')
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

