import numpy as np
import tensorflow as tf

#generating datapoints for linear regression#
num_points = 1000
vectors = list()

for i in range(num_points):
	x1 = np.random.normal(0.0,0.55)
	y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
	vectors.append([x1,y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

#defining parameters which have to be optimized#
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#defining loss function that has to be minim#
loss = tf.reduce_mean(tf.square(y - y_data))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#run the training#
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(30):
	sess.run(train)
	print('step {0:>2} : W = {1: 1.5f} and b = {2:1.5f} ; loss --> {3:1.5f}'\
	.format(step, sess.run(W)[0], sess.run(b)[0], sess.run(loss)))	


