import tensorflow as tf

params1 = tf.constant([1,2,3,4])
params2 = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
b = tf.constant(3)
c = tf.constant([[[0, 1], [0,0]],[[1,0],[1,1]]])

b1 = tf.gather(params1, b)
b2 = tf.gather(params2, b)
c1 = tf.gather(params1, c)
c2 = tf.gather(params2, c)

sess = tf.Session()
c2_val = sess.run([c2])
print(c2_val)
