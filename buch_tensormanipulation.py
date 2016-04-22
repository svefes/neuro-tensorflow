import tensorflow as tf
import numpy as np
from bokeh.plotting import figure, output_file, show

#create 2000 datapoints which belong to 2 clouds#
num_points = 2000
points = list()
for i in range(num_points):
	if np.random.random() > 0.5:
		points.append([np.random.normal(0.0, 0.9),np.random.normal(0.0, 0.9)])
	else:
		points.append([np.random.normal(3.0, 0.5),np.random.normal(1.0, 0.5)])

#give a graphic output#
output_file('scatter.html', title = 'kmeans1')
p = figure(title = 'kmeans original data')
p.circle([v[0] for v in points], [v[1] for v in points])
show(p)

#kmeans#
vectors = tf.constant(points)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0, 0],[k, -1]))

expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)),2),0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),reduction_indices = [1]) for c in range(k)])

update_centroides = tf.assign(centroides, means)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(100):
	_, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

res = list()
for t in range(k):
	res.append([])
i = iter(assignment_values)
for r in range(num_points):
	ass = i.next()
	res[ass].append(points[r])

output_file('scatter_res.html', title = 'kmeans2')
p1 = figure(title = 'kmeans clustered')
for i in range(k):
	p1.circle([v[0] for v in res[i]],[v[1] for v in res[i]], fill_color = (255*i/k,255-(255*i/k),0), line_color = (255*i/k,255-(255*i/k),0))
show(p1)


