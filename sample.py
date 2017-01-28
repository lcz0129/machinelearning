import tensorflow as tf
import numpy as np
import pylab

x_data = np.random.rand(100).astype(np.float32)
print(x_data)
noise = np.random.normal(scale=0.01, size=len(x_data))
print("noise is ",noise)
y_data = x_data * 0.1 + 0.3 + noise

# Uncomment the following line to plot our input data.
pylab.plot(x_data, y_data, '.')

W = tf.Variable(tf.random_uniform([1], 0.0, 1.0))

b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

print(W) 
print(b)

loss = tf.reduce_mean(tf.square(y - y_data))  # Create an operation that calculates loss.
optimizer = tf.train.GradientDescentOptimizer(0.5)  # Create an optimizer.
train = optimizer.minimize(loss)  # Create an operation that minimizes loss.
init = tf.initialize_all_variables()  # Create an operation initializes all the variables.

# Uncomment the following 3 lines to see what 'loss', 'optimizer' and 'train' are.

#print(tf.get_default_graph().as_graph_def())

sess = tf.Session()
sess.run(init)
y_initial_values = sess.run(y)  # Save initial values for plotting later.
print y_initial_values


for step in range(201):
    sess.run(train)





pylab.plot(x_data, y_data, '.', label="target_values")
#pylab.plot(x_data, y, '.', label="initial_values")
pylab.plot(x_data, y_initial_values, ".", label="initial_values")


pylab.plot(x_data, sess.run(y), ".", label="trained_values")
pylab.legend()
pylab.ylim(0, 1.0)
pylab.show()