import tensorflow as tf 
import numpy as np

#print np.random.rand(2,100) #create a 2*100 random array
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.1,0.2],x_data)+0.6
print y_data


#print tf.zeros([0])# 1-d vector , init as 0  tf.zeros([100]) means 100-d vector
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1,1))
#tf.get_variable: Gets an existing variable with these parameters or create a new one. You can also use initializer.
y=tf.matmul(W,x_data)+b

loss=tf.reduce_mean(tf.square(y-y_data))
opt=tf.train.GradientDescentOptimizer(0.5)
train=opt.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in xrange(0,201):
	sess.run(train)
	print step,sess.run(W),sess.run(b)
