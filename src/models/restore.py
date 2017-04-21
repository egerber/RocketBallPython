import tensorflow as tf
import numpy as np
from SequenceGenerator import SequenceGenerator
from RocketBall import RocketBall

NUM_TRAINING=100
COUNT_TIMESTEPS=100

NUM_HIDDEN_UNITS=16
LENGTH_INPUT=2
LENGTH_OUTPUT=6

#learning parameters
epsilon=10**(-8)
learning_rate=0.001
beta1=0.9
beta2=0.999

rocketBall=RocketBall.standardVersion()

fps=30.
dtmsec=1000./fps
dt=dtmsec/1000.


inputs=[SequenceGenerator.generateCustomInputs(COUNT_TIMESTEPS,0.3) for i in range(NUM_TRAINING)]
outputs=[SequenceGenerator.runInputs_6tuple(rocketBall, input, dt) for input in inputs]


data=tf.placeholder(tf.float32,[None,COUNT_TIMESTEPS,LENGTH_INPUT])
target=tf.placeholder(tf.float32,[None,COUNT_TIMESTEPS,LENGTH_OUTPUT])

cell=tf.contrib.rnn.LSTMCell(num_units=NUM_HIDDEN_UNITS,use_peepholes=True,state_is_tuple=True)

val,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
val=tf.unstack(tf.transpose(val,[1,0,2]))

output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,LENGTH_OUTPUT]))

l_output=[tf.matmul(out_t,output_layer) for out_t in val]
l_output=tf.transpose(l_output,[1,0,2])

rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target,l_output))))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
gradients=optimizer.compute_gradients(rmse)

saver=tf.train.Saver()


sess=tf.Session()

saver.restore(sess,"/home/emanuel/Coding/tensorflow/SessionData/sessTest.chkpt")

input_=[inputs[0]]
output_=[outputs[0]]
g,r=sess.run([gradients,rmse],feed_dict={data:input_,target:output_})

print(np.shape(g),np.shape(g[0]),np.shape(g[1]),np.shape(g[2]),np.shape(g[3]),np.shape(g[4]),np.shape(g[5]))
print(g)