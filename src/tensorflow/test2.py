import tensorflow as tf
import numpy as np
from SequenceGenerator import SequenceGenerator
from RocketBall import RocketBall

NUM_HIDDEN_UNITS=10
cell=tf.contrib.rnn.LSTMCell(num_units=NUM_HIDDEN_UNITS,use_peepholes=False,state_is_tuple=True)
state = cell.zero_state(1, tf.float32) #setup initial state

inputs=np.random.rand(100,10)
data=tf.placeholder(tf.float32,[None,10])

out,state=cell(data[0],state=state)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

d=sess.run([data],feed_dict={data:inputs})
print(d)


    #val,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
    #val=tf.unstack(tf.transpose(val,[1,0,2]))

    #output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,LENGTH_OUTPUT]))
