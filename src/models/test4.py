import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope

data=tf.placeholder(tf.float32,[1,3])

c_state = tf.placeholder(tf.float32,[2,1,5])
h_state = tf.placeholder(tf.float32,[2,1,5])

last_state=tf.contrib.rnn.LSTMStateTuple(tf.unstack(c_state),tf.unstack(h_state))

#ib.rnn.LSTMCell(num_units=5,use_peepholes=True,state_is_tuple=True)

stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=5,use_peepholes=True,state_is_tuple=True) for _ in range(2)],state_is_tuple=True)


result=stacked_lstm(data,last_state)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
aa=np.zeros((2,1,5))
bb=np.zeros((2,1,5))

for i in range(10):
    ls,res=sess.run([last_state,result],feed_dict={data:[[1.,2.,3.]],c_state:aa, h_state:bb})
    print(res)
    print("")
