import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from SequenceGenerator import SequenceGenerator
from RocketBall import RocketBall

NUM_HIDDEN_UNITS=4
TIMESTEPS=5
DIMS=3
COUNT=5
BATCH_SIZE=1
OUTPUT_SIZE=6


data=tf.placeholder(tf.float32, [BATCH_SIZE, TIMESTEPS, DIMS])
target=tf.placeholder(tf.float32,[BATCH_SIZE,TIMESTEPS,OUTPUT_SIZE])

input_data=np.random.rand(COUNT,TIMESTEPS,DIMS)
output_data=np.random.rand(COUNT,TIMESTEPS,OUTPUT_SIZE)

cell=tf.contrib.rnn.GRUCell(num_units=NUM_HIDDEN_UNITS)

outputs,states=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32,initial_state=None)#returns time,batch,outputsize
outputs=tf.unstack(tf.transpose(outputs,[1,0,2]),name="unstack_LSTM")

output_bias=tf.Variable(tf.random_normal([1,OUTPUT_SIZE]))
output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,OUTPUT_SIZE]),name="Weights_Output")

tf.summary.histogram("hist_weights",output_layer)

variables=tf.trainable_variables()
#variable_summaries(output_layer)
#output_bias=tf.Variable(tf.random_normal([1,LENGTH_OUTPUT]))

l_output=[tf.matmul(out_t,output_layer)+output_bias for out_t in outputs]
l_output=tf.transpose(l_output,[1,0,2])


loss=tf.reduce_mean(tf.square(tf.subtract(target,l_output)),axis=2,name="MSE")
optimizer= tf.train.AdamOptimizer()

merged=tf.summary.merge_all()

init=tf.global_variables_initializer()
with tf.Session() as sess:
    writer=tf.summary.FileWriter("/home/emanuel/Coding/tensorflow/tmp/sub3",sess.graph)
    sess.run(init)

    outs,stats,vars,summary=sess.run([outputs,states,variables,merged],feed_dict={data:[input_data[0]],target:[output_data[0]]})
    print(outs,stats)
    print(np.shape(vars))

    writer.add_summary(summary)
    for i in range(len(vars)):
        print(np.shape(vars[i]))

    writer.close()

