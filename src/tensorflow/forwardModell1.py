import tensorflow as tf
import numpy as np
from SequenceGenerator import SequenceGenerator
from RocketBall import RocketBall

NUM_TRAINING=1000
COUNT_TIMESTEPS=100
COUNT_EPOCHS=30

NUM_HIDDEN_UNITS=16
LENGTH_INPUT=4
LENGTH_OUTPUT=2

#learning parameters
epsilon=10**(-8)
learning_rate=0.001
beta1=0.9
beta2=0.999

rocketBall=RocketBall.standardVersion()

dt=1./30.


def createNetworkModel(data):
    cell=tf.contrib.rnn.LSTMCell(num_units=NUM_HIDDEN_UNITS,use_peepholes=False,state_is_tuple=True)


    initial_state = cell.zero_state(1, tf.float32)
    val,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32,initial_state=initial_state)

    val=tf.unstack(tf.transpose(val,[1,0,2]))

    output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,LENGTH_OUTPUT]))
    output_bias=tf.Variable(tf.random_normal([1,LENGTH_OUTPUT]))

    l_output=[tf.matmul(out_t,output_layer)+output_bias for out_t in val]
    l_output=tf.transpose(l_output,[1,0,2])

    return l_output


def initNetwork():
    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    return sess

def restoreNetwork(path):
    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess, path)

    return sess

def trainNetwork(data,target,inputs,outputs):

    output=createNetworkModel(data)
    rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target,output))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(rmse)

    #initialize all Variables
    sess=initNetwork()

    saver=tf.train.Saver()
    count_epochs=COUNT_EPOCHS
    for i in range(count_epochs):
        sum_error=0.
        for j in range(len(inputs)):
            input_=[inputs[j]]
            output_=[outputs[j]]
            _,r=sess.run([optimizer,rmse],feed_dict={data:input_,target:output_})

            sum_error+=r
        print("Epoch " + str(i) + ": " + str(sum_error/NUM_TRAINING))
        #saver.save(sess, "/home/emanuel/Coding/tensorflow/SessionData/sessTest.chkpt")



inputs=[SequenceGenerator.generateCustomInputs_tuple(rocketBall,COUNT_TIMESTEPS,0.3) for i in range(NUM_TRAINING)]
outputs=[SequenceGenerator.runInputs_2tuple(rocketBall,input,dt) for input in inputs]

data=tf.placeholder(tf.float32,[None,COUNT_TIMESTEPS,LENGTH_INPUT])
target=tf.placeholder(tf.float32,[None,COUNT_TIMESTEPS,LENGTH_OUTPUT])

trainNetwork(data,target,inputs,outputs)