import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from src.DataGenerators.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall

NUM_HIDDEN_UNITS=10


timesteps=10
dim=2
count=100
BATCH_SIZE=1
OUTPUT_SIZE=4

def createNetworkModel(data):
    global val,val2,l_output
    cell=tf.contrib.rnn.LSTMCell(num_units=NUM_HIDDEN_UNITS,use_peepholes=True,state_is_tuple=True)

    initial_state = cell.zero_state(1, tf.float32)

    val,_=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32,initial_state=initial_state)#returns time,batch,outputsize

    val=tf.unstack(tf.transpose(val,[1,0,2]),name="unstack_LSTM")

    output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,LENGTH_OUTPUT]),name="Weights_Output")
    #variable_summaries(output_layer)
    output_bias=tf.Variable(tf.random_normal([1,LENGTH_OUTPUT]))

    l_output=[tf.matmul(out_t,output_layer)+output_bias for out_t in val]
    l_output=tf.transpose(l_output,[1,0,2])


    return l_output


outputs=None
output=None
def createModel(data):
    global outputs,output
    outputs=[np.random.rand(1,NUM_HIDDEN_UNITS) for i in range(timesteps)]
    cell=tf.contrib.rnn.LSTMCell(num_units=NUM_HIDDEN_UNITS,use_peepholes=False,state_is_tuple=True)
    inputData=tf.unstack(tf.transpose(data,[1,0,2]))

    state=None
    with vs.variable_scope("LSTM") as lstm_scope:

        initial_state=cell.zero_state(1,tf.float32)
        output,state=cell(inputData[0],initial_state,lstm_scope)
        outputs[0]=output
    with vs.variable_scope("LSTM",reuse=True) as lstm_scope:

        for index,inp in enumerate(inputData[1:]):
            output,state=cell(inp,state,lstm_scope)
            outputs[index]=output


    #output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,OUTPUT_SIZE]))
    #l_output=[tf.matmul(out_t,output_layer) for out_t in outputs]
    #return l_output
    return None


inputs=np.random.rand(100,timesteps,dim)

data=tf.placeholder(tf.float32, [BATCH_SIZE, timesteps, dim])


l_output=createModel(data)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


l=sess.run([outputs],feed_dict={data:[inputs[1]]})
print(l)
print(outputs)
    #val,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
    #val=tf.unstack(tf.transpose(val,[1,0,2]))

    #output_layer=tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS,LENGTH_OUTPUT]))
