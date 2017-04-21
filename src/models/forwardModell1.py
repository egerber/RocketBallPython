
import tensorflow as tf
import numpy as np
from SequenceGenerator import SequenceGenerator
from RocketBall import RocketBall
from tensorflow.python.ops import variable_scope as vs
import time
from models.helper import *

#learning parameters
epsilon=10**(-8)
learning_rate=0.001
beta1=0.9
beta2=0.999

rocketBall=RocketBall.standardVersion()

dt=1./30.

val=None
val2=None
l_output=None
last_state=None
initial_state=None
#processes one timestep of the forward-pass
# input is a tensor 1xsize_input (where 1 is the batchsize)

def createTimestepModel(input,last_state,configuration):

    cell=None
    if(configuration["cell_type"]=="LSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_units=configuration["num_hidden_units"],use_peepholes=configuration["use_peepholes"],state_is_tuple=True)
    elif(configuration["cell_type"]=="GRUCell"):
        cell=tf.contrib.rnn.GRUCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="RNNCell"):
        cell=tf.contrib.rnn.RNNCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="BasicLSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_hidden_units=configuration["num_hidden_units"],state_is_tuple=True)
    #initial_state=cell.zero_state(1,tf.float32)
    with vs.variable_scope("LSTM") as lstm_scope:
        lstm_output,state=cell(input,last_state,lstm_scope)


    with vs.variable_scope("OUTPUT"):
        output_layer=tf.Variable(tf.random_normal([configuration["num_hidden_units"],configuration["size_output"]]),name="Weights_output")
        if(configuration["use_biases"]):
            biases_outputs=tf.Variable(tf.random_normal([1,configuration["size_output"]]))
            l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs
        else:
            l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")

    return l_output,state


def createNetworkModel2(data,configuration):
    global l_output
    outputs=[]
    cell=None
    if(configuration["cell_type"]=="LSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_units=configuration["num_hidden_units"],use_peepholes=configuration["use_peepholes"],state_is_tuple=True)
    elif(configuration["cell_type"]=="GRUCell"):
        cell=tf.contrib.rnn.GRUCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="RNNCell"):
        cell=tf.contrib.rnn.RNNCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="BasicLSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_hidden_units=configuration["num_hidden_units"],state_is_tuple=True)


    inputData=tf.unstack(tf.transpose(data,[1,0,2]))

    state=None
    with vs.variable_scope("LSTM") as lstm_scope:
        initial_state=cell.zero_state(1,tf.float32)
        output,state=cell(inputData[0],initial_state,lstm_scope)
        outputs.append(output)
    with vs.variable_scope("LSTM",reuse=True) as lstm_scope:
        for index,inp in enumerate(inputData[1:]):
            output,state=cell(inp,state,lstm_scope)
            outputs.append(output)

    with vs.variable_scope("OUTPUT"):

        output_layer=tf.Variable(tf.random_normal([configuration["num_hidden_units"],configuration["size_output"]]),name="Weights_output")
        if(configuration["use_biases"]):
            biases_outputs=tf.Variable(tf.random_normal([1,configuration["size_output"]]))
            l_output=[tf.matmul(out_t,output_layer,name="Multiply_lstm_output")+biases_outputs for out_t in outputs]
        else:
            l_output=[tf.matmul(out_t,output_layer,name="Multiply_lstm_output") for out_t in outputs]

        l_output=tf.transpose(l_output,[1,0,2],"Transpose_output")


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

def create_summary(tensor,name):
    mean=tf.reduce_mean(tensor)
    max=tf.reduce_max(tensor)
    min=tf.reduce_min(tensor)
    std = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
    tf.summary.scalar(name+"/Mean",mean)
    tf.summary.scalar(name+"/Min", min)
    tf.summary.scalar(name+"/Max",max)
    tf.summary.scalar(name+"/Std",std)


def trainForwardModel(inputs, outputs, configuration,count_epochs,logging=True,save=True,restore=None,override=False):
    begin=time.time()

    count_timesteps=len(inputs[0])
    data=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_input"]])
    target=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_output"]])


    output=createNetworkModel2(data,configuration)
    with vs.variable_scope("LossCalculation") as loss:
        rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target,output)),axis=2),name="RMSE")
        mse=tf.reduce_mean(tf.square(tf.subtract(target,output)),axis=2,name="MSE")

        if(logging):
            create_summary(rmse,"RMSE")
            create_summary(mse,"MSE")


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(mse,name="MSE_OPTIMIZER")
    merged=tf.summary.merge_all()


    #initialize all Variables

    saver=tf.train.Saver()
    sess=None
    if(restore is None):
        sess=initNetwork()
    else:
        sess=restoreNetwork(restore)

    if(logging):
        confString=createConfigurationString(configuration)
        writer=tf.summary.FileWriter("/home/emanuel/Coding/tensorflow/tmp/"+confString,sess.graph)

    end=time.time()
    print("Time for initialization: ", str(end-begin))

    for i in range(count_epochs):
        sum_error=0.
        begin=time.time()#take time before calculating the epoche
        for j in range(len(inputs)):
            input_=[inputs[j]]
            output_=[outputs[j]]
            _,r,summary=sess.run([optimizer,rmse,merged],feed_dict={data:input_,target:output_})
            sum_error+=r
            if(logging and j%100==0):
                writer.add_summary(summary)
        end=time.time()
        if((not restore and save) or (restore and override)):
            saver.save(sess, "/home/emanuel/Coding/tensorflow/SessionData/"+confString+".chkpt")
        print("Epoch " + str(i) + ": " + str(sum_error/NUM_TRAINING))
        print("Time: ", str(end-begin))


    writer.close()

if(__name__=="__main__"):

    COUNT_EPOCHS=50
    COUNT_TIMESTEPS=200
    NUM_TRAINING=2000

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=True

    inputs=[SequenceGenerator.generateCustomInputs_2tuple(COUNT_TIMESTEPS,0.25) for i in range(NUM_TRAINING)]
    outputs=[SequenceGenerator.runInputs_2tuple(rocketBall,input,dt) for input in inputs]


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"mse2"
    }

    trainForwardModel(inputs, outputs, configuration=configuration,count_epochs=COUNT_EPOCHS,logging=True,save=True)

