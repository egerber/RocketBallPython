
import tensorflow as tf
import numpy as np
from src.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall
from tensorflow.python.ops import variable_scope as vs
import time
from src.models.helper import *
import os

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


def createNetworkModel(data,configuration):
    cell=None

    if(configuration["cell_type"]=="LSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_units=configuration["num_hidden_units"],use_peepholes=configuration["use_peepholes"],state_is_tuple=True)
    elif(configuration["cell_type"]=="GRUCell"):
        cell=tf.contrib.rnn.GRUCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="RNNCell"):
        cell=tf.contrib.rnn.RNNCell(num_units=configuration["num_hidden_units"])
    elif(configuration["cell_type"]=="BasicLSTMCell"):
        cell=tf.contrib.rnn.LSTMCell(num_hidden_units=configuration["num_hidden_units"],state_is_tuple=True)


    initial_state = cell.zero_state(1, tf.float32)

    val,_=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32,initial_state=initial_state)#returns time,batch,outputsize

    val=tf.unstack(tf.transpose(val,[1,0,2]),name="unstack_LSTM")

    output_layer=tf.Variable(tf.random_normal([configuration["num_hidden_units"],configuration["size_output"]]),name="Weights_Output")
    #variable_summaries(output_layer)
    output_bias=tf.Variable(tf.random_normal([1,configuration["size_output"]]))

    l_output=[tf.matmul(out_t,output_layer)+output_bias for out_t in val]
    l_output=tf.transpose(l_output,[1,0,2])


    return l_output

def initNetwork():

    sess=tf.Session(config=None) #tf.ConfigProto(log_device_placement=True)
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


def trainForwardModel(inputs, outputs, configuration,count_epochs,logging=True,save=True,restore=None,override=False,device='/cpu:0'):
    begin=time.time()

    count_timesteps=len(inputs[0])

    with tf.device(device):
        data=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_input"]])
        target=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_output"]])


        output=createNetworkModel2(data,configuration)
        with vs.variable_scope("LossCalculation") as loss:
            rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target,output)),axis=2),name="RMSE")
            mse=tf.multiply(0.5,tf.reduce_sum(tf.square(tf.subtract(target,output)),axis=2),name="MSE")

            if(logging):
                create_summary(rmse,"RMSE")
                create_summary(mse,"MSE")


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(mse,name="MSE_OPTIMIZER")
        if(logging):
            merged=tf.summary.merge_all()


        #initialize all Variables

    saver=tf.train.Saver()
    sess=None
    if(restore is None):
        sess=initNetwork()
    else:
        sess=restoreNetwork(restore)

    loggingDirectory=os.path.dirname(__file__)+"/../../data/logs/"
    checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"

    confString=createConfigurationString(configuration)
    if(logging):
            writer=tf.summary.FileWriter(loggingDirectory+confString,sess.graph)

    end=time.time()
    print("Time for initialization: ", str(end-begin))

    for i in range(count_epochs):
        sum_error=0.
        begin=time.time()#take time before calculating the epoche
        for j in range(len(inputs)):
            input_=[inputs[j]]
            output_=[outputs[j]]

            _,r=sess.run([optimizer,rmse],feed_dict={data:input_,target:output_})
            sum_error+=r
            if(logging and j%100==0):
                summary=sess.run([merged],feed_dict={data:input_,target:output_})[0]
                writer.add_summary(summary)


        if(((not restore and save) or (restore and override)) and i%5==0): #create checkpoint after 5 epochs
            saver.save(sess, checkpointDirectory+confString+".chkpt")
        end=time.time()
        print("Epoch " + str(i) + ": " + str(sum_error/NUM_TRAINING))
        print("Time: ", str(end-begin))


    #writer.close()

if(__name__=="__main__"):

    COUNT_EPOCHS=200
    COUNT_TIMESTEPS=100
    NUM_TRAINING=200

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    inputs=[SequenceGenerator.generateCustomInputs_2tuple(COUNT_TIMESTEPS,0.25) for i in range(NUM_TRAINING)]
    outputs=[SequenceGenerator.runInputs_2tuple(rocketBall,input,1./30.) for input in inputs]


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"small_training"
    }


    trainForwardModel(inputs, outputs, configuration=configuration,count_epochs=COUNT_EPOCHS,logging=True,save=True,device='/cpu:0')

