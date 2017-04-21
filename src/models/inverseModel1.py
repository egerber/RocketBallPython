import tensorflow as tf
import numpy as np
from src.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall
from tensorflow.python.ops import variable_scope as vs
import time
from src.models.helper import *
import os
from src.models.forwardModell1 import *


def createInverseModel2(data,configuration):
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

def inferInputs(output,timesteps,configuration,count_iterations):
