
import tensorflow as tf
import numpy as np
from src.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall
from tensorflow.python.ops import variable_scope as vs
import time
from src.models.helper import *
import os

class forwardModel:


    def __init__(self,configuration):
        #learning parameters
        self.epsilon=10**(-8)
        self.learning_rate=0.001
        self.beta1=0.9
        self.beta2=0.999

        self.outputs=None
        self.sess=None
        self.configuration=configuration
        self.data=None
        self.target=None


    def writeSummary(self,tensor,name):
        mean=tf.reduce_mean(tensor)
        max=tf.reduce_max(tensor)
        min=tf.reduce_min(tensor)
        std = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar(name+"/Mean",mean)
        tf.summary.scalar(name+"/Min", min)
        tf.summary.scalar(name+"/Max",max)
        tf.summary.scalar(name+"/Std",std)

    def create(self,count_timesteps,device='/cpu:0'):
        t_begin=time.time()

        self.data=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_input"]])
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,configuration["size_output"]])
        lstm_outputs=[]
        cell=None
        if(configuration["cell_type"]=="LSTMCell"):
            cell=tf.contrib.rnn.LSTMCell(num_units=configuration["num_hidden_units"],use_peepholes=configuration["use_peepholes"],state_is_tuple=True)
        elif(configuration["cell_type"]=="GRUCell"):
            cell=tf.contrib.rnn.GRUCell(num_units=configuration["num_hidden_units"])
        elif(configuration["cell_type"]=="RNNCell"):
            cell=tf.contrib.rnn.RNNCell(num_units=configuration["num_hidden_units"])
        elif(configuration["cell_type"]=="BasicLSTMCell"):
            cell=tf.contrib.rnn.LSTMCell(num_hidden_units=configuration["num_hidden_units"],state_is_tuple=True)


        inputData=tf.unstack(tf.transpose(self.data,[1,0,2]))

        state=None
        with vs.variable_scope("LSTM") as lstm_scope:
            initial_state=cell.zero_state(1,tf.float32)
            lstm_output,state=cell(inputData[0],initial_state,lstm_scope)
            lstm_outputs.append(lstm_output)
        with vs.variable_scope("LSTM",reuse=True) as lstm_scope:
            for index,inp in enumerate(inputData[1:]):
                lstm_output,state=cell(inp,state,lstm_scope)
                lstm_outputs.append(lstm_output)

        with vs.variable_scope("OUTPUT"):

            output_layer=tf.Variable(tf.random_normal([configuration["num_hidden_units"],configuration["size_output"]]),name="Weights_output")
            if(configuration["use_biases"]):
                biases_outputs=tf.Variable(tf.random_normal([1,configuration["size_output"]]))
                outputs=[tf.matmul(out_t,output_layer,name="Multiply_lstm_output")+biases_outputs for out_t in lstm_outputs]
            else:
                outputs=[tf.matmul(out_t,output_layer,name="Multiply_lstm_output") for out_t in lstm_outputs]

            outputs=tf.transpose(outputs,[1,0,2],"Transpose_output")

        self.outputs=outputs

        t_end=time.time()
        print("Network Creation Done! (time: " + str(t_end-t_begin) + ")")



    def train(self,inputs,outputs,count_epochs,logging=True,save=True,restore=None,override=False):
        t_begin=time.time()

        with vs.variable_scope("LossCalculation") as loss:
            rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="RMSE")
            mse=tf.multiply(0.5,tf.reduce_sum(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="MSE")

            if(logging):
                self.writeSummary(rmse,"RMSE")
                self.writeSummary(mse,"MSE")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon).minimize(mse,name="MSE_OPTIMIZER")


        #initialize vars from optimizer
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(init_new_vars_op)


        if(logging):
            merged=tf.summary.merge_all()

        loggingDirectory=os.path.dirname(__file__)+"/../../data/logs/"
        checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"

        confString=createConfigurationString(self.configuration)
        if(logging):
            writer=tf.summary.FileWriter(loggingDirectory+confString,self.sess.graph)

        for i in range(count_epochs):
            sum_error=0.
            begin=time.time()#take time before calculating the epoche
            for j in range(len(inputs)):
                input_=[inputs[j]]
                output_=[outputs[j]]

                _,r=self.sess.run([optimizer,rmse],feed_dict={self.data:input_,self.target:output_})
                sum_error+=r
                if(logging and j%100==0):
                    summary=self.sess.run([merged],feed_dict={self.data:input_,self.target:output_})[0]
                    writer.add_summary(summary)

            if(((not restore and save) or (restore and override)) and i%5==0): #create checkpoint after 5 epochs
                self.save(checkpointDirectory+confString+".chkpt")

            end=time.time()
            print("Epoch " + str(i) + ": " + str(sum_error/len(inputs)))
            print("Time: ", str(end-begin))

        t_end=time.time()
        print("Training Done! (time: "+str(t_end-t_begin)+")")

    def init(self):
        t_begin=time.time()
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess=sess
        t_end=time.time()
        print("Initialization Done! (time: " + str(t_end-t_begin) +")")

    def restore(self,path):
        t_begin=time.time()
        sess=tf.Session()
        self.saver=tf.train.Saver()
        self.saver.restore(sess,path)

        self.sess=sess
        print("Restoring Done! (time: " + str(time.time()-t_begin)+ ")")

    def save(self,path):
        if(self.saver is None):
            self.saver=tf.train.Saver()

        self.saver.save(self.sess, path)

if __name__=='__main__':
    COUNT_EPOCHS=12
    COUNT_TIMESTEPS=12
    NUM_TRAINING=14

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
        "tag":"small_training_test"
    }


    fmodel=forwardModel(configuration)
    fmodel.create(COUNT_TIMESTEPS,'/cpu:0')
    fmodel.init()
    fmodel.train(inputs, outputs,count_epochs=COUNT_EPOCHS,logging=True,save=False)

