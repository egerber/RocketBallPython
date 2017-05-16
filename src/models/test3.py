
import time

import os
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.ops import variable_scope as vs

from src.DataGenerators.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall
from src.models.helper import *


class forwardModel:


    def __init__(self,configuration):
        #learning parameters
        self.epsilon=10**(-8)
        self.learning_rate=0.001
        self.beta1=0.9
        self.beta2=0.999


        self.saver=None
        self.outputs=None
        self.sess=None
        self.configuration=configuration
        self.data=None
        self.target=None

    @staticmethod
    def createNew(configuration,count_timesteps,device='/cpu:0'):
        fmodel=forwardModel(configuration)
        fmodel.create(count_timesteps,device)
        fmodel.init()
        return fmodel

    @staticmethod
    def createFromOld(configuration,count_timesteps,file_path,device='/cpu:0'):
        fmodel=forwardModel(configuration)
        fmodel.create(count_timesteps,device)
        fmodel.restore(file_path)

        return fmodel

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

        self.data=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_input"]])
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_output"]])
        lstm_outputs=[]
        cell=None
        with vs.variable_scope("network"):

            if(self.configuration["cell_type"]=="LSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)
            elif(self.configuration["cell_type"]=="GRUCell"):
                cell=tf.contrib.rnn.GRUCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="RNNCell"):
                cell=tf.contrib.rnn.RNNCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="BasicLSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_hidden_units=self.configuration["num_hidden_units"],state_is_tuple=True)


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
                output_layer=vs.get_variable("weights_output",
                                             [self.configuration["num_hidden_units"],self.configuration["size_output"]],
                                             initializer=tf.random_normal_initializer())
                if(self.configuration["use_biases"]):
                    biases_outputs=vs.get_variable("biases_output",[1,self.configuration["size_output"]],
                                                   initializer=tf.random_normal_initializer())
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

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _,r=self.sess.run([optimizer,rmse],feed_dict={self.data:input_,self.target:output_},options=run_options, run_metadata=run_metadata)

                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)

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

        #1st: initialize all variables (includes optimizer variables)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #2nd: set weights from pretrained model
        with vs.variable_scope("network",reuse=True) as scope:
            #with vs.variable_scope("OUTPUT",reuse=True) as scope:
            self.saver=tf.train.Saver(var_list={"OUTPUT/weights_output":vs.get_variable("OUTPUT/weights_output"),
                                                "LSTM/w_f_diag": vs.get_variable("LSTM/w_f_diag"),
                                                "LSTM/w_i_diag": vs.get_variable("LSTM/w_i_diag"),
                                                "LSTM/w_o_diag": vs.get_variable("LSTM/w_o_diag"),
                                                "LSTM/biases": vs.get_variable("LSTM/biases"),
                                                "LSTM/weights": vs.get_variable("LSTM/weights")})
        self.saver.restore(sess,path)

        self.sess=sess
        print("Restoring Done! (time: " + str(time.time()-t_begin)+ ")")

    def save(self,path):
        if(self.saver is None):
            with vs.variable_scope("network",reuse=True):
                self.saver=tf.train.Saver(var_list={"OUTPUT/weights_output":vs.get_variable("OUTPUT/weights_output"),
                                                    "LSTM/w_f_diag": vs.get_variable("LSTM/w_f_diag"),
                                                    "LSTM/w_i_diag": vs.get_variable("LSTM/w_i_diag"),
                                                    "LSTM/w_o_diag": vs.get_variable("LSTM/w_o_diag"),
                                                    "LSTM/biases": vs.get_variable("LSTM/biases"),
                                                    "LSTM/weights": vs.get_variable("LSTM/weights")})


        self.saver.save(self.sess, path)

    #runs several inputs and returns outputs
    def __call__(self,inputs):
        prediction=self.sess.run(self.outputs,feed_dict={self.data:[inputs]})
        return prediction[0]

    #runs one timestep of a network and returns output
    def runSingleTimestep(self,input):
        pass


if __name__=='__main__':
    COUNT_EPOCHS=31
    COUNT_TIMESTEPS=10
    NUM_TRAINING=100

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=True

    inputs=[SequenceGenerator.generateCustomInputs_2tuple(COUNT_TIMESTEPS,0.25) for i in range(NUM_TRAINING)]
    outputs=[SequenceGenerator.runInputs_2tuple(rocketBall,input,1./30.) for input in inputs]


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"absolute_borders"
    }


    fmodel=forwardModel.createNew(configuration,COUNT_TIMESTEPS)
    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"
    #fmodel.restore(path)
    #print(fmodel.sess.run(tf.global_variables()))
    fmodel.train(inputs, outputs,count_epochs=COUNT_EPOCHS,logging=True,save=True)
