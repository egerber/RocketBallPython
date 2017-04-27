import tensorflow as tf
import numpy as np
from src.SequenceGenerator import SequenceGenerator
from src.RocketBall import RocketBall
from tensorflow.python.ops import variable_scope as vs
import time
from src.models.helper import *
import os
from src.models.forwardModel import forwardModel

class inverseModel:

    def __init__(self,configuration):
        forwardModel.__init__(self,configuration)
        self.epsilon=10**(-8)
        self.learning_rate=0.01
        self.beta1=0.9
        self.beta2=0.9

        self.configuration=configuration
        self.inputs=None #Variable
        self.outputs=None #Placeholder


    def create(self,count_timesteps,device='/cpu:0'):
        t_begin=time.time()

        self.inputs=tf.Variable(tf.random_uniform([1,count_timesteps,self.configuration["size_input"]],minval=0.,maxval=1.))
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_output"]])

        lstm_outputs=[]
        cell=None

        with vs.variable_scope("inverse/network"):
            if(self.configuration["cell_type"]=="LSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)
            elif(self.configuration["cell_type"]=="GRUCell"):
                cell=tf.contrib.rnn.GRUCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="RNNCell"):
                cell=tf.contrib.rnn.RNNCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="BasicLSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_hidden_units=self.configuration["num_hidden_units"],state_is_tuple=True)

            inputData=tf.unstack(tf.transpose(self.inputs,[1,0,2]))

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

            with vs.variable_scope("LossCalculation") as loss:
                self.rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="RMSE")
                self.mse=tf.multiply(0.5,tf.reduce_sum(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="MSE")

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               beta1=self.beta1,
                                               beta2=self.beta2,
                                               epsilon=self.epsilon,name="MSE_OPTIMIZER")

                self.minimizer=self.optimizer.minimize(self.mse,var_list=[self.inputs])

                self.clipping=self.inputs.assign(tf.clip_by_value(self.inputs, 0.0,1.0))


        t_end=time.time()
        print("Network Creation Done! (time: " + str(t_end-t_begin) + ")")



    def restore(self,path):
        t_begin=time.time()

        #1st: initialize all variables (includes optimizer variables)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #2nd: set weights from pretrained model
        #sess=tf.Session()
        with vs.variable_scope("inverse"):
            with vs.variable_scope("network",reuse=True) as scope:
                #with vs.variable_scope("OUTPUT",reuse=True) as scope:
                self.saver=tf.train.Saver(var_list={"LSTM/w_f_diag": vs.get_variable("LSTM/w_f_diag"),
                                                    "LSTM/w_i_diag": vs.get_variable("LSTM/w_i_diag"),
                                                    "LSTM/w_o_diag": vs.get_variable("LSTM/w_o_diag"),
                                                    "LSTM/biases": vs.get_variable("LSTM/biases"),
                                                    "LSTM/weights": vs.get_variable("LSTM/weights"),
                                                    "OUTPUT/weights_output":vs.get_variable("OUTPUT/weights_output"),
                                                    "OUTPUT/biases_output":vs.get_variable("OUTPUT/biases_output")})
        self.saver.restore(sess,path)

        self.sess=sess
        print("Restoring Done! (time: " + str(time.time()-t_begin)+ ")")

    def infer(self,target_outputs,count_iterations):

        for i in range(count_iterations):
            i,o,r,_,_=self.sess.run([self.inputs,self.outputs,self.rmse,self.minimizer,self.clipping],feed_dict={self.target:[target_outputs]})
            print("errors",r)
            print("outputs:",o)
            print("inputs:",i)
        return i


if __name__=='__main__':
    COUNT_ITERATIONS=120
    COUNT_TIMESTEPS=10

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False
    rocketBall.use_sigmoid=False

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }


    outputs=[[-0.05]*configuration["size_output"] for i in range(COUNT_TIMESTEPS)]
    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)
    iModel.create(COUNT_TIMESTEPS,'/cpu:0')
    iModel.restore(path)

    print(iModel.infer(outputs,COUNT_ITERATIONS))

    #fModel=forwardModel.createFromOld(configuration,COUNT_TIMESTEPS,path)
    #print(fModel([[0.1,0.9]]))

    #print(iModel.infer(outputs,count_iterations=COUNT_ITERATIONS))
