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


        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_speed=np.zeros((1,self.configuration["size_output"]))

        self.c_state=None
        self.h_state=None
        self.speed=None

        ##
        self.count_timesteps=None

    def create_last_timestep_optimizer(self,clip_min=0.,clip_max=1.):
        last_output=self.outputs[-1]
        with vs.variable_scope("LossCalculation") as loss:
            self.rmse=tf.sqrt(tf.square(tf.subtract(self.target,last_output)),name="RMSE")
            self.mse=tf.multiply(0.5,tf.square(tf.subtract(self.target,last_output)),name="MSE")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=self.beta1,
                                                    beta2=self.beta2,
                                                    epsilon=self.epsilon,name="MSE_OPTIMIZER")

            self.minimizer=self.optimizer.minimize(self.mse,var_list=[self.inputs])

            self.clipping=self.inputs.assign(tf.clip_by_value(self.inputs, clip_min,clip_max))

    def create_all_timesteps_optimizer(self,clip_min=0.,clip_max=1.):
        with vs.variable_scope("LossCalculation") as loss:
            self.rmse=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="RMSE")
            self.mse=tf.multiply(0.5,tf.reduce_sum(tf.square(tf.subtract(self.target,self.outputs)),axis=2),name="MSE")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=self.beta1,
                                                    beta2=self.beta2,
                                                    epsilon=self.epsilon,name="MSE_OPTIMIZER")

            self.minimizer=self.optimizer.minimize(self.mse,var_list=[self.inputs])

            self.clipping=self.inputs.assign(tf.clip_by_value(self.inputs, clip_min,clip_max))

    #TODO works only with outputsize=4 so far, needs to be generalized
    def create_dontcare_optimizer(self,clip_min=0.,clip_max=1.):
        with vs.variable_scope("LossCalculation") as loss:

            slice_do_care=tf.slice(self.outputs,[0,0,0],[1,self.count_timesteps,2])
            slice_dont_care=tf.slice(self.target,[0,0,2],[1,self.count_timesteps,2])
            dont_care_outputs=tf.concat([slice_do_care,slice_dont_care],2)
            self.rmse=tf.sqrt(tf.square(tf.subtract(self.target,dont_care_outputs)),name="RMSE")
            self.mse=tf.multiply(0.5,tf.square(tf.subtract(self.target,dont_care_outputs)),name="MSE")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=self.beta1,
                                                    beta2=self.beta2,
                                                    epsilon=self.epsilon,name="MSE_OPTIMIZER")

            self.minimizer=self.optimizer.minimize(self.mse,var_list=[self.inputs])

            self.clipping=self.inputs.assign(tf.clip_by_value(self.inputs, clip_min, clip_max))


    def create_singleOutput(self,count_timesteps):
        t_begin=time.time()
        self.count_timesteps=count_timesteps
        self.inputs=tf.Variable(tf.random_uniform([1,count_timesteps,self.configuration["size_input"]],minval=0.,maxval=1.))
        self.target=tf.placeholder(tf.float32,[1,self.configuration["size_output"]])

        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.last_state=tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)

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
                #initial_state=cell.zero_state(1,tf.float32)
                lstm_output,state=cell(inputData[0],self.last_state,lstm_scope) #use last_state
                lstm_outputs.append(lstm_output)
                (self.next_c,self.next_h)=state
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
                    outputs=tf.matmul(lstm_outputs[-1],output_layer,name="Multiply_lstm_output")+biases_outputs
                else:
                    outputs=tf.matmul(lstm_outputs[-1],output_layer,name="Multiply_lstm_output")

                #outputs=tf.transpose(outputs,[1,0,2],"Transpose_output")

            self.outputs=outputs


        t_end=time.time()
        print("Network Creation Done! (time: " + str(t_end-t_begin) + ")")

    def create_self_feeding(self,count_timesteps):

        self.count_timesteps=count_timesteps
        self.inputs=tf.Variable(tf.random_uniform([1,count_timesteps,self.configuration["size_input"]-self.configuration["size_output"]],minval=0.,maxval=1.))
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_output"]])

        self.speed = tf.placeholder(tf.float32,[1,self.configuration["size_output"]])
        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.last_state=tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)

        inputData=tf.unstack(tf.transpose(self.inputs,[1,0,2]))
        lstm_outputs = []
        final_outputs = []
        with vs.variable_scope("inverse/network"):
            cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)
            with vs.variable_scope("OUTPUT"):
                output_layer=vs.get_variable("weights_output",
                                             [self.configuration["num_hidden_units"],self.configuration["size_output"]],
                                             initializer=tf.random_normal_initializer())
                biases_outputs=vs.get_variable("biases_output",[1,self.configuration["size_output"]],initializer=tf.random_normal_initializer())
            _speed=self.speed
            with vs.variable_scope("LSTM") as lstm_scope:
                jointInput=tf.concat([inputData[0],_speed],1)
                lstm_output,state=cell(jointInput,self.last_state,lstm_scope) #use last_state
                _speed=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs

                #lstm_outputs.append(lstm_output)
                final_outputs.append(_speed)
                (self.next_c,self.next_h)=state
                self.next_speed=_speed

            with vs.variable_scope("LSTM",reuse=True) as lstm_scope:
                for index,inp in enumerate(inputData[1:]):
                    jointInput=tf.concat([inp,_speed],1)
                    lstm_output,state=cell(jointInput,state,lstm_scope)
                    _speed=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs
                    final_outputs.append(_speed)

            self.outputs=tf.transpose(final_outputs,[1,0,2],"Transpose_output")



    def create(self,count_timesteps):
        t_begin=time.time()
        self.count_timesteps=count_timesteps
        self.inputs=tf.Variable(tf.random_uniform([1,count_timesteps,self.configuration["size_input"]],minval=0.,maxval=1.))
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_output"]])

        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.last_state=tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)

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

            with vs.variable_scope("LSTM") as lstm_scope:
                    #initial_state=cell.zero_state(1,tf.float32)
                    lstm_output,state=cell(inputData[0],self.last_state,lstm_scope) #use last_state
                    lstm_outputs.append(lstm_output)
                    (self.next_c,self.next_h)=state
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

    def infer_self_feeding(self,target_outputs,count_iterations):

        for i in range(count_iterations):
            result_c,result_h,result_speed,i,o,r,_,_=self.sess.run([self.next_c,self.next_h,self.next_speed,self.inputs,self.outputs,self.rmse,self.minimizer,self.clipping],
                                                      feed_dict={self.target:[target_outputs],
                                                                 self.c_state:self.last_c,
                                                                 self.h_state:self.last_h,
                                                                 self.speed:self.last_speed})


        self.last_c=result_c
        self.last_h=result_h
        self.last_speed=result_speed

        return i[0]
    def infer(self,target_outputs,count_iterations):

        for i in range(count_iterations):
            result_c,result_h,i,o,r,_,_=self.sess.run([self.next_c,self.next_h,self.inputs,self.outputs,self.rmse,self.minimizer,self.clipping],
                                feed_dict={self.target:[target_outputs],
                                           self.c_state:self.last_c,
                                           self.h_state:self.last_h})
            #print("error: ",r)
            #print("inputs: ",i)
            #print("output: ",o)
        self.last_c=result_c
        self.last_h=result_h
        return i[0]

    def reset(self):
        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_speed=np.zeros((1,self.configuration["size_output"]))
        #TODO Maybe self.inputs has to be reinitialized
        #self.inputs=tf.Variable(tf.random_uniform([1,self.count_timesteps,self.configuration["size_input"]],minval=0.,maxval=1.))

if __name__=='__main__':
    COUNT_ITERATIONS=100
    COUNT_TIMESTEPS=3

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":4,
        "size_input":4,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }


    #outputs=[[-0.0]*configuration["size_output"] for i in range(COUNT_TIMESTEPS)]
    outputs=[0.0,0.0,0.,0.0]
    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)
    iModel.create(COUNT_TIMESTEPS)
    iModel.create_dontcare_optimizer()
    iModel.restore(path)

    begin=time.time()
    for i in range(20):
        print(iModel.infer([outputs for i in range(COUNT_TIMESTEPS)],COUNT_ITERATIONS)[0])
    end=time.time()
    print(end-begin)