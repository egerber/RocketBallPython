
import time
import random
import os
import tensorflow as tf
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

        #this information allows to fetch the correct weights for saving depending on the created model
        self.version="singlebatch"

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

    def create_dynamicRNN(self,count_timesteps,device='/cpu_0'):
        self.version="multibatch"
        with tf.device(device):
            self.data=tf.placeholder(tf.float32,[None,count_timesteps,self.configuration["size_input"]])
            self.target=tf.placeholder(tf.float32,[None,count_timesteps,self.configuration["size_output"]])


            with vs.variable_scope("network"):

                inputData=tf.unstack(tf.transpose(self.data,[1,0,2]))

                with vs.variable_scope("LSTM") as lstm_scope:
                    cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)
                    lstm_outputs,state=tf.nn.dynamic_rnn(cell,self.data,dtype=tf.float32,scope=lstm_scope)
                    lstm_outputs=tf.unstack(tf.transpose(lstm_outputs,[1,0,2]))
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



    def train(self,inputs,outputs,count_epochs,batchsize=1,logging=True,save=True,restore=None,override=False):
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

        count_batches=int(len(inputs)/batchsize)
        for i in range(count_epochs):
            sum_error=0.
            begin=time.time()#take time before calculating the epoche

            tmp_list_inp_out = list(zip(inputs, outputs))
            random.shuffle(tmp_list_inp_out)

            inputs, outputs = zip(*tmp_list_inp_out)

            for j in range(count_batches):
                input_=inputs[j*batchsize:(j+1)*batchsize]
                output_=outputs[j*batchsize:(j+1)*batchsize]

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

        #1st: initialize all variables (includes optimizer variables)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #2nd: set weights from pretrained model
        #sess=tf.Session()
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

    def save(self,path):
        if(self.saver is None):
            if(self.version=="singlebatch"):
                with vs.variable_scope("network",reuse=True):
                    self.saver=tf.train.Saver(var_list={"LSTM/w_f_diag": vs.get_variable("LSTM/w_f_diag"),
                                                        "LSTM/w_i_diag": vs.get_variable("LSTM/w_i_diag"),
                                                        "LSTM/w_o_diag": vs.get_variable("LSTM/w_o_diag"),
                                                        "LSTM/biases": vs.get_variable("LSTM/biases"),
                                                        "LSTM/weights": vs.get_variable("LSTM/weights"),
                                                        "OUTPUT/weights_output":vs.get_variable("OUTPUT/weights_output"),
                                                        "OUTPUT/biases_output":vs.get_variable("OUTPUT/biases_output")})
            elif(self.version=="multibatch"):
                with vs.variable_scope("network",reuse=True):
                    self.saver=tf.train.Saver(var_list={"LSTM/w_f_diag": vs.get_variable("LSTM/lstm_cell/w_f_diag"),
                                                        "LSTM/w_i_diag": vs.get_variable("LSTM/lstm_cell/w_i_diag"),
                                                        "LSTM/w_o_diag": vs.get_variable("LSTM/lstm_cell/w_o_diag"),
                                                        "LSTM/biases": vs.get_variable("LSTM/lstm_cell/biases"),
                                                        "LSTM/weights": vs.get_variable("LSTM/lstm_cell/weights"),
                                                        "OUTPUT/weights_output":vs.get_variable("OUTPUT/weights_output"),
                                                        "OUTPUT/biases_output":vs.get_variable("OUTPUT/biases_output")})

        self.saver.save(self.sess, path)

    #runs several inputs and returns outputs
    def __call__(self,inputs):
        prediction=self.sess.run(self.outputs,feed_dict={self.data:[inputs]})
        return prediction[0]


if __name__=='__main__':
    COUNT_EPOCHS=4
    COUNT_TIMESTEPS=20
    NUM_TRAINING=100

    rocketBall= RocketBall.standardVersion()

    inputs=[SequenceGenerator.generateCustomInputs_4tuple(rocketBall,COUNT_TIMESTEPS,changingProbability=0.3) for i in range(NUM_TRAINING)]
    outputs=[SequenceGenerator.runInputs_2tuple_relative(rocketBall,input,dt=1./30.) for input in inputs]


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":len(outputs[0][0]),
        "size_input":len(inputs[0][0]),
        "use_biases":True,
        "use_peepholes":True,
        "tag":"test"
    }

    fmodel=forwardModel(configuration)
    fmodel.create_dynamicRNN(COUNT_TIMESTEPS)
    fmodel.init()

    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    fmodel.train(inputs, outputs,batchsize=2,count_epochs=COUNT_EPOCHS,logging=True,save=True)
