import time

import os
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from src.RocketBall import RocketBall
from src.models.forwardModel import forwardModel
from src.models.helper import *
from src.models.inverseModel1 import inverseModel
from src.LabyrinthGrid import LabyrinthGrid
from src.DataGenerators.GridLabyrinthSequenceGenerator import GridLabyrinthSequenceGenerator

class inverseModelGridLabyrinth(inverseModel):

    def __init__(self,configuration,learning_rate=0.03):
        inverseModel.__init__(self,configuration)
        forwardModel.__init__(self,configuration)
        self.epsilon=10**(-8)
        self.learning_rate=learning_rate
        self.beta1=0.9
        self.beta2=0.9

        self.obstacle_information=None


    def create_self_feeding(self,count_timesteps):

        self.count_timesteps=count_timesteps
        self.inputs=tf.Variable(tf.random_uniform([1,count_timesteps,4],minval=0.,maxval=1.))
        self.target=tf.placeholder(tf.float32,[1,count_timesteps,self.configuration["size_output"]])

        self.speed = tf.placeholder(tf.float32,[1,self.configuration["size_output"]])
        self.obstacle_information=tf.placeholder(tf.float32,[1,self.configuration["size_grid"]])
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
            _obstacle_information=self.obstacle_information
            with vs.variable_scope("LSTM") as lstm_scope:

                jointInput=tf.concat([inputData[0],_speed,_obstacle_information],1)
                lstm_output,state=cell(jointInput,self.last_state,lstm_scope) #use last_state
                _speed=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs

                #lstm_outputs.append(lstm_output)
                final_outputs.append(_speed)
                (self.next_c,self.next_h)=state

                #TODO when training labyrinth with obstacle information,append the array of obstacles here
                self.next_speed=_speed

            with vs.variable_scope("LSTM",reuse=True) as lstm_scope:
                for index,inp in enumerate(inputData[1:]):
                    jointInput=tf.concat([inp,_speed,self.obstacle_information],1)
                    lstm_output,state=cell(jointInput,state,lstm_scope)
                    _speed=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs
                    final_outputs.append(_speed)

            self.outputs=tf.transpose(final_outputs,[1,0,2],"Transpose_output")


    def infer_self_feeding(self,target_outputs,count_iterations,obstacle_information):

        #convert all inputs to one_hot
        convert_op=self.inputs.assign(tf.one_hot(tf.argmax(self.inputs,2),4))
        self.sess.run(convert_op)
        for i in range(count_iterations):
            print("targets",target_outputs)
            print("speed",self.last_speed)


            result_c,result_h,result_speed,i,o,r,_,_=self.sess.run([self.next_c,self.next_h,self.next_speed,self.inputs,self.outputs,self.rmse,self.minimizer,self.clipping],
                                                                   feed_dict={self.target:[target_outputs],
                                                                              self.c_state:self.last_c,
                                                                              self.h_state:self.last_h,
                                                                              self.speed:self.last_speed,
                                                                              self.obstacle_information:[obstacle_information]})

            print("result_speed",result_speed)
            print("next_inputs",i)
            print("next_outputs",o)

        self.last_c=result_c
        self.last_h=result_h
        self.last_speed=result_speed

        return i[0]

    def reset(self):
        init_input_var = tf.initialize_variables([self.inputs])
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_input_var)
        inverseModel.reset(self)


if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=100
    COUNT_OBSTACLES=30
    COUNT_ITERATIONS=30

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 32,
        "size_output":2,
        "size_grid": 100,
        "size_input":6+100,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_50_100_30"
    }

    lab=LabyrinthGrid.standardVersion()
    seed=1
    lab.setRandomObstacles(COUNT_OBSTACLES,seed)

    obstacle_information=GridLabyrinthSequenceGenerator.obstacleInformation(lab)
    outputs=[0.0,0.0]
    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModelGridLabyrinth(configuration)
    iModel.create_self_feeding(COUNT_TIMESTEPS)
    iModel.create_last_timestep_optimizer(0.,1.)
    iModel.restore(path)

    begin=time.time()
    for i in range(20):
        #pass
        print(iModel.infer_self_feeding([outputs for i in range(COUNT_TIMESTEPS)],COUNT_ITERATIONS,obstacle_information)[0])
    end=time.time()
    print(end-begin)