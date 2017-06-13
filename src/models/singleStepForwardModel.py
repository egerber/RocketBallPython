from src.models.forwardModel import *


class singleStepForwardModel(forwardModel):
    def __init__(self,configuration):
        forwardModel.__init__(self,configuration)

        self.configuration=configuration
        if(self.configuration["count_layers"]>1):
            self.last_c=np.zeros((self.configuration["count_layers"],1,self.configuration["num_hidden_units"]))
            self.last_h=np.zeros((self.configuration["count_layers"],1,self.configuration["num_hidden_units"]))
        else:
            self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
            self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

    def reset(self):
        if(self.configuration["count_layers"]>1):
            self.last_c=np.zeros((2,1,self.configuration["num_hidden_units"]))
            self.last_h=np.zeros((2,1,self.configuration["num_hidden_units"]))
        else:
            self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
            self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

    #runs one one timestep (given as input) and returns result
    #saves last state of the lstm and passes uses it in the next timestep
    def __call__(self,input):
        print("c_state",self.last_c)
        print("h_state",self.last_h)

        prediction,self.last_c,self.last_h=self.sess.run([self.output,self.result_c,self.result_h],feed_dict={self.data:[input],self.c_state:self.last_c, self.h_state:self.last_h})
        return prediction

    @staticmethod
    def createFromOld(configuration,file_path,device='/cpu:0'):
        tmodel=singleStepForwardModel(configuration)

        if(configuration["count_layers"]>1):
            tmodel.create_multilayer(configuration["count_layers"])
        else:
            tmodel.create()


        tmodel.restore(file_path)

        return tmodel

    def create(self):

        self.data=tf.placeholder(tf.float32,[1,self.configuration["size_input"]])
        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])

        self.last_state=tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)
        with vs.variable_scope("network"):
            cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)

            #initial_state=cell.zero_state(1,tf.float32)
            with vs.variable_scope("LSTM") as lstm_scope:
                lstm_output,(self.result_c,self.result_h)=cell(self.data,self.last_state,lstm_scope)

            with vs.variable_scope("OUTPUT"):
                output_layer=vs.get_variable("weights_output",
                                             [self.configuration["num_hidden_units"],self.configuration["size_output"]],
                                             initializer=tf.random_normal_initializer())
                if(self.configuration["use_biases"]):
                    biases_outputs=vs.get_variable("biases_output",[1,self.configuration["size_output"]],
                                                   initializer=tf.random_normal_initializer())
                    l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs
                else:
                    l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")

        self.output=l_output

    def create_multilayer(self,count_layers=2,device='/cpu:0'):
        self.version="multilayerbatch"
        with tf.device(device):
            self.data=tf.placeholder(tf.float32,[1,self.configuration["size_input"]])
            self.c_state = tf.placeholder(tf.float32,[count_layers,1,self.configuration["num_hidden_units"]])
            self.h_state = tf.placeholder(tf.float32,[count_layers,1,self.configuration["num_hidden_units"]])

            self.last_state=tf.contrib.rnn.LSTMStateTuple(tf.unstack(self.c_state),tf.unstack(self.h_state))

            with vs.variable_scope("network"):
                with vs.variable_scope("LSTM/multi_rnn_cell") as lstm_scope:

                    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=True,state_is_tuple=True) for _ in range(count_layers)],
                                                               state_is_tuple=True)

                    lstm_output,(self.result_c,self.result_h)=stacked_lstm(self.data,self.last_state,lstm_scope)

                with vs.variable_scope("OUTPUT"):
                    output_layer=vs.get_variable("weights_output",
                                                 [self.configuration["num_hidden_units"],self.configuration["size_output"]],
                                                 initializer=tf.random_normal_initializer())
                    if(self.configuration["use_biases"]):
                        biases_outputs=vs.get_variable("biases_output",[1,self.configuration["size_output"]],
                                                       initializer=tf.random_normal_initializer())
                        l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")+biases_outputs
                    else:
                        l_output=tf.matmul(lstm_output,output_layer,name="Multiply_lstm_output")

        self.output=l_output


if __name__=="__main__":

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }

    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"
    #tmodel=singleStepForwardModel.createFromOld(configuration,path)
    #print(tmodel([0.2,0.3]))
    #print(tmodel([0.1,0.2]))

    fmodel=forwardModel.createFromOld(configuration,2,path)
    print(fmodel([[0.2,0.3],[0.1,0.2]]))

