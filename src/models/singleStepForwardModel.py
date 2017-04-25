from models.forwardModel import *


class singleStepForwardModel(forwardModel):
    def __init__(self,configuration):
        forwardModel.__init__(self,configuration)

        self.configuration=configuration

        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

    def reset(self):
        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

    #runs one one timestep (given as input) and returns result
    #saves last state of the lstm and passes uses it in the next timestep
    def __call__(self,input):
        prediction,self.last_c,self.last_h=self.sess.run([self.output,self.result_c,self.result_h],feed_dict={self.data:[input],self.c_state:self.last_c, self.h_state:self.last_h})
        return prediction

    @staticmethod
    def createFromOld(configuration,file_path,device='/cpu:0'):
        tmodel=singleStepForwardModel(configuration)
        tmodel.create()
        tmodel.restore(file_path)

        return tmodel
    def create(self):

        self.data=tf.placeholder(tf.float32,[1,self.configuration["size_input"]])
        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])

        self.last_state=tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)
        with vs.variable_scope("network"):
            if(self.configuration["cell_type"]=="LSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_units=self.configuration["num_hidden_units"],use_peepholes=self.configuration["use_peepholes"],state_is_tuple=True)
            elif(self.configuration["cell_type"]=="GRUCell"):
                cell=tf.contrib.rnn.GRUCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="RNNCell"):
                cell=tf.contrib.rnn.RNNCell(num_units=self.configuration["num_hidden_units"])
            elif(self.configuration["cell_type"]=="BasicLSTMCell"):
                cell=tf.contrib.rnn.LSTMCell(num_hidden_units=self.configuration["num_hidden_units"],state_is_tuple=True)



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


if __name__=="__main__":

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":6,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative"
    }

    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"
    tmodel=singleStepForwardModel.createFromOld(configuration,path)
