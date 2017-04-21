from models.forwardModell1 import *

class singleStepPredictor(object):

    def __init__(self,configuration,restorePath):
        self.configuration=configuration
        self.input=tf.placeholder(tf.float32,[1,self.configuration["size_input"]])
        self.c_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.h_state = tf.placeholder(tf.float32,[1,self.configuration["num_hidden_units"]])
        self.state= tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state)

        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

        self.output,(self.result_c,self.result_h)=createTimestepModel(self.input,self.state,self.configuration)
        self.sess=restoreNetwork(restorePath)

    def reset(self):
        self.last_c=np.zeros((1,self.configuration["num_hidden_units"]))
        self.last_h=np.zeros((1,self.configuration["num_hidden_units"]))

    #runs one one timestep (given as input) and returns result
    #saves last state of the lstm and passes uses it in the next timestep
    def __call__(self,input):
        prediction,self.last_c,self.last_h=self.sess.run([self.output,self.result_c,self.result_h],feed_dict={self.input:[input],self.c_state:self.last_c, self.h_state:self.last_h})
        return prediction


if __name__=="__main__":
    NUM_TRAINING=500
    COUNT_TIMESTEPS=10
    COUNT_EPOCHS=10
    BATCH_SIZE=1

    NUM_HIDDEN_UNITS=16
    LENGTH_INPUT=2
    LENGTH_OUTPUT=2


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_input":2,
        "size_output":2,
        "use_biases":True,
        "use_peepholes":True,
        "count_epochs":30
    }

    restorePath="/home/emanuel/Coding/tensorflow/SessionData(copy)/sess2_2_(10000).chkpt"
    predictor=singleStepPredictor(configuration,restorePath)

    inputs=SequenceGenerator.generateCustomInputs_2tuple(COUNT_TIMESTEPS,changingProbability=0.3)

    print(predictor(inputs[0]))
