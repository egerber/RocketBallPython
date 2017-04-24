
from src.models import forwardModel

class forwardPredictor(object):

    def __init__(self,configuration,count_timesteps,restorePath):
        self.forwardModel=forwardModel(self.configuration)
        self.forwardModel.create(count_timesteps)
        self.forwardModel.restore(restorePath)

    def __call__(self,inputs):

        prediction=self.sess.run(self.output,feed_dict={self.data:[inputs]})
        return prediction[0]

    #runs one timestep of the model and returns output
    def forwardPass(self,input):
        pass



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

    #restorePath="/home/emanuel/Coding/tensorflow/SessionData/"+createConfigurationString(configuration)+".chkpt"
    restorePath="/home/emanuel/Coding/tensorflow/SessionData(copy)/sess2_2_(10000).chkpt"
    predictor=forwardPredictor(configuration,COUNT_TIMESTEPS,restorePath)

    inputs=SequenceGenerator.generateCustomInputs_2tuple(COUNT_TIMESTEPS,changingProbability=0.3)
    print(predictor(inputs))
    print(predictor(inputs))