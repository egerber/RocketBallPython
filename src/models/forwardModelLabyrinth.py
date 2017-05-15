from src.models.forwardModel import *
from src.models.inverseModel1 import inverseModel
from src.SequenceGenerator2 import SequenceGenerator2
from src.helper.JsonHelper import JsonHelper
from src.LabyrinthSequenceGenerator import *

if __name__=="__main__":

    COUNT_TRAININGS=2000
    COUNT_TIMESTEPS=200
    COUNT_EPOCHS=31
    MAX_STEPSIZE=0.2

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":4,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"Labyrinth_200_0.2_custom(0.3)"
    }

    lab=Labyrinth.standardVersion()
    inputs=[LabyrinthSequenceGenerator.generateCustomInputs_4tuple(lab,COUNT_TIMESTEPS,MAX_STEPSIZE,0.3) for i in range(COUNT_TRAININGS)]
    outputs=[LabyrinthSequenceGenerator.generateOutputs_2tuple(lab,input) for input in inputs]

    fmodel=forwardModel.createNew(configuration,COUNT_TIMESTEPS)
    fmodel.train(inputs,outputs,count_epochs=COUNT_EPOCHS,logging=True,save=True)
