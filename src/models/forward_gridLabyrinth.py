from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *

if __name__=="__main__":

    COUNT_TRAININGS=2000
    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":6+25,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_50_one_hot"
    }

    lab=LabyrinthGrid.standardVersion()
    inputs=[GridLabyrinthSequenceGenerator.generateInputs_one_hot_obstacles(lab,COUNT_TIMESTEPS) for i in range(COUNT_TRAININGS)]
    outputs=[GridLabyrinthSequenceGenerator.generateOutputs_one_hot(lab,input) for input in inputs]


    fmodel=forwardModel.createNew(configuration,COUNT_TIMESTEPS)
    fmodel.train(inputs,outputs,count_epochs=COUNT_EPOCHS,logging=True,save=True)
