from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *

if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=1000
    COUNT_OBSTACLES=30
    BATCH_SIZE=128
    COUNT_TRAININGS=COUNT_OBSTALCE_CONFIGURATIONS*(100-COUNT_OBSTACLES)*4

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 32,
        "size_output":2,
        "size_input":6+100,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_50_1000_30"
    }

    lab=LabyrinthGrid.standardVersion()
    seeds=list(range(COUNT_OBSTALCE_CONFIGURATIONS))

    inputs=[GridLabyrinthSequenceGenerator.generateInputs_one_hot_obstacles(lab,COUNT_TIMESTEPS,COUNT_OBSTACLES,seed) for seed in seeds for i in range(int(COUNT_TRAININGS/COUNT_OBSTALCE_CONFIGURATIONS))]
    outputs=[GridLabyrinthSequenceGenerator.generateOutputs_one_hot(lab,input) for input in inputs]

    fmodel=forwardModel(configuration)

    fmodel.create_dynamicRNN(COUNT_TIMESTEPS,device='/cpu:0')
    fmodel.init()

    fmodel.train(inputs, outputs,batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)
