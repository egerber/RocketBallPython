from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *

if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=100
    COUNT_OBSTACLES=30
    BATCH_SIZE=1
    COUNT_TRAININGS_PER_CONFIGURATION=500

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 64,
        "size_output":2,
        "size_input":106,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    }

    lab=LabyrinthGrid.standardVersion(30,1)
    seeds=list(range(COUNT_OBSTALCE_CONFIGURATIONS))

    inputs=[GridLabyrinthSequenceGenerator.generateInputs_one_hot_obstacles(lab,COUNT_TIMESTEPS,COUNT_OBSTACLES,seed) for seed in seeds for i in range(int(COUNT_TRAININGS_PER_CONFIGURATION))]
    outputs=[GridLabyrinthSequenceGenerator.generateOutputs_one_hot(lab,input) for input in inputs]

    fmodel=forwardModel(configuration)

    fmodel.create_dynamicRNN(COUNT_TIMESTEPS,device='/cpu:0')
    fmodel.init()

    fmodel.train(inputs, outputs,batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)