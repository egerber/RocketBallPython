from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *
from src.helper.JsonHelper import JsonHelper
if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=10
    COUNT_OBSTACLES=30
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=2000

    trainingDataPath="../../data/trainingData/GridLabyrinth/"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)
    trainingData=JsonHelper.restore(trainingDataPath)

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":106,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth(0.001)_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)
    }

    lab=LabyrinthGrid.standardVersion(COUNT_OBSTACLES,1)
    seeds=list(range(1,COUNT_OBSTALCE_CONFIGURATIONS+1))


    fmodel=forwardModel(configuration)

    fmodel.create_dynamicRNN(COUNT_TIMESTEPS,device='/cpu:0')
    fmodel.init()

    fmodel.train(trainingData["inputs"], trainingData["outputs"],batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)