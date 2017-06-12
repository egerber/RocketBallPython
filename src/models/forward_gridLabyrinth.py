from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *
from src.helper.JsonHelper import JsonHelper

if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=1
    COUNT_OBSTACLES=9
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=10000
    LEARNING_RATE=0.01

    #trainingDataPath="../../data/trainingData/GridLabyrinthSmall/"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)
    trainingDataPath="../../data/trainingData/GridLabyrinthSmall/single_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)

    trainingData=JsonHelper.restore(trainingDataPath)

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":31,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinthSmall("+str(LEARNING_RATE)+")_single_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)
    }

    #curriculum_configuration={
    #    "cell_type":"LSTMCell",
    #    "num_hidden_units": 16,
    #    "size_output":2,
    #    "size_input":31,
    #    "use_biases":True,
    #    "use_peepholes":True,
   #     "tag":"GridLabyrinth("+str(LEARNING_RATE)+")_PRETRAINED_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    #}

    #path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"


    lab=LabyrinthGrid.smallVersion(COUNT_OBSTACLES,1)


    fmodel=forwardModel(configuration,LEARNING_RATE)

    fmodel.create_dynamicRNN(COUNT_TIMESTEPS,device='/cpu:0')
    fmodel.init()
    #fmodel.restore(path)
    fmodel.train(trainingData["inputs"], trainingData["outputs"],batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)