from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *
from src.helper.JsonHelper import JsonHelper

if __name__=="__main__":


    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=1
    COUNT_OBSTACLES=0
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=10000

    trainingDataPath="../../data/trainingData/GridLabyrinth/random"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)
    trainingData=JsonHelper.restore(trainingDataPath)

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":106,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth(0.001)_random_"+str(50)+"_"+str(10000)+"_"+str(0)+"_"+str(1)+"_"+str(32)
    }

    #curriculum_configuration={
    #    "cell_type":"LSTMCell",
    #    "num_hidden_units": 16,
    #   "size_output":2,
    #    "size_input":106,
    #    "use_biases":True,
   #     "use_peepholes":True,
    #    "tag":"GridLabyrinth(0.0001)_PRETRAINED_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    #}

    #path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(pretrained_conf)+".chkpt"


    lab=LabyrinthGrid.standardVersion(COUNT_OBSTACLES,1)
    seeds=list(range(1,COUNT_OBSTALCE_CONFIGURATIONS+1))


    fmodel=forwardModel(configuration)

    fmodel.create_dynamicRNN(COUNT_TIMESTEPS,device='/gpu:0')
    fmodel.init()
    #fmodel.restore(path)
    fmodel.train(trainingData["inputs"], trainingData["outputs"],batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)