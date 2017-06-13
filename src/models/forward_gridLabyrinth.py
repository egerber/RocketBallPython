from src.DataGenerators.GridLabyrinthSequenceGenerator import *
from src.LabyrinthGrid import *
from src.models.forwardModel import *
from src.helper.JsonHelper import JsonHelper

if __name__=="__main__":


    COUNT_TIMESTEPS=10
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=50000

    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=1
    LEARNING_RATE=0.01

    trainingDataPath="../../data/trainingData/GridLabyrinthSmall/various_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)

    trainingData=JsonHelper.restore(trainingDataPath)

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":31,
        "use_biases":True,
        "use_peepholes":True,
        "count_layers":2,
        "tag":"GridLabyrinthSmall("+str(LEARNING_RATE)+")_various_multi_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)
    }
    print(configuration["tag"])

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




    fmodel=forwardModel(configuration,LEARNING_RATE)

    fmodel.create_multiLayerRNN(2,COUNT_TIMESTEPS,device='/gpu:0')
    fmodel.init()
    #fmodel.restore(path)
    fmodel.train(trainingData["inputs"], trainingData["outputs"],batchsize=BATCH_SIZE,count_epochs=COUNT_EPOCHS,logging=True,save=True)