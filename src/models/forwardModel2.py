from src.helper.JsonHelper import JsonHelper
from src.models.forwardModel import *
from src.models.inverseModel1 import inverseModel

if __name__=="__main__":

    COUNT_TIMESTEPS_INVERSEMODEL=1
    COUNT_TIMESTEPS_INPUT=50

    COUNT_EPOCHS=31

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    configuration_restore={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }

    configuration_new={
        "cell_type":"LSTMCell",
        "num_hidden_units": 128,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"forwardModel2(absolute4_2,500,50,30,0.009,0.06)"
    }


    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration_restore)+".chkpt"

    iModel=inverseModel(configuration_restore)

    iModel.create(COUNT_TIMESTEPS_INVERSEMODEL)
    iModel.create_all_timesteps_optimizer()
    iModel.restore(path)

    rocketBall=rocketBall.standardVersion()

    trainingDict=JsonHelper.restore("../../data/trainingData/training2_(absolute4_2,500,50,30,0.009,0.06).json")
    inputs=trainingDict["inputs"]
    outputs=trainingDict["outputs"]

    fmodel=forwardModel.createNew(configuration_new,COUNT_TIMESTEPS_INPUT)
    fmodel.train(inputs,outputs,count_epochs=COUNT_EPOCHS,logging=True,save=True)

    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration_new)+".chkpt"

