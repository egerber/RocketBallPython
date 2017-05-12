import numpy as np
import time as time
from src.Vector2f import Vector2f
from src.helper.JsonHelper import JsonHelper

from src.models.inverseModel1 import *

class SequenceGenerator2:

    #generates inputs within some range, where input(t+1) can be only a slight change compared to input(t)
    @staticmethod
    def generateInputs_probOffset(countTimesteps,maxChange,probabilityChange):
        inputs=np.random.uniform(low=-maxChange,high=maxChange,size=(countTimesteps,2))

        for i in range(1,countTimesteps):
            if(np.random.rand()<=probabilityChange):
                inputs[i][0]=inputs[i-1][0]+inputs[i][0]
                inputs[i][1]=inputs[i-1][1]+inputs[i][1]
            else:
                inputs[i][0]=inputs[i-1][0]
                inputs[i][1]=inputs[i-1][1]

        return inputs

    #TODO change so that various timesteps can be inferred
    @staticmethod
    def generateOutputs(rocketBall,inverseModel,inputs,count_iterations,count_timesteps=1,dt=1./30.):
        outputs=np.empty((len(inputs),2))

        rocketBall.reset()
        inverseModel.reset()

        prevPosition=Vector2f(rocketBall.position.x,rocketBall.position.y)
        for i in range(len(inputs)):
            motorInputs=inverseModel.infer([inputs[i] for i in range(count_timesteps)],count_iterations)
            rocketBall.setThrust1(motorInputs[0][0])
            rocketBall.setThrust2(motorInputs[0][1])
            rocketBall.update(dt)
            outputs[i][0]=rocketBall.position.x-prevPosition.x
            outputs[i][1]=rocketBall.position.y-prevPosition.y

            prevPosition.x=rocketBall.position.x
            prevPosition.y=rocketBall.position.x

        return outputs

if __name__=="__main__":
    COUNT_ITERATIONS=30
    COUNT_TIMESTEPS=1
    COUNT_TIMESTEPS_INPUT=50
    COUNT_TRAINING=500

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }


    path=checkpointDirectory=os.path.dirname(__file__)+"/../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)

    iModel.create(COUNT_TIMESTEPS)
    iModel.create_all_timesteps_optimizer()
    iModel.restore(path)

    rocketBall=rocketBall.standardVersion()

    begin=time.time()
    inputs=[SequenceGenerator2.generateInputs_probOffset(COUNT_TIMESTEPS_INPUT,0.1,0.4).tolist() for i in range(COUNT_TRAINING)]
    outputs=[SequenceGenerator2.generateOutputs(rocketBall,iModel,input,COUNT_ITERATIONS,COUNT_TIMESTEPS).tolist() for input in inputs]

    trainingsDict={"inputs": inputs,"outputs": outputs}

    JsonHelper.save("../data/trainingData/training2_(500,50,30,1,0.1).json",trainingsDict)

    end=time.time()
    print(end-begin)