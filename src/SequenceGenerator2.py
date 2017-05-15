import numpy as np
import time as time
from src.Vector2f import Vector2f
from src.helper.JsonHelper import JsonHelper

from src.models.inverseModel1 import *

class SequenceGenerator2:

    #generates inputs within some range, where input(t+1) can be only a slight change compared to input(t)
    @staticmethod
    def generateInputs_probOffset(countTimesteps,maxChange,maxSpeed,probabilityChange):
        inputs=np.random.uniform(low=-maxChange,high=maxChange,size=(countTimesteps,2))

        for i in range(1,countTimesteps):
            if(np.random.rand()<=probabilityChange):
                inputs[i][0]=inputs[i-1][0]+inputs[i][0]
                inputs[i][1]=inputs[i-1][1]+inputs[i][1]
            else:
                inputs[i][0]=inputs[i-1][0]
                inputs[i][1]=inputs[i-1][1]

        np.clip(inputs,-maxSpeed,maxSpeed,inputs)
        return inputs

    #TODO change so that various timesteps can be inferred
    @staticmethod
    def generateOutputs_relative(rocketBall,inverseModel,inputs,count_iterations,count_timesteps=1,dt=1./30.):
        outputs=np.empty((len(inputs),2))

        rocketBall.reset()
        inverseModel.reset()

        prevPosition=[rocketBall.position.x,rocketBall.position.y]
        for i in range(len(inputs)):
            motorInputs=inverseModel.infer([inputs[i] for i in range(count_timesteps)],count_iterations)
            rocketBall.setThrust1(motorInputs[0][0])
            rocketBall.setThrust2(motorInputs[0][1])

            rocketBall.update(dt)
            outputs[i][0]=rocketBall.position.x-prevPosition[0]
            outputs[i][1]=rocketBall.position.y-prevPosition[1]

            prevPosition[0]=rocketBall.position.x
            prevPosition[1]=rocketBall.position.y

        return outputs

    @staticmethod
    def generateOutputs_absolute(rocketBall,inverseModel,inputs,count_iterations,count_timesteps=1,dt=1./30.):
        outputs=np.empty((len(inputs),2))

        rocketBall.reset()
        inverseModel.reset()

        startPosition=[rocketBall.position.x,rocketBall.position.y]
        for i in range(len(inputs)):
            motorInputs=inverseModel.infer([inputs[i] for i in range(count_timesteps)],count_iterations)
            rocketBall.setThrust1(motorInputs[0][0])
            rocketBall.setThrust2(motorInputs[0][1])

            rocketBall.update(dt)
            outputs[i][0]=rocketBall.position.x-startPosition[0]
            outputs[i][1]=rocketBall.position.y-startPosition[1]



        return outputs
if __name__=="__main__":
    COUNT_ITERATIONS=30
    COUNT_TIMESTEPS=1
    COUNT_TIMESTEPS_INPUT=50
    COUNT_TRAINING=100

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
    inputs=[SequenceGenerator2.generateInputs_probOffset(COUNT_TIMESTEPS_INPUT,maxChange=0.01,maxSpeed=0.05,probabilityChange=0.4).tolist() for i in range(COUNT_TRAINING)]
    outputs=[SequenceGenerator2.generateOutputs_relative(rocketBall,iModel,input,COUNT_ITERATIONS).tolist() for input in inputs]

    trainingsDict={"inputs": inputs,"outputs": outputs}

    #configuration order trainingsitems, timesteps, iterations, inferedTimesteps, maxStepsize
    JsonHelper.save("../data/trainingData/training2_(relative,100,50,30,1,0.05).json",trainingsDict)


    end=time.time()
    print(end-begin)