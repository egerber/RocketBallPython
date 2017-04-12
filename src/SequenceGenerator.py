import numpy as np
from RocketBall import *
import copy

class SequenceGenerator:

    @staticmethod
    def generateUniformInputs(timesteps):
        return np.random.rand(timesteps, 2)

    @staticmethod
    def generateNormalInputs(timesteps, mean, std):
        return np.clip(np.random.normal(loc=mean, scale=std, size=(timesteps, 2)), a_min=0., a_max=1.)

    @staticmethod
    def generateCustomInputs(timesteps, changingProbability):
        inputs=np.random.rand(timesteps,2)
        for i in range(1,timesteps):
            #do not change input for thrust1
            if(np.random.rand()>changingProbability):
                inputs[i][0]=inputs[i-1][0]
            #do not change input for thrust2
            if(np.random.rand()>changingProbability):
                inputs[i][1]=inputs[i-1][1]

        return inputs

    @staticmethod
    def runInput(self):
        pass
    @staticmethod
    def generateCustomInputs_4tuple(rocketBall, timesteps, changingProbability, dt=1. / 30.):
        rocketBall.reset()


        #choose random initial Position on the field (considers radius of rocketBall)
        initialPosition=Vector2f(np.random.uniform(low=rocketBall.getLeftBorder()+rocketBall.getRadius(),
                                                high=rocketBall.getRightBorder()-rocketBall.getRadius()),
                              np.random.uniform(low=0+rocketBall.getRadius(),
                                                high=rocketBall.getTopBorder()-rocketBall.getRadius()))

        inputs=np.random.rand(timesteps,4) # indices 0,1 are inputs, indices 2,3 are x and y position
        for i in range(1,timesteps):
            inputs[i][2:4]=[0.,0.]
        for i in range(1,timesteps):
            #do not change input for thrust1
            if(np.random.rand()>changingProbability):
                inputs[i][0]=inputs[i-1][0]
            #do not change input for thrust2
            if(np.random.rand()>changingProbability):
                inputs[i][1]=inputs[i-1][1]

        #set intial Position as First
        inputs[0][2]=initialPosition.x
        inputs[0][3]=initialPosition.y

        for i in range(0, timesteps-1):
            rocketBall.setThrust1(inputs[i][0])
            rocketBall.setThrust2(inputs[i][1])
            #run Input
            rocketBall.update(dt)
            #get Position
            nextPosition=rocketBall.getPosition()
            #add Position to next Input
            inputs[i+1][2]=nextPosition.x
            inputs[i+1][3]=nextPosition.y


        return inputs

    @staticmethod
    def runInputs_2tuple(rocketBall, inputs, dt=1. / 30.):
        rocketBall.reset()


        outputs=np.empty((len(inputs),2))

        if(len(inputs[0])==4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))

        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            outputs[index][0]=nextPosition.x
            outputs[index][1]=nextPosition.y
        return outputs

    @staticmethod
    def runInputs_6tuple(rocketBall,inputs,dt):
        rocketBall.reset()
        if(len(inputs[0])==4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))

        outputs=np.empty((len(inputs),6))
        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            nextVelocity=rocketBall.velocity
            nextAcceleration=rocketBall.acceleration

            outputs[index][0]=nextPosition.x
            outputs[index][1]=nextPosition.y
            outputs[index][2]=nextVelocity.x
            outputs[index][3]=nextVelocity.y
            outputs[index][4]=nextAcceleration.x
            outputs[index][5]=nextAcceleration.y

        return outputs

    @staticmethod
    def test4tuple(rocketBall,inputs,outputs,dt):
        rocketBall.reset()

        rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        for input,output in zip(inputs,outputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])
            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            if(not(nextPosition.x==output[0]) or not (nextPosition.y==output[1])):
                print("Error in input/output pair")



    @staticmethod
    def test2tuple(rocketBall,inputs,outputs,dt):
        rocketBall.reset()

        for input,output in zip(inputs,outputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])
            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            if(not(nextPosition.x==output[0]) or not(nextPosition.y==output[1])):
                print("Error in input/output pair")




if __name__=="__main__":
    rocketBall=RocketBall.standardVersion()
    inputs=SequenceGenerator.generateCustomInputs_tuple(rocketBall,timesteps=50, changingProbability=0.3,dt=1./30.)

    #rocketBall=rocketBall.standardVersion()
    outputs=SequenceGenerator.runInputs_6tuple(rocketBall,inputs=inputs,dt=1./30.)


    #rocketBall=rocketBall.standardVersion()
    SequenceGenerator.test4tuple(rocketBall,inputs=inputs,outputs=outputs,dt=1./30.)
