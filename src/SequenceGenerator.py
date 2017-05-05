import numpy as np
from src.RocketBall import *
import copy

class SequenceGenerator:

    @staticmethod
    def generateUniformInputs(timesteps):
        return np.random.rand(timesteps, 2)

    @staticmethod
    def generateNormalInputs(timesteps, mean, std):
        return np.clip(np.random.normal(loc=mean, scale=std, size=(timesteps, 2)), a_min=0., a_max=1.)

    @staticmethod
    def generateCustomInputs_2tuple(timesteps, changingProbability,gaussian=False,mean=0.5,std=0.1):

        if(gaussian):
            inputs=np.random.normal(loc=mean,scale=std,size=(timesteps,2))
        else:
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
    def generateSequenceInputs(rocketBall,timesteps,changingProbability,dt=1./30.):
        motorInputs=SequenceGenerator.generateCustomInputs_2tuple(timesteps,changingProbability)
        relative_outputs=SequenceGenerator.runInputs_2tuple_relative(rocketBall,motorInputs,dt)
        return relative_outputs

    @staticmethod
    def generateSequenceOutputs(sequenceInputs):
        outputs=np.empty(np.shape(sequenceInputs))

        outputs[0][0]=sequenceInputs[0][0]
        outputs[0][1]=sequenceInputs[0][1]
        for i in range(1,len(sequenceInputs)):
            outputs[i][0]=sequenceInputs[i][0]+outputs[i-1][0]
            outputs[i][1]=sequenceInputs[i][1]+outputs[i-1][1]

        return outputs

    @staticmethod
    def generateCustomInputs_4tuple(rocketBall, timesteps, changingProbability, dt=1. / 30.,gaussian=False):
        rocketBall.reset()

        #choose random initial Position on the field (considers radius of rocketBall)
        initialPosition=Vector2f(np.random.uniform(low=rocketBall.getLeftBorder()+rocketBall.getRadius(),
                                                high=rocketBall.getRightBorder()-rocketBall.getRadius()),
                              np.random.uniform(low=0+rocketBall.getRadius(),
                                                high=rocketBall.getTopBorder()-rocketBall.getRadius()))

        if gaussian:
            inputs=np.random.normal(loc=0.6,scale=0.2,size=(timesteps,4))
        else:
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

    #0=thrust1
    #1=thrust2
    #2=delta_x
    #3=delta_y
    @staticmethod
    def generateCustomInputs_4tuple_relative(rocketBall, timesteps, changingProbability, dt=1. / 30.):
        rocketBall.reset()
        rocketBall.placeDefault()

        inputs=np.random.rand(timesteps,4) # indices 0,1 are inputs, indices 2,3 are x and y position

        #set initial_speed to zero
        inputs[0][2:4]=[0.,0.]

        for i in range(1,timesteps):
            #do not change input for thrust1
            if(np.random.rand()>changingProbability):
                inputs[i][0]=inputs[i-1][0]
            #do not change input for thrust2
            if(np.random.rand()>changingProbability):
                inputs[i][1]=inputs[i-1][1]

        prevPosition=Vector2f(0.,0.)
        prevPosition.x=rocketBall.position.x
        prevPosition.y=rocketBall.position.y
        for i in range(0, timesteps-1):

            rocketBall.setThrust1(inputs[i][0])
            rocketBall.setThrust2(inputs[i][1])
            #run Input
            rocketBall.update(dt)
            #get Position
            nextPosition=rocketBall.getPosition()
            #add Position to next Input
            inputs[i+1][2]=nextPosition.x-prevPosition.x
            inputs[i+1][3]=nextPosition.y-prevPosition.y
            prevPosition.x=nextPosition.x
            prevPosition.y=nextPosition.y




        return inputs

    def generateCustomInputs_8tuple(rocketBall, timesteps, changingProbability, dt=1. / 30.):
        rocketBall.reset()


        #choose random initial Position on the field (considers radius of rocketBall)
        initialPosition=Vector2f(np.random.uniform(low=rocketBall.getLeftBorder()+rocketBall.getRadius(),
                                                   high=rocketBall.getRightBorder()-rocketBall.getRadius()),
                                 np.random.uniform(low=0+rocketBall.getRadius(),
                                                   high=rocketBall.getTopBorder()-rocketBall.getRadius()))

        inputs=np.random.rand(timesteps,8) # indices 0,1 are inputs, indices 2,3 are x and y position
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
        inputs[0][4]=rocketBall.velocity.x
        inputs[0][5]=rocketBall.velocity.y
        inputs[0][6]=rocketBall.acceleration.x
        inputs[0][7]=rocketBall.acceleration.y


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
            inputs[i+1][4]=rocketBall.velocity.x
            inputs[i+1][5]=rocketBall.velocity.y
            inputs[i+1][6]=rocketBall.acceleration.x
            inputs[i+1][7]=rocketBall.acceleration.y



        return inputs
    @staticmethod
    def runInputs_2tuple(rocketBall, inputs, dt=1. / 30.):
        rocketBall.reset()

        outputs=np.empty((len(inputs),2))

        if(len(inputs[0])==4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        else:
            rocketBall.placeDefault()

        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            outputs[index][0]=nextPosition.x
            outputs[index][1]=nextPosition.y
        return outputs

    @staticmethod
    def runInputs_2tuple_relative(rocketBall,inputs,dt=1./30.):
        rocketBall.reset()

        outputs=np.empty((len(inputs),2))

        if(len(inputs[0])==4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        else:
            rocketBall.placeDefault()


        prevPosition=Vector2f(0.,0.)
        prevPosition.x=rocketBall.position.x
        prevPosition.y=rocketBall.position.y

        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            outputs[index][0]=nextPosition.x-prevPosition.x
            outputs[index][1]=nextPosition.y-prevPosition.y

            #assignin new x and y coordinates (Warning: assigning prevPosition=nextPosition causes mistakes)
            prevPosition.x=nextPosition.x
            prevPosition.y=nextPosition.y

        return outputs

    @staticmethod
    def runInputs_4tuple_relative(rocketBall,inputs,dt=1./30.):
        rocketBall.reset()

        outputs=np.empty((len(inputs),4))


        rocketBall.placeDefault()


        prevPosition=Vector2f(0.,0.)
        prevPosition.x=rocketBall.position.x
        prevPosition.y=rocketBall.position.y
        prevDelta=Vector2f(0.,0.)
        prevDelta.x=0.
        prevDelta.y=0.

        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()

            deltax=nextPosition.x-prevPosition.x
            deltay=nextPosition.y-prevPosition.y
            outputs[index][0]=deltax
            outputs[index][1]=deltay
            outputs[index][2]=deltax-prevDelta.x
            outputs[index][3]=deltay-prevDelta.y

            #assignin new x and y coordinates (Warning: assigning prevPosition=nextPosition causes mistakes)
            prevPosition.x=nextPosition.x
            prevPosition.y=nextPosition.y
            prevDelta.x=deltax
            prevDelta.y=deltay

        return outputs

    @staticmethod
    def runInputs_2tuple_delta(rocketBall,inputs,dt):
        rocketBall.reset()

        outputs=np.empty((len(inputs),2))

        if(len(inputs[0])==4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        else:
            rocketBall.placeDefault()



        startPosition=Vector2f(0.,0.)
        startPosition.x=rocketBall.position.x
        startPosition.y=rocketBall.position.y

        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            outputs[index][0]=nextPosition.x-startPosition.x
            outputs[index][1]=nextPosition.y-startPosition.y



        return outputs

    @staticmethod
    def runInputs_6tuple(rocketBall,inputs,dt):
        rocketBall.reset()
        if(len(inputs[0])>=4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        else:
            rocketBall.placeDefault()

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

    def runInputs_6tuple_relative(rocketBall,inputs,dt):
        rocketBall.reset()
        if(len(inputs[0])>=4):
            rocketBall.setPosition(Vector2f(inputs[0][2],inputs[0][3]))
        else:
            rocketBall.placeDefault()

        prevPosition=Vector2f(0.,0.)
        prevAcceleration=Vector2f(0.,0.)
        prevVelocity=Vector2f(0.,0.)

        prevPosition.x=rocketBall.position.x
        prevPosition.y=rocketBall.position.y
        prevAcceleration.x=rocketBall.acceleration.x
        prevAcceleration.y=rocketBall.acceleration.y
        prevVelocity.x=rocketBall.velocity.x
        prevVelocity.y=rocketBall.velocity.y

        outputs=np.empty((len(inputs),6))
        for index,input in enumerate(inputs):
            rocketBall.setThrust1(input[0])
            rocketBall.setThrust2(input[1])

            rocketBall.update(dt)
            nextPosition=rocketBall.getPosition()
            nextVelocity=rocketBall.velocity
            nextAcceleration=rocketBall.acceleration

            outputs[index][0]=nextPosition.x-prevPosition.x
            outputs[index][1]=nextPosition.y-prevPosition.y
            outputs[index][2]=nextVelocity.x-prevVelocity.x
            outputs[index][3]=nextVelocity.y-prevVelocity.y
            outputs[index][4]=nextAcceleration.x-prevAcceleration.x
            outputs[index][5]=nextAcceleration.y-prevAcceleration.y

            prevPosition.x=rocketBall.position.x
            prevPosition.y=rocketBall.position.y
            prevAcceleration.x=rocketBall.acceleration.x
            prevAcceleration.y=rocketBall.acceleration.y
            prevVelocity.x=rocketBall.velocity.x
            prevVelocity.y=rocketBall.velocity.y

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
    rocketBall.enable_borders=False

    inputs=SequenceGenerator.generateCustomInputs_4tuple_relative(rocketBall,10,0.3)
    outputs=SequenceGenerator.runInputs_2tuple_relative(rocketBall,inputs)
    print(inputs)
    print(outputs)