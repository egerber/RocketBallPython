import numpy as np
from src.Labyrinth import *
class LabyrinthSequenceGenerator:

    @staticmethod
    def generateCustomInputs_4tuple(labyrinth,count_timesteps,max_stepsize,changingProbability):

        inputs=(np.random.rand(count_timesteps,4)*2-1)*max_stepsize
        inputs[0][2]=labyrinth.position.x
        inputs[0][3]=labyrinth.position.y
        for i in range(1,count_timesteps):
            if(np.random.rand()>changingProbability):
                inputs[i][0]=inputs[i-1][0]
            if(np.random.rand()>changingProbability):
                inputs[i][1]=inputs[i-1][1]

            inputs[i][2]=labyrinth.position.x
            inputs[i][3]=labyrinth.position.y
            labyrinth.move(inputs[i][0],inputs[i][1])

        return inputs

    @staticmethod
    def generateInputs_4tuple(labyrinth,count_timesteps,max_stepsize):
        labyrinth.placeRandomPosition()

        inputs=(np.random.rand(count_timesteps,4)*2-1)*max_stepsize
        for i in range(count_timesteps):
            inputs[i][2]=labyrinth.position.x
            inputs[i][3]=labyrinth.position.y
            labyrinth.move(inputs[i][0],inputs[i][1])
        return inputs

    @staticmethod
    def generateOutputs_2tuple(labyrinth,inputs):
        labyrinth.position.x=inputs[0][2]
        labyrinth.position.y=inputs[0][3]

        outputs=np.zeros((len(inputs),2))
        for i in range(len(inputs)):
            labyrinth.move(inputs[i][0],inputs[i][1])
            outputs[i][0]=labyrinth.position.x
            outputs[i][1]=labyrinth.position.y

        return outputs

if __name__=="__main__":
    lab=Labyrinth(4,4,3,2)
    inputs=LabyrinthSequenceGenerator.generateCustomInputs_4tuple(lab,200,0.1,0.3)
    print(inputs)
    print(LabyrinthSequenceGenerator.generateOutputs_2tuple(lab,inputs))