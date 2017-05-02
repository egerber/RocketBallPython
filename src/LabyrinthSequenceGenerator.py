import numpy as np
from src.Labyrinth import *
class LabyrinthSequenceGenerator:

    @staticmethod
    def generateInputs_4tuple(labyrinth,count_timesteps,max_stepsize):
        labyrinth.placeRandomPosition()

        inputs=np.random.rand(count_timesteps,4)*max_stepsize*2-max_stepsize
        for i in range(count_timesteps):
            inputs[i][2]=labyrinth.position.x
            inputs[i][3]=labyrinth.position.y
        labyrinth.move(inputs[i][0],inputs[i][1])
        return inputs

if __name__=="__main__":
    lab=Labyrinth(4,4,3,2)
    inputs=LabyrinthSequenceGenerator.generateInputs_4tuple(lab,20,10)
    print(inputs)
