from src.LabyrinthGrid import LabyrinthGrid
import numpy as np
from src.helper import JsonHelper

class GridLabyrinthSequenceGenerator:


    @staticmethod
    def generateInputs_4tuple(labyrinth,count_timesteps):
        labyrinth.placeRandomPosition()
        inputs=np.random.uniform(-1.,1.,size=(count_timesteps,4))
        inputs[0][2:4]=labyrinth.normed_position()
        labyrinth.move(inputs[0][0],inputs[0][1])
        for i in range(1,count_timesteps):
            inputs[i][2:4]=labyrinth.normed_position()
            labyrinth.move(inputs[i][0],inputs[i][1])

        return inputs

    #instead of one_hot vectors it creates random vectors with elements element of [0,1] (they get "arg-maxed" into one_hot internally"
    @staticmethod
    def generateTrainingData_random_obstacles(labyrinth,count_timesteps,count_obstacles,seed):
        labyrinth.placeRandomPosition()
        labyrinth.setRandomObstacles(count_obstacles,seed)

        obstacle_info=GridLabyrinthSequenceGenerator.obstacleInformation(labyrinth)
        inputs=np.random.rand(count_timesteps,106)

        inputs[:,6:]=obstacle_info

        outputs=np.zeros((count_timesteps,2))
        inputs[0][4:6]=labyrinth.normed_position()
        labyrinth.move_one_hot(inputs[0,:4])#
        outputs[0]=labyrinth.normed_position()
        for i in range(1,count_timesteps):
            inputs[i][4:6]=labyrinth.normed_position()
            labyrinth.move_one_hot(inputs[i,:4])
            outputs[i,:]=labyrinth.normed_position()

        return [inputs,outputs]

    @staticmethod
    def generateTrainingData_one_hot_obstacles(count_timesteps,count_obstacles,seed):
        labyrinth=LabyrinthGrid.standardVersion(count_obstacles,seed)
        obstacle_info=GridLabyrinthSequenceGenerator.obstacleInformation(labyrinth)
        inputs=np.zeros(shape=(count_timesteps,106))
        inputs[:,6:]=obstacle_info
        rand_indices_one=np.random.randint(low=0,high=4,size=(count_timesteps))
        for i in range(count_timesteps):
            inputs[i,rand_indices_one[i]]=1

        outputs=np.zeros((count_timesteps,2))
        inputs[0][4:6]=labyrinth.normed_position()
        labyrinth.move_one_hot(inputs[0,:4])#
        outputs[0]=labyrinth.normed_position()
        for i in range(1,count_timesteps):
            inputs[i][4:6]=labyrinth.normed_position()
            labyrinth.move_one_hot(inputs[i,:4])
            outputs[i,:]=labyrinth.normed_position()

        return [inputs,outputs]

    @staticmethod
    def generateTrainingData_one_hot(labyrinth,count_timesteps):
        labyrinth.placeRandomPosition()

        inputs=np.zeros(shape=(count_timesteps,6))
        rand_indices_one=np.random.randint(low=0,high=4,size=(count_timesteps))
        for i in range(count_timesteps):
            inputs[i,rand_indices_one[i]]=1

        outputs=np.zeros((count_timesteps,2))
        inputs[0][4:6]=labyrinth.normed_position()
        labyrinth.move_one_hot(inputs[0,:4])#
        outputs[0]=labyrinth.normed_position()
        for i in range(1,count_timesteps):
            inputs[i][4:6]=labyrinth.normed_position()
            labyrinth.move_one_hot(inputs[i,:4])
            outputs[i,:]=labyrinth.normed_position()

        return inputs,outputs
    @staticmethod
    def generateInputs_one_hot_obstacles(count_timesteps,count_obstacles,seed):
        labyrinth=LabyrinthGrid.standardVersion(COUNT_OBSTACLES,seed)

        obstacle_info=GridLabyrinthSequenceGenerator.obstacleInformation(labyrinth)
        inputs=np.empty(shape=(count_timesteps,len(obstacle_info)+6))
        inputs[:,:6]=GridLabyrinthSequenceGenerator.generateInputs_one_hot(labyrinth,count_timesteps)
        inputs[:,6:]=obstacle_info

        return inputs
    @staticmethod
    def generateInputs_one_hot(labyrinth,count_timesteps):
        labyrinth.placeRandomPosition()
        inputs=np.zeros(shape=(count_timesteps,6))

        rand_indices_one=np.random.randint(low=0,high=4,size=(count_timesteps))
        for i in range(count_timesteps):
            inputs[i,rand_indices_one[i]]=1

        inputs[0][4:6]=labyrinth.normed_position()
        labyrinth.move_one_hot(inputs[0,:4])
        for i in range(1,count_timesteps):
            inputs[i][4:6]=labyrinth.normed_position()
            labyrinth.move_one_hot(inputs[i,:4])

        return inputs

    @staticmethod
    def obstacleInformation(labyrinth):
        obstacles=np.zeros(labyrinth.columns*labyrinth.rows)
        for i in range(labyrinth.rows*labyrinth.columns):
            row,col=divmod(i,labyrinth.columns)
            obstacles[i]=int(labyrinth.obstacle[row][col])

        return obstacles

    @staticmethod
    def generateOutputs_one_hot(labyrinth,inputs):
        labyrinth.position=labyrinth.unnormed_position(inputs[0][4:6])

        outputs=np.zeros((len(inputs),2))
        for i in range(len(inputs)):
            labyrinth.move_one_hot(inputs[i][:4])
            outputs[i][0:2]=labyrinth.normed_position()

        return outputs

    @staticmethod
    def generateOutputs(labyrinth,inputs):
        labyrinth.position=labyrinth.unnormed_position(inputs[0][2:4])
        outputs=np.zeros((len(inputs),2))
        for i in range(len(inputs)):
            labyrinth.move(inputs[i][0],inputs[i][1])
            outputs[i][0:2]=labyrinth.normed_position()

        return outputs

    @staticmethod
    def testTrainingData_one_hot_obstacles(trainingData):
        labyrinth=LabyrinthGrid.standardVersion(0,1)
        inputs,outputs=trainingData
        for i,obstacle in enumerate(inputs[0,6:]):
            row,col=divmod(i,labyrinth.columns)
            labyrinth.obstacle[row][col]=bool(obstacle)

        labyrinth.position=labyrinth.unnormed_position(inputs[0,4:6])
        for index,input in enumerate(inputs):
            position_before=labyrinth.position
            output=outputs[index]
            labyrinth.move_one_hot(input[:4])
            normed_position=labyrinth.normed_position()
            if(normed_position[0]!=output[0] or normed_position[1]!=output[1]):
                print("Wrong dataset found")
                print(normed_position,outputs[index])
                position_after=labyrinth.position


if __name__=='__main__':
    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=10
    COUNT_OBSTACLES=30
    COUNT_TRAININGS_PER_CONFIGURATION=2000



    lab=LabyrinthGrid.standardVersion(COUNT_OBSTACLES,1)


    #trainingsData=[GridLabyrinthSequenceGenerator.generateTrainingData_random_obstacles(lab,COUNT_TIMESTEPS,COUNT_OBSTACLES,seed) for seed in seeds for i in range(COUNT_TRAININGS_PER_CONFIGURATION)]
    #inputs=[trainingsSet[0].tolist() for trainingsSet in trainingsData]
    #outputs=[trainingsSet[1].tolist() for trainingsSet in trainingsData]
    #dict={"inputs":inputs,"outputs":outputs}
    #JsonHelper.JsonHelper.save("../../data/trainingData/GridLabyrinth/"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS),dict)

