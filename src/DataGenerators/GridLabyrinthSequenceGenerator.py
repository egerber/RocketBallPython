from src.LabyrinthGrid import LabyrinthGrid
import numpy as np

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
    @staticmethod
    def generateInputs_one_hot_obstacles(labyrinth,count_timesteps):
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
        for r in range(labyrinth.rows):
            for c in range(labyrinth.columns):
                if(labyrinth.obstacle[r][c]):
                    obstacles[r*labyrinth.rows+c]=1

        return obstacles

    @staticmethod
    def generateOutputs_one_hot(labyrinth,inputs):
        labyrinth.position=labyrinth.unnormed_position(inputs[0][4:6])
        outputs=np.zeros((len(inputs),2))
        for i in range(len(inputs)):
            labyrinth.move_one_hot(inputs[i,:4])
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


if __name__=='__main__':
    lab=LabyrinthGrid.standardVersion()

    inputs=GridLabyrinthSequenceGenerator.generateInputs_one_hot_obstacles(lab,10)
    outputs=GridLabyrinthSequenceGenerator.generateOutputs_one_hot(lab,inputs)
    print(inputs,outputs)
    #print(inputs)