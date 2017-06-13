import numpy as np
from src.Vector2f import *
import math

class LabyrinthGrid:

    def __init__(self,rows,columns):
        self.position=[0,0]
        self.obstacle=np.zeros((columns,rows),dtype=np.bool) #holds list of [i,j] refering to row i and column j

        self.rows=rows
        self.columns=columns

    @staticmethod
    def standardVersion(count_obstacles=30,seed=1):
        lab=LabyrinthGrid(10,10)
        lab.setRandomObstacles(count_obstacles,seed)
        lab.placeRandomPosition()
        return lab

    @staticmethod
    def smallVersion(count_obstacles=9,seed=1):
        lab=LabyrinthGrid(5,5)
        lab.setRandomObstacles(count_obstacles,seed)
        lab.placeRandomPosition()
        return lab

    def setRandomObstacles(self,count_obstacles,seed=100):
        self.obstacle=np.zeros((self.columns,self.rows),dtype=np.bool) #holds list of [i,j] refering to row i and column j
        r = np.random.RandomState(seed)

        count_fields=self.rows*self.columns

        indices=[divmod(index,self.rows) for index in r.choice(range(count_fields),count_obstacles,replace=False)]


        for i in indices:
            self.obstacle[i[0]][i[1]]=True


    def placeRandomPosition(self):

        while(True):
            x=np.random.randint(0,self.columns)
            y=np.random.randint(0,self.rows)
            if(not self.obstacle[x][y]):
                self.position[0]=x
                self.position[1]=y
                return
            else:
                pass#repeat


    @staticmethod
    def convert_motorInputs(scalar):
        #split interval [-1,1] into 3 equal parts and return -1, 0, or 1 if they lie inside this part
        size_interval=2./3.
        if(scalar<=-1.+size_interval):
            return int(-1)
        elif(scalar>=1.-size_interval):
            return int(1)
        else:
            return int(0)

    def apply_configuration(self,inputs):
        initial_position=self.unnormed_position(inputs[4:6])
        self.position=initial_position

        obstacle_information=inputs[6:]
        self.obstacle=np.zeros((self.columns,self.rows))
        for i in range(len(obstacle_information)):
            row,col=divmod(i,self.columns)
            self.obstacle[row][col]=bool(obstacle_information[i])


    def unnormed_position(self,normed_position,round_discrete=False):
        factor_x=(self.columns-1)
        factor_y=(self.rows-1)

        #round to discrete values
        if(round_discrete):
            return [round(normed_position[0]*factor_x),round(normed_position[1]*factor_y)]
        else:
            return [normed_position[0]*factor_x,normed_position[1]*factor_y]

    def normed_position(self):
        if(self.columns>1 and self.rows>1):
            factor_x=1/(self.columns-1)
            factor_y=1/(self.rows-1)
            return [self.position[0]*factor_x,self.position[1]*factor_y]
        else:
            return self.position
    @staticmethod
    def convert_one_hot(vector):
        max_index=np.argmax(vector)
        one_hot_vector=np.zeros(4)
        one_hot_vector[max_index]=1

        return one_hot_vector

    @staticmethod
    def is_one_hot(vector):
        if(np.sum(vector)==1 and max(vector)==1):
            return True
        else:
            return False

    def sensor_information(self):
        sensor_information=np.zeros(4)
        #check for left obstacle
        if(self.position[0]-1<0 or self.obstacle[self.position[0]-1][self.position[1]]):
            sensor_information[0]=1
        if(self.position[0]+1>=self.columns or self.obstacle[self.position[0]+1][self.position[1]]):
            sensor_information[1]=1
        if(self.position[1]-1<0 or self.obstacle[self.position[0]][self.position[1]-1]):
            sensor_information[2]=1
        if(self.position[1]+1>=self.rows or self.obstacle[self.position[0]][self.position[1]+1]):
            sensor_information[3]=1

        return sensor_information

    #takes only values € [-1,0,1]
    def move(self,dxy):
        delta_x,delta_y=dxy

        #check for boundaries
        #check within bounds
        if(not (self.position[0]+delta_x<0 or self.position[0]+delta_x>(self.columns-1))
            and not( self.position[1]+delta_y<0 or self.position[1]+delta_y>(self.rows-1))
            and not (self.obstacle[int(self.position[0]+delta_x)][int(self.position[1]+delta_y)])):
            self.position=[int(self.position[0]+delta_x),int(self.position[1]+delta_y)]

    def move_one_hot(self,vector):
        if(not LabyrinthGrid.is_one_hot(vector)):
            vector=LabyrinthGrid.convert_one_hot(vector)


        if(vector[0]==1):
            self.move([-1,0])
        elif(vector[1]==1):
            self.move([1,0])
        elif(vector[2]==1):
            self.move([0,1])
        elif(vector[3]==1):
            self.move([0,-1])


if __name__=="__main__":
    lab=LabyrinthGrid.smallVersion(20,1)
    print(lab.obstacle)
    print(lab.position)
    print(lab.sensor_information())
    print(lab.obstacle[lab.position[0]][lab.position[1]])