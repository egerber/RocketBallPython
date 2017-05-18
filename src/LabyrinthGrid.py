import numpy as np
from src.Vector2f import *
import math

class LabyrinthGrid:

    def __init__(self,rows,columns):
        self.position=[0,0]
        self.obstacle=np.zeros((rows,columns),dtype=np.bool) #holds list of [i,j] refering to row i and column j

        self.rows=rows
        self.columns=columns

    @staticmethod
    def standardVersion():
        lab=LabyrinthGrid(10,10)
        lab.setRandomObstacles(5,100)
        lab.placeRandomPosition()
        return lab

    def setRandomObstacles(self,count_obstacles,seed=100):
        self.obstacle=np.zeros((self.rows,self.columns),dtype=np.bool) #holds list of [i,j] refering to row i and column j
        np.random.seed(seed)

        count_fields=self.rows*self.columns

        indices=[divmod(index,self.rows) for index in np.random.choice(range(count_fields),count_obstacles,replace=False)]


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
    def convert_position(x,y):
        pass

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


    def unnormed_position(self,normed_position):
        factor_x=(self.columns-1)
        factor_y=(self.rows-1)
        return [normed_position[0]*factor_x,normed_position[1]*factor_y]


    def normed_position(self):
        if(self.columns>1 and self.rows>1):
            factor_x=1/(self.columns-1)
            factor_y=1/(self.rows-1)
            return [self.position[0]*factor_x,self.position[1]*factor_y]
        else:
            return self.position
    @staticmethod
    def convert_softmax(vector):
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

    def move(self,dxy):
        dx,dy=dxy
        delta_x=LabyrinthGrid.convert_motorInputs(dx)
        delta_y=LabyrinthGrid.convert_motorInputs(dy)

        #check for boundaries
        #check within bounds
        if(not (self.position[0]+delta_x<0 or self.position[0]+delta_x>(self.columns-1))
            and not( self.position[1]+delta_y<0 or self.position[1]+delta_y>(self.rows-1))
            and not (self.obstacle[int(self.position[0]+delta_x)][int(self.position[1]+delta_y)])):
            self.position=[int(self.position[0]+delta_x),int(self.position[1]+delta_y)]

    def move_one_hot(self,vector):
        if(not LabyrinthGrid.is_one_hot(vector)):
            vector=LabyrinthGrid.convert_softmax(vector)


        if(vector[0]==1):
            self.move([-1,0])
        elif(vector[1]==1):
            self.move([1,0])
        elif(vector[2]==1):
            self.move([0,1])
        elif(vector[3]==1):
            self.move([0,-1])


if __name__=="__main__":
    lab=LabyrinthGrid.standardVersion()

    #lab.obstacle[2][:]=True
    lab.placeRandomPosition()
    #lab.position=[0,1]
    lab.setRandomObstacles(24,101)
    print(lab.obstacle)