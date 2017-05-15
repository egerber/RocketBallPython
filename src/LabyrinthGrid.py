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
        lab=LabyrinthGrid(5,5)
        lab.obstacle[0][0]=True
        lab.obstacle[0][3]=True
        lab.obstacle[3][1]=True
        lab.obstacle[0][2]=True
        lab.obstacle[2][2]=True
        return lab

    def placeRandomPosition(self):

        while(True):
            x=np.random.randint(0,self.rows-1)
            y=np.random.randint(0,self.columns-1)
            if(not self.obstacle[x][y]):
                self.position[0]=x
                self.position[1]=y
                return
            else:
                pass#repeat

    def move(self,dx,dy):
        delta_x=int(max(min(math.ceil(dx), 1), -1))
        delta_y=int(max(min(math.ceil(dy), 1), -1))

        print(delta_x,delta_y)

        #check for boundaries
        #check within bounds
        if(not (self.position[0]+delta_x<0 or self.position[0]+delta_x>(self.columns-1))
            and not(self.position[1]+delta_y<0 or self.position[1]+delta_y>(self.rows-1))
            and not (self.obstacle[self.position[0]+delta_x][self.position[1]+delta_y])):
            self.position=[self.position[0]+delta_x,self.position[1]+delta_y]

        print("new position", str(self.position))
if __name__=="__main__":
    lab=LabyrinthGrid.standardVersion()

    #lab.obstacle[2][:]=True
    lab.placeRandomPosition()
    lab.position=[0,1]
    for i in range(30):
        lab.move(1,0)
