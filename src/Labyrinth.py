import numpy as np
from src.Vector2f import *

class Labyrinth:

    def __init__(self,rows,columns,width,height):
        self.position=Vector2f(0.2,1.)
        self.obstacle=np.zeros((rows,columns),dtype=np.bool) #holds list of [i,j] refering to row i and column j
        self.radius=0.06
        self.width=width
        self.height=height
        self.width_cell= width / columns
        self.height_cell= height / rows

        self.rows=rows
        self.columns=columns
    def placeRandomPosition(self):

        while(True):
            random_x=np.random.uniform(0.0+self.radius,self.width-self.radius)
            random_y=np.random.uniform(0.0+self.radius,self.height-self.radius)
            i,_=divmod(random_y,self.height_cell)
            j,_=divmod(random_x,self.width_cell)
            if(not self.obstacle[int(i)][int(j)]):
                self.position.x=random_x
                self.position.y=random_y
                return
            else:
                pass#repeat

    def move(self,delta_x,delta_y):
        self.position.x+=delta_x
        self.position.y+=delta_y
        epsilon=0.01
        #check for boundaries
        self.position.x = max(0+self.radius+epsilon, min(self.position.x, self.width-self.radius-epsilon))
        self.position.y=  max(0+self.radius+epsilon,min(self.position.y,self.height-self.radius-epsilon))

        #check for top-collision
        bottom_i,_=divmod(self.position.y - self.radius, self.height_cell)
        current_j,_=divmod(self.position.x, self.width_cell)

        if((self.obstacle[int(bottom_i)][int(current_j)])):
            self.position.y=(bottom_i+1)*self.height_cell+self.radius
            print("top_collision")
        top_i,_=divmod(self.position.y +self.radius, self.height_cell)
        if((self.obstacle[int(top_i)][int(current_j)])):
            self.position.y=(top_i)*self.height_cell-self.radius
            print("bottom_collision")
        current_i,_=divmod(self.position.y, self.height_cell)
        left_i,_=divmod(self.position.x - self.radius, self.width_cell)
        if((self.obstacle[int(current_i)][int(left_i)])):
            self.position.x=(left_i+1)*self.width_cell+self.radius

        right_i,_=divmod(self.position.x + self.radius, self.width_cell)
        if((self.obstacle[int(current_i)][int(right_i)])):
            self.position.x=(right_i)*self.width_cell-self.radius

        #check colisions
if __name__=="__main__":
    lab=Labyrinth(4,4,4,2)
    lab.position=Vector2f(0.07,0.2)
    lab.obstacle[3][:]=True
    #lab.obstacle[2][:]=True
    lab.placeRandomPosition()
    for i in range(30):
        lab.move(-0.1,0.05)

    print(lab.position)