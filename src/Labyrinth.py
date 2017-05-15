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

    @staticmethod
    def standardVersion():
        lab=Labyrinth(4,4,3,2)
        lab.obstacle[0][0]=True
        lab.obstacle[0][3]=True
        lab.obstacle[3][0]=True
        lab.obstacle[0][2]=True
        lab.obstacle[2][2]=True
        return lab

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

        div_top,mod_top=divmod(self.position.y+self.radius,self.height_cell)
        div_bottom,mod_bottom=divmod(self.position.y-self.radius,self.height_cell)
        div_left,mod_left=divmod(self.position.x-self.radius,self.width_cell)
        div_right,mod_right=divmod(self.position.x+self.radius,self.width_cell)


        #check for top-collision
        #bottom_i,bottom_mod=divmod(self.position.y - self.radius, self.height_cell)
        #current_j,_=divmod(self.position.x, self.width_cell)
        #top_i,top_mod=divmod(self.position.y +self.radius, self.height_cell)
        #current_i,_=divmod(self.position.y, self.height_cell)
        #right_i,right_mod=divmod(self.position.x + self.radius, self.width_cell)
        #left_i,left_mod=divmod(self.position.x - self.radius, self.width_cell)

        distance_boundary_bottom=self.height_cell-mod_bottom
        distance_boundary_top=mod_top
        distance_boundary_left=self.width_cell-mod_left
        distance_boundary_right=mod_right



        epsilon=0.01 #add a small offset for detecting collision

        if( (self.obstacle[int(div_bottom)][int(div_left)] or self.obstacle[int(div_bottom)][int(div_right)]) and distance_boundary_bottom<=-1*delta_y + epsilon):
            self.position.y=(div_bottom+1)*self.height_cell+self.radius
            #print("bottom_collision")

        if((self.obstacle[int(div_top)][int(div_left)] or self.obstacle[int(div_top)][int(div_right)]) and distance_boundary_top<delta_y+ epsilon):
            self.position.y=(div_top)*self.height_cell-self.radius
            #print("top_collision")


        if((self.obstacle[int(div_bottom)][int(div_left)] or self.obstacle[int(div_top)][int(div_left)]) and distance_boundary_left<=-1*delta_x + epsilon):
            self.position.x=(div_left+1)*self.width_cell+self.radius
            #print("left_collision")

        if((self.obstacle[int(div_bottom)][int(div_right)] or self.obstacle[int(div_top)][int(div_right)]) and distance_boundary_right<=delta_x+ epsilon):
            self.position.x=(div_right)*self.width_cell-self.radius
            #print("right_collision")
        #check colisions
if __name__=="__main__":
    lab=Labyrinth(4,4,4,2)
    lab.position=Vector2f(0.07,0.2)
    lab.obstacle[3][:]=True
    #lab.obstacle[2][:]=True
    lab.placeRandomPosition()
    for i in range(30):
        lab.move(-0.13,0.05)

    print(lab.position)