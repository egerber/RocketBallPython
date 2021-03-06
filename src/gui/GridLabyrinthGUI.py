import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from src.LabyrinthGrid import *
from src.DataGenerators.GridLabyrinthSequenceGenerator import GridLabyrinthSequenceGenerator

class GridLabyrinthGUI:

    def __init__(self,labyrinth):
        self.labyrinth=labyrinth
        self.patches=[]
        self.offsetX=0.04
        self.offsetY=0.04

        self.leftborder=0
        self.rightborder=labyrinth.columns
        self.topborder=labyrinth.rows
        self.width_field=self.rightborder-self.leftborder
        self.height_field=self.topborder

        self.radius=0.06
        self.ax=plt.axes(xlim=(self.leftborder-self.width_field*self.offsetX,self.rightborder+self.width_field*self.offsetX),
                         ylim=(0-self.height_field*self.offsetY,self.topborder+self.height_field*self.offsetY))

        self.ax.axis('off')
        self.ground=Rectangle(xy=(self.leftborder-self.width_field*self.offsetX,0-self.height_field*self.offsetY),width=self.width_field+self.width_field*2*self.offsetX,
                              height=self.height_field+self.height_field*2*self.offsetX)
        self.field=Rectangle(xy=(self.leftborder,0),width=self.width_field,height=self.height_field)


        self.ground.set_facecolor((0.1,0.1,0.1))
        self.field.set_facecolor((1.,1.,1.))

        position=self.labyrinth.position

        self.robot=plt.Rectangle((position[0],position[1]),width=1,height=1)


        self.ax.add_patch(self.ground)
        self.ax.add_patch(self.field)

        for i in range(labyrinth.rows):
            for j in range(labyrinth.columns):
                if(labyrinth.obstacle[i][j]):

                    obstacle=Rectangle(xy=(i,j),width=1,
                                       height=1)
                    obstacle.set_facecolor((0.9,0.9,0.9))
                    self.ax.add_patch(obstacle)

        self.patches=[self.robot]

        for p in self.patches:
            self.ax.add_patch(p)


    def initGraphics(self):
        self.robot.set_facecolor((30./255.,120./255.,220./255.))


    def drawAll(self):
        position=self.labyrinth.position
        self.robot.xy=(position[0],position[1])

    def animate(self,i):
        self.drawAll()

    def keypress(self,event):
        if(event.key=="left"):
            self.labyrinth.move_one_hot([1,0,0,0])
        elif(event.key=="right"):
            self.labyrinth.move_one_hot([0,1,0,0])
        elif(event.key=="up"):
            self.labyrinth.move_one_hot([0,0,1,0])
        elif(event.key=="down"):
            self.labyrinth.move_one_hot([0,0,0,1])



if __name__ == "__main__":
    lab=LabyrinthGrid.smallVersion(8,15)

    fig=plt.figure()
    gui=GridLabyrinthGUI(lab)

    fig.canvas.mpl_connect('key_press_event', gui.keypress)

    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=10000,
                                 interval=100)


    plt.show()
