import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from src.Labyrinth import *
from src.LabyrinthSequenceGenerator import *

class LabyrinthSequenceAnimation:

    def __init__(self,labyrinth,inputs):
        self.labyrinth=labyrinth
        self.inputs=inputs

        self.patches=[]
        self.offsetX=0.04
        self.offsetY=0.04

        self.leftborder=0
        self.rightborder=labyrinth.width
        self.topborder=self.labyrinth.height
        self.width_field=self.rightborder-self.leftborder
        self.height_field=labyrinth.height

        self.radius=self.labyrinth.radius
        self.ax=plt.axes(xlim=(self.leftborder-self.width_field*self.offsetX,self.rightborder+self.width_field*self.offsetX),
                         ylim=(0-self.height_field*self.offsetY,self.topborder+self.height_field*self.offsetY))

        self.ax.axis('off')
        self.ground=Rectangle(xy=(self.leftborder-self.width_field*self.offsetX,0-self.height_field*self.offsetY),width=self.width_field+self.width_field*2*self.offsetX,
                              height=self.height_field+self.height_field*2*self.offsetX)
        self.field=Rectangle(xy=(self.leftborder,0),width=self.width_field,height=self.height_field)


        self.ground.set_facecolor((0.1,0.1,0.1))
        self.field.set_facecolor((1.,1.,1.))

        position=self.labyrinth.position

        self.robot=plt.Circle((position.x,position.y),self.radius)


        self.ax.add_patch(self.ground)
        self.ax.add_patch(self.field)

        for i in range(labyrinth.rows):
            for j in range(labyrinth.columns):
                if(labyrinth.obstacle[i][j]):
                    print(str(i*labyrinth.width_cell),str(j*labyrinth.height_cell),str(self.labyrinth.width_cell),str(self.labyrinth.height_cell))
                    obstacle=Rectangle(xy=(j*labyrinth.width_cell,i*labyrinth.height_cell),width=self.labyrinth.width_cell,
                                       height=self.labyrinth.height_cell)
                    obstacle.set_facecolor((0.9,0.9,0.9))
                    self.ax.add_patch(obstacle)

        self.patches=[self.robot]

        for p in self.patches:
            self.ax.add_patch(p)


    def initGraphics(self):

        self.robot.set_facecolor((30./255.,120./255.,220./255.))


    def drawAll(self):
        position=self.labyrinth.position

        self.robot.center=(position.x,position.y)

    def animate(self,i):
        self.labyrinth.move(self.inputs[i][0],self.inputs[i][1])
        self.drawAll()

    def keypress(self,event):
        if(event.key=="left"):
            self.labyrinth.move(-0.07,0)
        elif(event.key=="right"):
            self.labyrinth.move(0.06,0)
        elif(event.key=="up"):
            self.labyrinth.move(0.,0.12)
        elif(event.key=="down"):
            self.labyrinth.move(0.,-0.09)



if __name__ == "__main__":

    COUNT_TIMESTEPS=2000
    MAX_STEPSIZE=0.2

    lab=Labyrinth.standardVersion()
    fig=plt.figure()

    inputs=LabyrinthSequenceGenerator.generateCustomInputs_4tuple(lab,COUNT_TIMESTEPS,MAX_STEPSIZE,0.3)
    gui=LabyrinthSequenceAnimation(lab,inputs)

    fig.canvas.mpl_connect('key_press_event', gui.keypress)

    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=COUNT_TIMESTEPS,
                                 interval=150)


    plt.show()
