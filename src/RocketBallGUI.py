import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from src.RocketBall import *

class RocketBallGUI:

    def __init__(self,rocketBall,dt=1./30.):
        self.dt=dt

        self.rocketBall=rocketBall
        self.patches=[]
        self.offsetX=0.04
        self.offsetY=0.04

        self.leftborder=self.rocketBall.leftborder
        self.rightborder=self.rocketBall.rightborder
        self.topborder=self.rocketBall.topborder
        self.width_field=self.rightborder-self.leftborder
        self.height_field=self.topborder
        self.radius=self.rocketBall.getRadius()
        self.ax=plt.axes(xlim=(self.leftborder-self.width_field*self.offsetX,self.rightborder+self.width_field*self.offsetX),
                         ylim=(0-self.height_field*self.offsetY,self.topborder+self.height_field*self.offsetY))

        self.ax.axis('off')
        self.ground=Rectangle(xy=(self.leftborder-self.width_field*self.offsetX,0-self.height_field*self.offsetY),width=self.width_field+self.width_field*2*self.offsetX,
                              height=self.height_field+self.height_field*2*self.offsetX)
        self.field=Rectangle(xy=(self.leftborder,0),width=self.width_field,height=self.height_field)


        self.ground.set_facecolor((0.1,0.1,0.1))
        self.field.set_facecolor((1.,1.,1.))

        position=self.rocketBall.getPosition()

        self.robot=plt.Circle((position.x,position.y),self.radius)

        self.scalingFactor_thrust=3.
        self.thrust1=Ellipse(xy=(position.x-self.radius*2./3.,position.y-self.radius*2./3.),
                             width=self.radius*self.rocketBall.getThrust1()*self.scalingFactor_thrust,
                             height=self.radius/2,
                             angle=45.)
        self.thrust2=Ellipse(xy=(position.x+self.radius*2./3.,position.y-self.radius*2./3.),
                             width=self.radius*self.rocketBall.getThrust2()*self.scalingFactor_thrust,
                             height=self.radius/2,
                             angle=-45.)


        self.ax.add_patch(self.ground)
        self.ax.add_patch(self.field)
        self.patches=[self.thrust1,self.thrust2,self.robot]

        for p in self.patches:
            self.ax.add_patch(p)


    def initGraphics(self):

        self.thrust1.set_facecolor((1.,0.5,0.))
        self.thrust2.set_facecolor((1.,0.5,0.))
        self.robot.set_facecolor((30./255.,120./255.,220./255.))


    def drawAll(self):
        position=self.rocketBall.getPosition()
        self.thrust1.center=(position.x-self.radius*2./3.,position.y-self.radius*2./3.)
        self.thrust1.width=self.radius*self.rocketBall.getThrust1()*self.scalingFactor_thrust

        self.thrust2.center=(position.x+self.radius*2./3.,position.y-self.radius*2./3.)
        self.thrust2.width=self.radius*self.rocketBall.getThrust2()*self.scalingFactor_thrust

        self.robot.center=(position.x,position.y)

    def animate(self,i):
        self.rocketBall.update(self.dt)
        self.drawAll()

    def animateInputSequence(self,i):
        self.rocketBall.setThrust1(self.inputs[i,0])
        self.rocketBall.setThrust2(self.inputs[i,1])

        return self.animate(i)

    def keypress(self,event):
        if(event.key=="left"):
            self.rocketBall.setThrust1(1.)
        elif(event.key=="right"):
            self.rocketBall.setThrust2(1.)

    def keyrelease(self,event):
        if(event.key=="left"):
            self.rocketBall.setThrust1(0.)
        elif(event.key=="right"):
            self.rocketBall.setThrust2(0.)




        #return anim



if __name__ == "__main__":
    rocketBall=RocketBall(9.81,0.0001,0.06,0.0012,-1.5,1.5,2.0,enable_borders=True)

    fig=plt.figure()
    gui=RocketBallGUI(rocketBall)

    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fps=30.
    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=10000,
                                 interval=1000./fps)


    plt.show()
