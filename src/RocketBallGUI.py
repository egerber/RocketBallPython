import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from RocketBall import *

class RocketBallGUI:

    def __init__(self,rocketBall,dt=1./30.):
        self.dt=dt

        self.rocketBall=rocketBall
        self.patches=[]
        self.offsetX=0.04
        self.offsetY=0.04

    def initGraphics(self):
        leftborder=self.rocketBall.leftborder
        rightborder=self.rocketBall.rightborder
        topborder=self.rocketBall.topborder
        width_field=rightborder-leftborder
        height_field=topborder

        self.ax=plt.axes(xlim=(leftborder-width_field*self.offsetX,rightborder+width_field*self.offsetX),
                         ylim=(0-height_field*self.offsetY,topborder+height_field*self.offsetY))

        self.ax.axis('off')
        self.ground=Rectangle(xy=(leftborder-width_field*self.offsetX,0-height_field*self.offsetY),width=width_field+width_field*2*self.offsetX,
                              height=height_field+height_field*2*self.offsetX)
        self.field=Rectangle(xy=(leftborder,0),width=width_field,height=height_field)


        self.ground.set_facecolor((0.1,0.1,0.1))
        self.field.set_facecolor((1.,1.,1.))

        position=self.rocketBall.getPosition()
        self.radius=self.rocketBall.getRadius()
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
        self.thrust1.set_facecolor((1.,0.5,0.))
        self.thrust2.set_facecolor((1.,0.5,0.))
        self.robot.set_facecolor((30./255.,120./255.,220./255.))

        self.ax.add_patch(self.ground)
        self.ax.add_patch(self.field)
        self.patches=[self.thrust1,self.thrust2,self.robot]

        for p in self.patches:
            self.ax.add_patch(p)


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
    rocketBall=RocketBall(9.81,0.0001,0.06,0.0012,-1.5,1.5,2.0)

    gui=RocketBallGUI(rocketBall)
    #fig.canvas.mpl_connect('key_press_event', gui.keypress)
    #fig.canvas.mpl_connect('key_release_event',gui.keyrelease)

    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fps=30.
    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=10000,
                                 interval=1000./fps)


    plt.show()
