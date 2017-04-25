import numpy as np
from src.Vector2f import *

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class RocketBall:

    @staticmethod
    def standardVersion():
        return RocketBall(9.81,0.0001,0.06,0.0012,-1.5,1.5,2.0)

    def reset(self):

        self.gvec=Vector2f(0.,0.)#Check if okay
        self.position=Vector2f(0.0,0.2+self.radius)

        #initialize thrust1
        self.thrust1dir=Vector2f.normalize(Vector2f(1.,1.))
        self.thrust1forcevec=Vector2f(0.,0.)
        self.thrust1=0.
        #initialize thrust2
        self.thrust2dir=Vector2f.normalize(Vector2f(-1.,1.))
        self.thrust2forcevec=Vector2f(0.,0.)
        self.thrust2=0.

        self.forcesumvec=Vector2f(0.,0.)
        self.velocity=Vector2f(0.,0.)
        self.acceleration=Vector2f(0.,0.)

        self.decay=0.9
    def __init__(self,g,mass,radius,maxthrust,leftborder,rightborder,topborder,enable_borders=True):
        self.g=g
        self.gvec=Vector2f(0.,0.)#Check if okay
        self.mass=mass
        self.radius=radius
        self.leftborder=leftborder
        self.rightborder=rightborder
        self.maxthrust=maxthrust
        self.topborder=topborder
        self.position=Vector2f(0.0,0.2+radius)

        #initialize thrust1
        self.thrust1dir=Vector2f.normalize(Vector2f(1.,1.))
        self.thrust1forcevec=Vector2f(0.,0.)
        self.thrust1=0.
        #initialize thrust2
        self.thrust2dir=Vector2f.normalize(Vector2f(-1.,1.))
        self.thrust2forcevec=Vector2f(0.,0.)
        self.thrust2=0.

        self.forcesumvec=Vector2f(0.,0.)
        self.velocity=Vector2f(0.,0.)
        self.acceleration=Vector2f(0.,0.)

        self.decay=0.9
        self.enable_borders=enable_borders
        self.use_sigmoid=False

    def placeDefault(self):
        self.position=Vector2f((self.rightborder-self.leftborder)/2.,self.topborder/2)

    @staticmethod
    def computeGravityForceVec(mass,g,forcevec):
        forcevec.x=0.0
        forcevec.y=-1.
        Vector2f.mul(forcevec,mass*g,forcevec)

    def getTopBorder(self):
        return self.topborder

    def getLeftBorder(self):
        return self.leftborder

    def getRightBorder(self):
        return self.rightborder

    def getRadius(self):
        return self.radius

    def getThrust1Dir(self):
        return self.thrust1dir

    def getThrust2Dir(self):
        return self.thrust2dir

    def getThrust1(self):
        return self.thrust1

    def getThrust2(self):
        return self.thrust2

    def setThrust1(self,thrust):
        self.thrust1=thrust

    def setThrust2(self,thrust):
        self.thrust2=thrust

    def getPosition(self):
        return self.position

    def setPosition(self,position):
        self.position=position

    def update(self,dt=1./30.):
        self.computeGravityForceVec(self.mass,self.g,self.gvec)

        if(self.use_sigmoid):
            self.thrust1=sigmoid(self.thrust1)
            self.thrust2=sigmoid(self.thrust2)
        else:
            self.thrust1=np.clip(self.thrust1,0.,1.)
            self.thrust2=np.clip(self.thrust2,0.,1.)

        self.thrust1dir=Vector2f.normalize(self.thrust1dir)
        self.thrust2dir=Vector2f.normalize(self.thrust2dir)
        Vector2f.mul(self.thrust1dir,self.thrust1*self.maxthrust,self.thrust1forcevec)
        Vector2f.mul(self.thrust2dir,self.thrust2*self.maxthrust,self.thrust2forcevec)

        #sum forces
        Vector2f.add(self.gvec,self.thrust1forcevec,self.forcesumvec)
        Vector2f.add(self.forcesumvec,self.thrust2forcevec,self.forcesumvec)

        #compute acceleration vector
        Vector2f.mul(self.forcesumvec,1.0/self.mass,self.acceleration)

        hypvel=Vector2f(
            self.velocity.x + self.acceleration.x*dt,
            self.velocity.y + self.acceleration.y*dt
        )
        hyppos=Vector2f(
            self.position.x+hypvel.x*dt,
            self.position.y+hypvel.y*dt
        )

        ylodiff=hyppos.y - self.radius
        yhidiff=hyppos.y-(self.topborder - self.radius)

        nvelx=hypvel.x
        nvely=hypvel.y
        nposx=hyppos.x
        nposy=hyppos.y
        if(self.enable_borders):
            if (ylodiff<=0) and (hypvel.y<0):
                nvely=0.0
                nvelx=hypvel.x*self.decay
                nposy=self.radius
            elif (yhidiff>=0) and (hypvel.y>0):
                nvely=0.
                nvelx=hypvel.x*self.decay
                nposy=self.topborder-self.radius

            xlediff=hyppos.x-(self.leftborder+self.radius)
            xridiff=hyppos.x-(self.rightborder-self.radius)

            if (xlediff<=0) and (hypvel.x<0):
               nvelx=0.
               nvely=hypvel.y*self.decay
               nposx=self.leftborder+self.radius
            elif (xridiff>=0) and (hypvel.x>0):
                nvelx=0.
                nvely=(hypvel.y*self.decay)
                nposx=self.rightborder-self.radius

        self.velocity.x=nvelx
        self.velocity.y=nvely
        self.position.x=nposx
        self.position.y=nposy


