import numpy as np
import math

class Vector2f:

    def __init__(self):
        pass

    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __str__(self):
        return "("+str(self.x)+", "+str(self.y) +")"

    @staticmethod
    def mul(pv,pscalar,presult):
        presult.x=pv.x*pscalar
        presult.y=pv.y*pscalar

    @staticmethod
    def add(p1,p2,presult):
        presult.x=p1.x+p2.x
        presult.y=p1.y+p2.y

    @staticmethod
    def normalize(vector2f):
        norm=math.sqrt(vector2f.x**2+vector2f.y**2)
        return Vector2f(vector2f.x/norm,vector2f.y/norm)

    def toArray(self):
        return [self.x,self.y]
