from src.RocketBallGUI import *
from src.RocketBall import RocketBall
from src.SequenceGenerator import SequenceGenerator

class RocketBallSequenceAnimation(RocketBallGUI):

    def __init__(self,rocketBall,inputs, dt=1./30.):
        RocketBallGUI.__init__(self,rocketBall,dt)
        self.inputs=inputs

    def initGraphics(self):

        rocketBall.reset()
        if(len(self.inputs[0])==4):
            rocketBall.setPosition(Vector2f(self.inputs[0][2],self.inputs[0][3]))

        RocketBallGUI.initGraphics(self)

    def animate(self,i):
        self.rocketBall.setThrust1(self.inputs[i,0])
        self.rocketBall.setThrust2(self.inputs[i,1])
        RocketBallGUI.animate(self,i)



if __name__ == "__main__":
    rocketBall=RocketBall.standardVersion()

    TIMESTEPS=200
    inputs=SequenceGenerator.generateCustomInputs_4tuple(rocketBall,TIMESTEPS,0.2,dt=1/30,gaussian=True)


    fig=plt.figure()
    gui=RocketBallSequenceAnimation(rocketBall,inputs)
    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=TIMESTEPS,
                                 interval=30.)


    plt.show()
