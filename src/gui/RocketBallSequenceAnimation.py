from src.DataGenerators.SequenceGenerator import SequenceGenerator
from src.gui.RocketBallGUI import *


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
    rocketBall.enable_borders=False
    rocketBall.use_sigmoid=False

    TIMESTEPS=100


    inputs=None
    anim=None
    def resetAnimation(gui):
        global anim
        rocketBall.reset()
        inputs=SequenceGenerator.generateCustomInputs_2tuple(TIMESTEPS,0.7,gaussian=True,mean=0.35,std=0.7)


        gui.inputs=inputs
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=TIMESTEPS-1,
                                     interval=40.)
        anim._start()
    fig=plt.figure()
    gui=RocketBallSequenceAnimation(rocketBall,inputs)

    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui))

    resetAnimation(gui)
    anim=resetAnimation(gui)
    plt.show()


    plt.show()
