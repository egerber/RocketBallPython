from src.gui.RocketBallGUI import *
from src.models.inverseModel1 import *
from numpy.linalg import norm
from src.helper.JsonHelper import JsonHelper
from src.SequenceGenerator2 import SequenceGenerator2

class RocketBallForward2Animation(RocketBallGUI):

    def __init__(self,rocketBall,inputs,outputs,relative=False):

        RocketBallGUI.__init__(self,rocketBall)

        self.inputs=None
        self.outputs=None
        self.initialPosition=None
        self.last_target=None
        self.relative=relative
        self.reset(inputs,outputs)

        self.target_robot=plt.Circle((self.rocketBall.position.x,self.rocketBall.position.y),self.radius)
        self.target_robot.set_facecolor((1.,0.,0.))
        self.ax.add_patch(self.target_robot)

        self.last_position=[self.rocketBall.position.x,self.rocketBall.position.y]

    def initGraphics(self):
        RocketBallGUI.initGraphics(self)

    def reset(self,inputs,outputs):
        self.initialPosition=[self.rocketBall.position.x,self.rocketBall.position.y]

        self.inputs=inputs
        self.outputs=outputs
        self.rocketBall.reset()
        self.initialPosition=[self.rocketBall.position.x,self.rocketBall.position.y]
        self.last_position=[self.rocketBall.position.x,self.rocketBall.position.y]
        self.last_target=self.initialPosition
    def drawAll(self):

        RocketBallGUI.drawAll(self)

    def animate(self,i):

        target=[self.rocketBall.position.x+self.inputs[i][0],self.rocketBall.position.y+self.inputs[i][1]]
        self.target_robot.center=(target[0],target[1])
        self.last_target=target
        if(self.relative):
            self.rocketBall.position=Vector2f(self.last_position[0]+self.outputs[i][0],self.last_position[1]+self.outputs[i][1])
            self.last_position=[self.rocketBall.position.x,self.rocketBall.position.y]
        else:
            self.rocketBall.position=Vector2f(self.initialPosition[0]+self.outputs[i][0],self.initialPosition[1]+self.outputs[i][1])

        self.drawAll()

if __name__ == "__main__":
    COUNT_TIMESTEPS=50

    trainingDict=JsonHelper.restore("../../data/trainingData/training2_(relative,100,50,30,1,0.05).json")
    inputs=trainingDict["inputs"]
    outputs=trainingDict["outputs"]

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False



    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        #rocketBall.reset()
        if(event is None):
            targetPosition=Vector2f(0.,1.)
        else:
            targetPosition=Vector2f(event.xdata,event.ydata)

        rand_index=np.random.randint(0,len(inputs)-1)
        gui.reset(inputs[rand_index],outputs[rand_index])

        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=COUNT_TIMESTEPS,
                                     interval=100.)
        anim._start()

    fig=plt.figure()

    rand_index=np.random.randint(0,len(inputs)-1)
    gui=RocketBallForward2Animation(rocketBall,inputs[rand_index],outputs[rand_index],relative=True)
    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
