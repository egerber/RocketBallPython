from src.gui.RocketBallGUI import *
from src.models.inverseModel1 import *

class RocketBallInferenceAnimation(RocketBallGUI):

    def __init__(self,rocketBall,targetPosition,inferencer,count_iterations,count_timesteps,dt=1./30.):

        RocketBallGUI.__init__(self,rocketBall,dt)
        self.targetPosition=targetPosition
        self.inferencer=inferencer
        self.count_iterations=count_iterations
        self.count_timesteps=count_timesteps
    def initGraphics(self):

        RocketBallGUI.initGraphics(self)


    def animate(self,i):

        discrepancy=np.array([self.targetPosition.x-self.rocketBall.position.x,self.targetPosition.y-self.rocketBall.position.y])
        np.clip(discrepancy,a_min=-0.05,a_max=0.05,out=discrepancy)
        nextInput=self.inferencer.infer([discrepancy for i in range(self.count_timesteps)],self.count_iterations)
        #print("nextInput: ",nextInput)

        self.rocketBall.setThrust1(nextInput[0][0])
        self.rocketBall.setThrust2(nextInput[0][1])

        RocketBallGUI.animate(self,i)



if __name__ == "__main__":
    COUNT_ITERATIONS=60
    COUNT_TIMESTEPS=3

    rocketBall= RocketBall.standardVersion()
    rocketBall.enable_borders=False

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"relative_noborders"
    }


    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)

    iModel.create(COUNT_TIMESTEPS)
    iModel.restore(path)
    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        #rocketBall.reset()
        if(event is None):
            targetPosition=Vector2f(0.,1.)
        else:
            targetPosition=Vector2f(event.xdata,event.ydata)

        gui.inferencer.reset()

        gui.targetPosition=targetPosition
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=10000,
                                     interval=30.)
        anim._start()

    fig=plt.figure()

    gui=RocketBallInferenceAnimation(rocketBall,Vector2f(0.,1.),iModel,COUNT_ITERATIONS,COUNT_TIMESTEPS)
    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
