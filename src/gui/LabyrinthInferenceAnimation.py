from src.gui.LabyrinthGUI import *
from src.models.inverseModel1 import *
from numpy.linalg import norm
from src.gui.RocketBallInferenceAnimation import RocketBallInferenceAnimation

class LabyrinthInferenceAnimation(LabyrinthGUI):

    def __init__(self,labyrinth,targetPosition,inferencer,count_iterations,count_timesteps):
        LabyrinthGUI.__init__(self,labyrinth)
        self.targetPosition=targetPosition
        self.inferencer=inferencer
        self.count_iterations=count_iterations
        self.count_timesteps=count_timesteps

    def animate(self,i):

        #discrepancy=np.array([self.targetPosition.x-self.labyrinth.position.x,self.targetPosition.y-self.labyrinth.position.y])
        #distance=min(0.1,math.sqrt(discrepancy[0]**2+discrepancy[1]**2))
        #np.clip(discrepancy,a_min=-0.05,a_max=0.05,out=discrepancy)
        #discrepancy=discrepancy/norm(discrepancy) *distance

        nextInput=self.inferencer.infer_self_feeding([[self.targetPosition.x,self.targetPosition.y] for i in range(self.count_timesteps)],self.count_iterations)[0]
        print(self.labyrinth.position)
        self.labyrinth.move(nextInput[0],nextInput[1])

        LabyrinthGUI.animate(self,i)
        #for self feeding network with real last position
        self.inferencer.last_speed=[[self.labyrinth.position.x,self.labyrinth.position.y]]


if __name__ == "__main__":
    COUNT_ITERATIONS=30
    COUNT_TIMESTEPS=100

    lab= Labyrinth.standardVersion()
    lab.position=Vector2f(1.75,0.75)
    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":4,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"Labyrinth_200_0.2_custom(0.3)"
    }


    #outputs=[[-0.0]*configuration["size_output"] for i in range(COUNT_TIMESTEPS)]
    outputs=[2.8,1.8]
    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)
    iModel.create_self_feeding(COUNT_TIMESTEPS)
    iModel.create_last_timestep_optimizer(clip_min=-0.05,clip_max=0.05)
    iModel.restore(path)

    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        #rocketBall.reset()
        if(event is None):
            targetPosition=Vector2f(2.8,1.8)
        else:
            targetPosition=Vector2f(event.xdata,event.ydata)

        #TODO check if this is necessary
        #gui.inferencer.reset()

        gui.targetPosition=targetPosition
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=10000,
                                     interval=50.)
        anim._start()

    fig=plt.figure()

    gui=LabyrinthInferenceAnimation(lab,Vector2f(0.,1.),iModel,COUNT_ITERATIONS,COUNT_TIMESTEPS)
    #fig.canvas.mpl_connect('key_press_event', gui.keypress)
    #fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
