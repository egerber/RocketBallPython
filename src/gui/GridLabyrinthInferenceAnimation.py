from src.gui.GridLabyrinthGUI import *
from src.models.inverseModel1 import *
from numpy.linalg import norm

class GridLabyrinthInferenceAnimation(GridLabyrinthGUI):

    def __init__(self,labyrinth,targetPosition,inferencer,count_iterations,count_timesteps):
        GridLabyrinthGUI.__init__(self,labyrinth)
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
        print("currentPosition",str(self.labyrinth.normed_position()))
        print("targetPosition",str(self.targetPosition))
        print("nextInput",str(nextInput))
        self.labyrinth.move_one_hot(nextInput)

        GridLabyrinthGUI.animate(self,i)
        #for self feeding network with real last position

        #for network that is given obstacles
        #self.inferencer.last_speed=[np.append(self.labyrinth.position,self.obstacles)]
        self.inferencer.last_speed=[[self.labyrinth.position[0],self.labyrinth.position[1]]]


if __name__ == "__main__":
    COUNT_ITERATIONS=30
    COUNT_TIMESTEPS=6

    lab= LabyrinthGrid.standardVersion()
    lab.position=[0,0]

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 32,
        "size_output":2,
        "size_input":6,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_50_one_hot"
    }



    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)
    iModel.create_self_feeding(COUNT_TIMESTEPS)
    iModel.create_last_timestep_optimizer(clip_min=0,clip_max=1)
    iModel.restore(path)

    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        if(event is None):
            targetPosition=Vector2f(1,0)
        else:
            targetPosition=Vector2f(int(event.xdata),int(event.ydata))

        #TODO check if this is necessary
        #gui.inferencer.reset()

        gui.targetPosition=targetPosition
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=10000,
                                     interval=1000.)
        anim._start()

    fig=plt.figure()

    gui=GridLabyrinthInferenceAnimation(lab,Vector2f(0.,1.),iModel,COUNT_ITERATIONS,COUNT_TIMESTEPS)
    #fig.canvas.mpl_connect('key_press_event', gui.keypress)
    #fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
