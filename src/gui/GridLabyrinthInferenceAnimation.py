from src.gui.GridLabyrinthGUI import *
from src.models.inverseModel1 import *
from numpy.linalg import norm
from src.DataGenerators.GridLabyrinthSequenceGenerator import GridLabyrinthSequenceGenerator
from src.models.inverseModelGridLabyrinth import inverseModelGridLabyrinth

class GridLabyrinthInferenceAnimation(GridLabyrinthGUI):

    def __init__(self,labyrinth,targetPosition,obstacle_information,inferencer,count_iterations,count_timesteps):
        GridLabyrinthGUI.__init__(self,labyrinth)
        self.targetPosition=targetPosition
        self.inferencer=inferencer
        self.count_iterations=count_iterations
        self.count_timesteps=count_timesteps
        self.obstacle_information=obstacle_information

    def animate(self,i):

        #discrepancy=np.array([self.targetPosition.x-self.labyrinth.position.x,self.targetPosition.y-self.labyrinth.position.y])
        #distance=min(0.1,math.sqrt(discrepancy[0]**2+discrepancy[1]**2))
        #np.clip(discrepancy,a_min=-0.05,a_max=0.05,out=discrepancy)
        #discrepancy=discrepancy/norm(discrepancy) *distance

        nextInput=self.inferencer.infer_self_feeding([[self.targetPosition.x,self.targetPosition.y] for i in range(self.count_timesteps)],self.count_iterations,
                                                     self.obstacle_information)[0]
        print("currentPosition",str(self.labyrinth.normed_position()))
        print("targetPosition",str(self.targetPosition))
        print("nextInput",str(nextInput))
        self.labyrinth.move_one_hot(nextInput)

        GridLabyrinthGUI.animate(self,i)

        self.inferencer.last_speed=[[self.labyrinth.position[0],self.labyrinth.position[1]]]
        #MAYBE
        # self.inferencer.reset()


if __name__ == "__main__":
    COUNT_TIMESTEPS=1
    COUNT_OBSTALCE_CONFIGURATIONS=100
    COUNT_OBSTACLES=30
    COUNT_ITERATIONS=30

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 32,
        "size_output":2,
        "size_grid": 100,
        "size_input":6+100,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth_50_100_30"
    }

    lab=LabyrinthGrid.standardVersion()
    seed=1
    lab.setRandomObstacles(COUNT_OBSTACLES,seed)

    obstacle_information=GridLabyrinthSequenceGenerator.obstacleInformation(lab)

    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModelGridLabyrinth(configuration)
    iModel.create_self_feeding(COUNT_TIMESTEPS)
    iModel.create_last_timestep_optimizer(0.,1.)
    iModel.restore(path)

    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        if(event is None):
            targetPosition=Vector2f(1,0)
        targetPosition=Vector2f(1.,1.)
        #else:
            #targetPosition=Vector2f(int(event.xdata)/lab.columns,int(event.ydata)/lab.rows)

        gui.targetPosition=targetPosition
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=10000,
                                     interval=1000.)
        anim._start()

    fig=plt.figure()

    gui=GridLabyrinthInferenceAnimation(lab,Vector2f(0.,1.),obstacle_information,iModel,COUNT_ITERATIONS,COUNT_TIMESTEPS)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
