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

        self.inferencer.last_speed
        nextInput=self.inferencer.infer_self_feeding([[self.targetPosition.x,self.targetPosition.y] for i in range(self.count_timesteps)],self.count_iterations,
                                                     self.obstacle_information)[0]
        print("currentPosition",str(self.labyrinth.normed_position()))
        print("targetPosition",str(self.targetPosition))
        print("nextInput",str(nextInput))

        self.labyrinth.move_one_hot(nextInput)
        GridLabyrinthGUI.animate(self,i)
        self.inferencer.last_speed=[self.labyrinth.normed_position()]
        #MAYBE
        # self.inferencer.reset()


if __name__ == "__main__":
    COUNT_TIMESTEPS_INFERENCE=5
    COUNT_TIMESTEPS=50
    COUNT_EPOCHS=31
    COUNT_OBSTALCE_CONFIGURATIONS=1
    COUNT_OBSTACLES=0
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=10000

    COUNT_ITERATIONS=1


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":106,
        "use_biases":True,
        "use_peepholes":True,
        "size_grid":100,
        "tag":"GridLabyrinth(0.001)_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    }

    lab=LabyrinthGrid.standardVersion()
    seed=1
    lab.setRandomObstacles(COUNT_OBSTACLES,seed)

    obstacle_information=GridLabyrinthSequenceGenerator.obstacleInformation(lab)

    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModelGridLabyrinth(configuration,learning_rate=100.)
    iModel.create_self_feeding(COUNT_TIMESTEPS_INFERENCE)
    iModel.create_last_timestep_optimizer(0.,1.)
    iModel.restore(path)

    anim=None
    def resetAnimation(gui,event=None):
        global anim,iModel,path

        if(event is None):
            targetPosition=Vector2f(1,1)
        else:
            targetPosition=Vector2f(int(event.xdata)/lab.columns,int(event.ydata)/lab.rows)

        gui.inferencer.last_speed=[lab.normed_position()]
        gui.targetPosition=targetPosition
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=10000,
                                     interval=400.)
        anim._start()

    fig=plt.figure()

    gui=GridLabyrinthInferenceAnimation(lab,Vector2f(0.,1.),obstacle_information,iModel,COUNT_ITERATIONS,COUNT_TIMESTEPS_INFERENCE)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui,event))

    resetAnimation(gui)
    plt.show()
