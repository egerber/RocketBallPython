import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from src.LabyrinthGrid import *
from src.gui.GridLabyrinthGUI import GridLabyrinthGUI
from src.DataGenerators.GridLabyrinthSequenceGenerator import GridLabyrinthSequenceGenerator
from src.models.singleStepForwardModel import singleStepForwardModel
import os
from src.models.helper import *

class GridLabyrinthPredictionAnimation(GridLabyrinthGUI):

    def __init__(self,labyrinth,inputs,predictor):
        GridLabyrinthGUI.__init__(self,labyrinth)
        self.predictor=predictor
        self.inputs=inputs

        position=self.labyrinth.position
        self.predicted_position=self.labyrinth.position
        self.predicted_robot=plt.Rectangle((position[0],position[1]),width=1,height=1)

        self.ax.add_patch(self.predicted_robot)

    def initGraphics(self):
        self.robot.set_facecolor((30./255.,120./255.,220./255.))
        self.predicted_robot.set_facecolor((1.,0.,0.))
        self.predictor.reset()
        startPosition=self.labyrinth.unnormed_position(self.inputs[0][4:6])
        self.labyrinth.position=startPosition

    def drawAll(self):
        GridLabyrinthGUI.drawAll(self)
        self.predicted_robot.xy=(self.predicted_position[0],self.predicted_position[1])

    def animate(self,i):

        motor_input=self.inputs[i][:4]
        self.labyrinth.move_one_hot(motor_input)
        prediction=predictor(self.inputs[i])[0]
        print(self.labyrinth.normed_position(),prediction)

        self.predicted_position=self.labyrinth.unnormed_position(prediction)
        GridLabyrinthGUI.animate(self,i)



if __name__ == "__main__":
    COUNT_TIMESTEPS=50
    COUNT_OBSTALCE_CONFIGURATIONS=1
    COUNT_OBSTACLES=30
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=10000
    SEED=1

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":106,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth(0.001)_"+str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    }



    lab=LabyrinthGrid.standardVersion()
    lab.setRandomObstacles(COUNT_OBSTACLES,SEED)
    fig=plt.figure()


    anim=None
    inputs=None


    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"
    predictor=singleStepForwardModel.createFromOld(configuration,path)
    gui=GridLabyrinthPredictionAnimation(lab,inputs,predictor)
    def resetAnimation(gui):
        global anim,iModel,path

        #for 6->2 mapping
        #inputs=GridLabyrinthSequenceGenerator.generateInputs_one_hot(lab,COUNT_TIMESTEPS)
        #for 106->2 mapping
        inputs,_=GridLabyrinthSequenceGenerator.generateTrainingData_random_obstacles(lab,COUNT_TIMESTEPS,COUNT_OBSTACLES,SEED)

        gui.predictor.reset()

        gui.inputs=inputs

        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=COUNT_TIMESTEPS-1,
                                     interval=500)
        anim._start()

    resetAnimation(gui)
    plt.show()
