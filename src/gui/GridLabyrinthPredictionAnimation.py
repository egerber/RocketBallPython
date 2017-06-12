import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
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
        self.predicted_robot=Rectangle((position[0],position[1]),width=1.,height=1.)
        self.arrow,=self.ax.plot([0.,1.],[1.,2.],color='c')
        self.arrow_base=Circle((0.,0.),0.15,color='c')
        self.ax.add_patch(self.predicted_robot)
        self.ax.add_patch(self.arrow_base)
        #self.ax.add_patch(self.arrow)

    def initGraphics(self):
        self.robot.set_facecolor((30./255.,120./255.,220./255.))
        self.predicted_robot.set_facecolor((1.,0.,0.))
        self.predictor.reset()

        self.labyrinth.apply_configuration(self.inputs[0])

    def drawAll(self):
        GridLabyrinthGUI.drawAll(self)

        self.arrow_base.center=(self.last_x+0.5,self.last_y+0.5)
        self.arrow.set_xdata([self.last_x+0.5,self.last_x+0.5+self.dx])
        self.arrow.set_ydata([self.last_y+0.5,self.last_y+0.5+self.dy])
        self.predicted_robot.xy=(self.predicted_position[0],self.predicted_position[1])




    def animate(self,i):
        motor_input=self.inputs[i][:4]
        self.last_x=self.labyrinth.position[0]
        self.last_y=self.labyrinth.position[1]

        self.labyrinth.move_one_hot(motor_input)
        prediction=predictor(self.inputs[i])[0]
        print(self.labyrinth.normed_position(),prediction)
        if(inputs[i][0]==1):
            self.dx=-1.
            self.dy=0.
        elif(inputs[i][1]==1):
            self.dx=1
            self.dy=0.
        elif(inputs[i][2]==1):
            self.dx=0.
            self.dy=1.
        elif(inputs[i][3]==1):
            self.dx=0.
            self.dy=-1.

        print(self.dx,self.dy)

        self.predicted_position=self.labyrinth.unnormed_position(prediction,round_discrete=True)
        GridLabyrinthGUI.animate(self,i)



if __name__ == "__main__":
    COUNT_TIMESTEPS=1000
    COUNT_OBSTALCE_CONFIGURATIONS=100
    COUNT_OBSTACLES=9
    BATCH_SIZE=32
    COUNT_TRAININGS_PER_CONFIGURATION=1000
    SEED=20

    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":31,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"GridLabyrinth(0.001)"+"_PRETRAINED_50_100_500_32"#str(COUNT_TIMESTEPS)+"_"+str(COUNT_TRAININGS_PER_CONFIGURATION)+"_"+str(COUNT_OBSTACLES)+"_"+str(COUNT_OBSTALCE_CONFIGURATIONS)+"_"+str(BATCH_SIZE)
    }



    fig=plt.figure()

    lab=LabyrinthGrid.smallVersion(COUNT_OBSTACLES,SEED)
    anim=None
    inputs,_=GridLabyrinthSequenceGenerator.generateTrainingData_one_hot_obstacles(5,5,COUNT_TIMESTEPS,COUNT_OBSTACLES,SEED)

    path=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"
    predictor=singleStepForwardModel.createFromOld(configuration,path)
    gui=GridLabyrinthPredictionAnimation(lab,inputs,predictor)
    def resetAnimation(gui):
        global anim,iModel,path,inputs

        gui.predictor.reset()

        #gui.inputs=inputs

        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=COUNT_TIMESTEPS-1,
                                     interval=200)
        anim._start()

    resetAnimation(gui)
    plt.show()