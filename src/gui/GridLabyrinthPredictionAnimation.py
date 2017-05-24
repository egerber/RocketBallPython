import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import colors
from src.LabyrinthGrid import *
from src.gui.GridLabyrinthGUI import GridLabyrinthGUI
from src.DataGenerators.GridLabyrinthSequenceGenerator import GridLabyrinthSequenceGenerator

class GridLabyrinthPredictionAnimation(GridLabyrinthGUI):

    def __init__(self,labyrinth,inputs,predictor):
        GridLabyrinthGUI.__init__(self,labyrinth)
        self.predictor=predictor
        self.inputs=inputs

        position=self.labyrinth.getPosition()
        self.predicted_position=[position.x,position.y]
        self.predicted_robot=plt.Rectangle((position[0],position[1]),width=1,height=1)

        self.patches.add(self.predicted_robot)

    def initGraphics(self):
        self.robot.set_facecolor((30./255.,120./255.,220./255.))
        self.predictor.reset()


    def animate(self,i):
        self.labyrinth.move_one_hot(self.inputs[i])
        _input=self.labyrinth.position+self.inputs[i]
        predicted_position=self.predictor(_input)
        self.predicted_position=predicted_position
        self.drawAll()



if __name__ == "__main__":
    COUNT_TIMESTEPS=100
    COUNT_OBSTACLES=30
    SEED=1

    lab=LabyrinthGrid.standardVersion()
    lab.setRandomObstacles(30,3)
    fig=plt.figure()

    inputs=GridLabyrinthSequenceGenerator.generateInputs_one_hot_obstacles(lab,COUNT_TIMESTEPS,COUNT_OBSTACLES,SEED)

    gui=GridLabyrinthPredictionAnimation(lab,


    anim=animation.FuncAnimation(fig,gui.animate,
                                 init_func=gui.initGraphics,
                                 frames=10000,
                                 interval=10)


    plt.show()
