from RocketBallGUI import *
from RocketBall import RocketBall
from SequenceGenerator import SequenceGenerator
from models.forwardPredictor import *
from models.singleStepPredictor import *

class RocketBallPredictionAnimation(RocketBallGUI):

    def __init__(self,rocketBall,inputs, predictor,dt=1./30.,relative=False):

        RocketBallGUI.__init__(self,rocketBall,dt)
        self.inputs=inputs
        self.predictor=predictor
        self.relative=relative

        position=self.rocketBall.getPosition()
        self.predicted_position=[position.x,position.y]
        self.predicted_robot=plt.Circle((self.predicted_position[0],self.predicted_position[1]),self.radius)

        self.ax.add_patch(self.predicted_robot)




    def drawAll(self):
        self.predicted_robot.center=(self.predicted_position[0],self.predicted_position[1])
        RocketBallGUI.drawAll(self)


    def initGraphics(self):
        rocketBall.reset()

        if(len(self.inputs[0])==4):
            rocketBall.setPosition(Vector2f(self.inputs[0][2],self.inputs[0][3]))

        RocketBallGUI.initGraphics(self)
        self.predicted_robot.set_facecolor((1.,0.,0.))




    def animate(self,i):
        self.rocketBall.setThrust1(self.inputs[i,0])
        self.rocketBall.setThrust2(self.inputs[i,1])

        RocketBallGUI.animate(self,i)

        #make prediction for the next timestep
        prediction=predictor([self.inputs[i+1][0],self.inputs[i+1][1]])[0]
        print(prediction)
        if(self.relative):
            prediction[0]+=rocketBall.position.x
            prediction[1]+=rocketBall.position.y

        self.predicted_position=prediction




if __name__ == "__main__":
    rocketBall=RocketBall.standardVersion()
    rocketBall.enable_borders=False
    TIMESTEPS=200


    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":2,
        "use_biases":True,
        "use_peepholes":True,
    }


    restorePath="/home/emanuel/Coding/tensorflow/SessionData(copy)/sess2_2_(10000).chkpt"
    predictor=singleStepPredictor(configuration,restorePath)

    anim=None
    inputs=SequenceGenerator.generateCustomInputs_2tuple(TIMESTEPS,0.25)
    def resetAnimation(gui):
        global anim
        rocketBall.reset()
        inputs=SequenceGenerator.generateCustomInputs_2tuple(TIMESTEPS,0.25)
        gui.predictor.reset()

        gui.inputs=inputs
        if(not anim is None):
            anim._stop()
        anim=animation.FuncAnimation(fig,gui.animate,
                                     init_func=gui.initGraphics,
                                     frames=TIMESTEPS-1,
                                     interval=30.)
        anim._start()

    fig=plt.figure()

    gui=RocketBallPredictionAnimation(rocketBall,inputs,predictor,relative=True)
    fig.canvas.mpl_connect('key_press_event', gui.keypress)
    fig.canvas.mpl_connect('key_release_event',gui.keyrelease)
    fig.canvas.mpl_connect('button_press_event',lambda event: resetAnimation(gui))

    resetAnimation(gui)
    #anim=resetAnimation(gui)
    plt.show()
