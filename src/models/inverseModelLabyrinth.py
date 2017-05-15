from src.models.inverseModel1 import *
from src.Labyrinth import *
from src.models.helper import *
import os

if __name__=='__main__':
    COUNT_ITERATIONS=100
    COUNT_TIMESTEPS=3

    lab= Labyrinth.standardVersion()
    lab.position=Vector2f(1.75,0.75)
    configuration={
        "cell_type":"LSTMCell",
        "num_hidden_units": 16,
        "size_output":2,
        "size_input":4,
        "use_biases":True,
        "use_peepholes":True,
        "tag":"Labyrinth_200_0.2"
    }


    #outputs=[[-0.0]*configuration["size_output"] for i in range(COUNT_TIMESTEPS)]
    outputs=[2.8,1.8]
    path=checkpointDirectory=os.path.dirname(__file__)+"/../../data/checkpoints/"+createConfigurationString(configuration)+".chkpt"

    iModel=inverseModel(configuration)
    iModel.create_self_feeding(COUNT_TIMESTEPS)
    iModel.create_all_timesteps_optimizer(clip_min=-0.2,clip_max=0.2)
    iModel.restore(path)

    begin=time.time()
    for i in range(200):
        print(iModel.infer_self_feeding([outputs for i in range(COUNT_TIMESTEPS)],COUNT_ITERATIONS)[0])
    end=time.time()
    print(end-begin)