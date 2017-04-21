import numpy as np

def createConfigurationString(configuration):

    confString=configuration["cell_type"]+"("+str(configuration["num_hidden_units"])+")_"+str(configuration["size_input"])+"_"+str(configuration["size_output"])
    if("tag" in configuration):
        confString+="("+str(configuration['tag'])+")"

    return confString

def toAbsoluteOutputs(outputs,starting_position):

    absolute_outputs=np.empty((len(outputs)+1,2))
    absolute_outputs[0]=starting_position
    for i in range(len(outputs)):
        absolute_outputs[i+1]=absolute_outputs[i]+outputs[i]

    return absolute_outputs

