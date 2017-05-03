import numpy as np
import codecs,json

class JsonHelper:

    #saves a dictionary in a specified file
    @staticmethod
    def save(filename,dict):
        #convert numpy arrays into lists TODO
        with open(filename, 'w') as fp:
            json.dump(dict, fp)

    #restores a dictionary from a specified path
    @staticmethod
    def restore(filename):
        with open(filename,'r') as fp:
            data=json.load(fp)
        return data



if __name__=="__main__":

    #print(isinstance(np.ones(10),np.ndarray))
    JsonHelper.save("../../data/trainingData/test.json",{"a":[3,4,5],"b":[3,4,5]})

    print(JsonHelper.restore("../../data/trainingData/test.json")["a"])
