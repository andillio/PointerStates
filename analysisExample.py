import utils as u
import QUtils as qu 
import numpy as np
import multiprocessing as mp 
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import pickle 
from numpy import linalg as LA 
import scipy.stats as st 
import sys
import yt; yt.enable_parallelism()
import sys

simName = "NE_HS_current"
decimate = 2
label = ""
PLOT = True

class figObj(object):

    def __init__(self):
        self.meta = None

        self.tags = None 

        self.N = None 
        self.dt = None 
        self.framesteps = None 
        self.IC = None 
        self.phi = None 

        self.name = None 
        self.fileNames_psi = None

        self.indToTuple = None 
        self.tupleToInd = None    

        self.decimate = None
fo = figObj()

def setFigObj(name, decimate = decimate):
    '''
    This function populates the attributes of the instance of the figObj class
    with values.
    '''
    # read in simulation parameters
    meta = u.getMetaKno(name, dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", IC = "IC", omega0 = "omega0", Lambda0 = "Lambda0")
    fo.meta = meta

    # sets the figure object with these parameters
    # this is basically just so I can access them in the glocal scope
    fo.name = name

    fo.N = fo.meta["N"]
    fo.dt = fo.meta["dt"]
    fo.framsteps = fo.meta["framesteps"]
    fo.IC =  fo.meta["IC"]

    fo.decimate = decimate

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)

    # this is basically just to see how many time drops there were
    fo.fileNames_psi = u.getNamesInds('Data/' + name + "/" + "psi" + fo.tags[0])


def main(name, tags = [], label = "", decimate = 1, plot = PLOT):
    time0 = time.time()

    fo.tags = tags
    fo.decimate = decimate

    setFigObj(name)

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))

if __name__ == "__main__":
    # load in the tags on the data directories
    try:
        fo.tags = np.load("../Data/" + simName + "/tags.npy")
    except IOError:
        fo.tags = [""]
    main(simName, fo.tags, decimate=decimate)