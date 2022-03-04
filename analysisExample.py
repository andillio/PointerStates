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
traceOutModes = [3,4]
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
        self.N_files = None 

        self.indToTuple = None 
        self.tupleToInd = None  
        self.indToTupleR = None 
        self.tupleToIndR = None  
        self.reduceMap = None   

        self.decimate = None
fo = figObj()

def GetDicts(fo, reduce_):
    newIndToTuple = {} # dictionary describing index to tuple mapping for the total hilbert space
    newTupleToInd = {} # dictionary describing tuple to index mapping -- --
    newReduceMap = {} # the reduction map describing how to efficiently reduce the total hilbert space

    # for state in the initial super position
    for tag_ in fo.tags:

        # load its "special" Hilbert space map
        with open("../" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple = pickle.load(f)

        # for state in the special hilbert space
        for i in range(len(indToTuple)):
            state_ = indToTuple[i] # get the state

            # add this state to the total hilbert space maps
            ind_ = len(newIndToTuple) 
            newIndToTuple[ind_] = state_
            newTupleToInd[state_] = ind_

            # construct the reduction map

            # get the number of particles in the modes that are
            # being traced out, i.e. calculate |Environment>_i
            toReduce = []
            for k in range(len(reduce_)):
                toReduce.append( state_[reduce_[k]] )
            toReduce = tuple(toReduce)

            # if |Environment>_i is in the reduction map already
            if (toReduce in newReduceMap):
                # then add it to the list of states with the same |Environment>_i
                newReduceMap[toReduce].append(ind_)
            else:
                # else add a new |Environment>_i to the map
                newReduceMap[toReduce] = [ind_]
    
    # get the index <-> tuple maps for the reduced hilbert space
    fo.indToTupleR, fo.tupleToIndR = qu.GetReducedDicts(newIndToTuple, newTupleToInd, reduce_)

    return newIndToTuple, newTupleToInd, newReduceMap


def setFigObj(name, reduce_, decimate = decimate):
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

    print fo.tags[0]
    # this is basically just to see how many time drops there were
    fo.fileNames_psi = u.getNamesInds('Data/' + name + "/" + "psi" + fo.tags[0])
    fo.N_files = len(fo.fileNames_psi)

    fo.indToTuple, fo.tupleToInd, fo.reduceMap = GetDicts(fo, reduce_)

def GetLinEntropy(reduce_):
    # get dictionaries
    qu.GetDicts(fo)

    S = np.zeros(fo.N_files)
    t = np.zeros(fo.N_files)

    # get wavefunction 
    for i in range(len(fo.N_files)):
        psi_, N_ = qu.GetPsiAndN(i, fo)

        # take the partial trace of the modes in reduce_
        reducedRho = qu.PsiToReduceRhoSmart(psi_, fo.indToTuple, fo.tupleToInd,\
                fo.indToTupleR, fo.tupleToIndR, fo.reduceMap, reduce_)

        t[fo.dt*fo.framsteps*i] # get the timestamp

        # calculate the appropriate entropies
        S[i] = qu.S_linAlt(reducedRho)
    return S


def main(name, tags = [], reduce_ = traceOutModes, label = "", decimate = 1, plot = PLOT):
    time0 = time.time()

    fo.tags = tags
    fo.decimate = decimate

    setFigObj(name, reduce_)

    S = GetLinEntropy(reduce_)

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))

if __name__ == "__main__":
    # load in the tags on the data directories
    try:
        fo.tags = np.load("../Data/" + simName + "/tags.npy", dtype = 'U')
    except IOError:
        fo.tags = [""]
    main(simName, fo.tags, traceOutModes, decimate=decimate)