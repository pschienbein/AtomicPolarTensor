import numpy as np
import logging

class Box:
    def __init__(self):
        self.mTrans = np.zeros((3,3))

        pass

    def loadFromVectors(self, lattice):
        self.mTrans = np.array(lattice).transpose()
        self.mInvTrans = np.linalg.inv(self.mTrans)
        pass

    def loadFromStatusFile(self, sf):
        vectors = sf.get_boxVectors()

        if vectors == None:
            logging.error('Box vectors not found in status file, but construction of box requested; maybe program not properly initialized??')
            raise RuntimeError()

        self.mTrans[0][0] = vectors['a'][0]
        self.mTrans[0][1] = vectors['a'][1] 
        self.mTrans[0][2] = vectors['a'][2] 
        self.mTrans[1][0] = vectors['b'][0] 
        self.mTrans[1][1] = vectors['b'][1] 
        self.mTrans[1][2] = vectors['b'][2] 
        self.mTrans[2][0] = vectors['c'][0] 
        self.mTrans[2][1] = vectors['c'][1] 
        self.mTrans[2][2] = vectors['c'][2] 
        self.mTrans = self.mTrans.transpose()

        self.mInvTrans = np.linalg.inv(self.mTrans)

        pass

    def getAbs2FracMatrix(self):
        return self.mInvTrans

    def getFrac2AbsMatrix(self):
        return self.mTrans

    def getLatticeVectors(self):
        return self.mTrans.transpose()

    def pbc(self,v):
        vInt = self.mInvTrans.dot(v)
        vCorr = vInt.round()
        return self.mTrans.dot(vInt - vCorr)

    def abs2frac(self, v):
        return self.mInvTrans.dot(v)

    def frac2abs(self, v):
        return self.mTrans.dot(v)

