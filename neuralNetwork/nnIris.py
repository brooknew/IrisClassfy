#coding:utf-8
import sys

sys.path.append('../data' )

import numpy as np
import irisdata
import neuNetwork

BATCH_SIZE = 8
TRAIN_STEPS = 50000

class nnIrisClassify():
    def __init__( self   ) :
        node_acts= [ [4,neuNetwork.signoid()] , [3,neuNetwork.signoid()] ] 
        self.nn = neuNetwork.neuNetworker( 4 , node_acts )
    def fit(self) :
        dataGr0 = irisdata.getDataFromDatafile( 0 ,112 )
        dataGr = irisdata.normalizeData( dataGr0 )
        xa = [ data[0:4] for data in dataGr ]
        y_a = []
        hot = [ [1.0,0.,0.] , [0.,1.0,0.] , [0.,0.,1.0] ]
        for one in dataGr :
            y_a.append ( hot [one[4] ] )  
        print ('xa:\n', xa )
        #print ( 'y_a' , y_a )
        Groups = len(xa)// BATCH_SIZE
        for step in range( TRAIN_STEPS ) :
            ind = (step % Groups) *BATCH_SIZE
            e = self.nn.learn( xa[ind:ind+8] ,  y_a[ind:ind+8] ,0.005 )
            if step % 100 == 0 :
                print( 'step ', step , ': ' , e )
        self.nn.dumpWeightBias()
         
        
        


def nnIrisMain() :
    nnItisC = nnIrisClassify()
    nnItisC.fit()

nnIrisMain()
    
