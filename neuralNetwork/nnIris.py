#coding:utf-8
import sys

sys.path.append('../data' )

import numpy as np
import irisdata
import neuNetwork

BATCH_SIZE = 8
TRAIN_STEPS = 1000

class nnIrisClassify():
    def __init__( self   ) :
        node_acts= [ [4,neuNetwork.signoid()] , [3,neuNetwork.signoid()] ] 
        #coste = neuNetwork.quadraticCoster() 
        coste = neuNetwork.cross_entropyCoster() 
        self.nn = neuNetwork.neuNetworker( 4 , node_acts , coste  )
    def fit(self) :
        dataGr0 = irisdata.getDataFromDatafile( 0 ,112 )
        dataGr = irisdata.normalizeData( dataGr0 )
        xa = [ data[0:4] for data in dataGr ]
        y_a = []
        hot = [ [1.0,0.,0.] , [0.,1.0,0.] , [0.,0.,1.0] ]
        for one in dataGr :
            y_a.append ( hot [one[4] ] )  
        Groups = len(xa)
        for step in range( TRAIN_STEPS ) :
            ind = (step % Groups)//BATCH_SIZE*BATCH_SIZE
            e = self.nn.learn( xa[ind:ind+8] ,  y_a[ind:ind+8] ,0.05 )
            if step % 500 == 0 :
                print( 'step ', step , ': ' , e )
        self.nn.dumpWeightBias()
    def pred( self , x ) :
        y = self.nn.pred( x )
        print( 'pred y :' , y )
        max = -100. ;
        maxIndex = -1 ;
        for i in range( len(y) ) :
            if y[i] > max :
                max = y[i]
                maxIndex  = i
        print( 'max out :' , max ) 
        return maxIndex

    def test(self ) :
        dataGr0 = irisdata.getDataFromDatafile( 112 ,150 )
        dataGr = irisdata.normalizeData( dataGr0 )
        xa = [ data[0:4] for data in dataGr ]
        y_a = [data[4] for data in dataGr]
        r = []
        t = 0
        y_ya = [] 
        for x , y_ in zip(xa , y_a ) :
            y = self.pred( x )
            y_ya.append( [y , y_]  )
            if y == y_:
                rone = True
                t += 1
            else :
                rone = False
            r.append( rone )
        
        print( 'pred right ' , t , ' of  38 ' , t/38*100 , '%' )
        #print( 'y_ya' , y_ya ) 

    def storeModel(self ) :
        self.nn.storeModel()
        self.nn.restoreModel()
        


def nnIrisMain() :
    nnItisC = nnIrisClassify()
    nnItisC.fit()
    nnItisC.storeModel()
    nnItisC.test()

nnIrisMain()
    
