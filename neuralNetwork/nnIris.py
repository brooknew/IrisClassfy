#coding:utf-8
import sys

sys.path.append('../data' )

import numpy as np
import irisdata
import neuNetwork
import matplotlib.pyplot as plt


BATCH_SIZE = 16
TRAIN_STEPS = 1000

class nnIrisClassify():
    def __init__( self   ) :
        node_acts= [ [4,neuNetwork.signoid()] , [3,neuNetwork.signoid()] ] 
        #coste = neuNetwork.quadraticCoster() 
        coste = neuNetwork.cross_entropyCoster() 
        self.nn = neuNetwork.neuNetworker( 4 , node_acts , coste  )
        self.nn.restoreModel()
        
    def fit(self) :
        dataGr0 = irisdata.getDataFromDatafile( 0 ,112 )
        dataGr = irisdata.normalizeData( dataGr0 )
        xa = [ data[0:4] for data in dataGr ]
        y_a = []
        hot = [ [1.0,0.,0.] , [0.,1.0,0.] , [0.,0.,1.0] ]
        for one in dataGr :
            y_a.append ( hot [one[4] ] )  
        Groups = len(xa)
        costt = [] 
        for step in range( TRAIN_STEPS ) :
            ind = (step % Groups)//BATCH_SIZE*BATCH_SIZE
            e = self.nn.learn( xa[ind:ind+BATCH_SIZE] ,  y_a[ind:ind+BATCH_SIZE] ,0.000005 )
            if step % 500 == 0 :
                print( 'step ', step , ': ' , e )
            if step % 500 == 0 :   
                self.storeModel( step , e )
            costt.append( e )
        plt.plot( costt )
        plt.show() 
        self.nn.dumpWeightBias()
    def pred( self , x ) :
        y = self.nn.pred( x )
        #print( 'pred y :' , y )
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

    def storeModel(self , traineds , ecost ) :
        self.nn.storeModel( traineds ,  ecost )
        


def nnIrisMain() :
    nnItisC = nnIrisClassify()
    
    nnItisC.fit()
    
    nnItisC.test()

nnIrisMain()
    
