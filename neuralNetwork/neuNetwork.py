#coding:utf-8
import numpy as np
import math

DEBUG00 = False
DEBUG01 = False

class activation() :
    def act(self , x ) :
        print("do noth" ) 
class signoid( activation ) :
    def act( self ,x ) :
        xx = 1./(1 + np.exp(-x) )
        return xx 
class relu( activation ) :
    def act( self ,x ) :
        xx = [ i if i > 0 else -i for i in x ]
        return xx 

class coster() :
    def cost( self , y_ , y ) :
        print( 'nothing' ) 
class quadraticCoster( coster ) :
    def cost( self , y_ , y ) :
        ee = 0.
        for comp1 , comp2 in zip ( y_ , y ) :
            ee +=  (comp1 - comp2)**2
        return ee
class cross_entropyCoster( coster ) :
    def cost( self , y_ , y ) :
        ee = 0.
        for comp1 , comp2 in zip ( y_ , y ) :
            e = comp1 * math.log(comp2) + (1-comp1)* math.log( 1- comp2 )
            ee -=  e
        return ee    
    

class neuNetworker() :
    def __init__(self ,  inDim , n_a , coste ) :
        '''inDim =>Dimension of input
           n_a =>list of tuple ( nodes , activation ) of each layer
           coste=> cost(computing cost/loss) instance of class inherted from coster 
        '''
        self.indepDelta = 0.0001
        self.inDim = inDim
        self.n_a = n_a
        self.wba = []
        self.coster = coste 
        lastDim = inDim
        lev = 1 
        for i in  n_a :
            dim = i[0]
            if  not DEBUG00: 
                w = np.random.random( (lastDim , dim  ))
                b = np.random.random( ( dim  ) )
            else :
                w = np.ones( (lastDim , dim  ))
                w[1][1] = 2
                w[2][0] = 3
                w[3][2] = -1                 
                b = np.ones( ( dim  ) )
                b[0] = -4
            if DEBUG01:
                print('w',lev , ':\n' , w )
                print('b',lev , ':\n' , b ) 
            a = i[1]
            wba = [w,b,a]
            self.wba.append( wba )
            lev += 1
    def storeModel( self ) :
        with open('model.txt' , 'wt' ) as fw :
            st = 'layers:%d coster:%s\n'%( len( self.wba ) ,type( self.coster ) )            
            fw.write( st  )
            i = 0 
            for wba in  self.wba  :
                st = '  %d activation:\n  %s\n'%( i, type( wba[2] ))
                i += 1 
                fw.write( st )
            
            for i in range( len( self.wba ) ) :
                st = ' layer:%d\n'%i
                fw.write( st )
                wba = self.wba[i]
                w = wba[0]
                st = ' w:\n'
                fw.write( st )
                for j in range(w.shape[0]) :
                    st = '   '
                    for k in range(w.shape[1]) :
                        st += str(w[j][k] ) + ' '
                    st += '\n'
                    fw.write( st )
                b = wba[1] 
                st = ' b:\n   '
                for k in range(b.shape[0]) : 
                    st += str(b[k] ) + ' '
                st += '\n'
                fw.write( st )
    def restoreModel( self ) :
        with open('model.txt' , 'rt' ) as fr :
            st = 'layers:%d coster:%s\n'%( len( self.wba ) ,type( self.coster ) )            
            st1 = fr.read()
            print( 'st1' , st1 ) 
            if st != st1 :
                print ( 'no restore' )
                return 
            i = 0 
            for wba in  self.wba  :
                st = '  %d activation:\n  %s\n'%( i, type( wba[2] ))
                i += 1 
                st1 = fr.read()
                if st != st1 :
                    print ( 'no restore' )
                    return             
            for i in range( len( self.wba ) ) :
                st = ' layer:%d\n'%i
                fw.write( st )
                wba = self.wba[i]
                w = wba[0]
                st = ' w:\n'
                fw.write( st )
                for j in range(w.shape[0]) :
                    st = '   '
                    for k in range(w.shape[1]) :
                        st += str(w[j][k] ) + ' '
                    st += '\n'
                    fw.write( st )
                b = wba[1] 
                st = ' b:\n   '
                for k in range(b.shape[0]) : 
                    st += str(b[k] ) + ' '
                st += '\n'
                fw.write( st )
                
    def dumpWeightBias( self ) :
        i = 1 
        for wba in self.wba :
            w = wba[0]
            b = wba [1]
            print( 'layer ' ,  i , ':\n' ,'weight:\n' , w )
            print( 'bias:\n' , b )
            i = i + 1           
    def forwardOne(self , x ) :
        xx = x 
        for wba in self.wba :
            xx = np.dot( xx , wba[0])
            xx +=  wba[1]
            #if DEBUG01 :
                #print( 'y:' , xx ) 
            xx = wba[2].act(xx)
        return xx
        
    
    def forward(self , xa ) :
        ya = [] 
        for  x in xa :
            xx = self.forwardOne( x )  
            ya.append (xx)
        return ya
    
    def costAvg( self , y_a , ya ) :
        e = 0.0
        for  y_  , y   in  zip( y_a , ya )  :
            ee = self.coster.cost ( y_ , y )
            e += ee ;
        e  /=  len( y_a )
        return e

    def learn(self , xa ,  y_a , learnRate ) :
        ya = self.forward( xa )
        e1 = self.costAvg( y_a , ya )
        wbaTemps =[]
        for i in range(len( self.wba)):
            wba = self.wba[i] 
            w = wba[0]
            wTemp = np.zeros( w.shape , dtype = np.float64 )
            for  r  in   range( len( w ) )  :
                for c in  range ( len ( w[r] ) ) :
                    v = self.wba[i][0][r][c] 
                    self.wba[i][0][r][c] += self.indepDelta
                    ya = self.forward( xa  )
                    e2 = self.costAvg( y_a , ya )
                    learnN = (e2-e1)/self.indepDelta * learnRate # (e2-e1)/self.indepDelta :loss函數的微分 
                    self.wba[i][0][r][c] =  v
                    wTemp[r][c] = 10.0
                    wTemp[r][c]  =  v - learnN
            b = wba[1]
            bTemp = np.zeros( b.shape , dtype = np.float64 )
            for  r  in  range( len( b ) )  :
                v = self.wba[i][1][r] 
                self.wba[i][1][r] += self.indepDelta
                ya = self.forward( xa  )
                e2 = self.costAvg( y_a , ya )
                learnN = (e2-e1)/self.indepDelta * learnRate # (e2-e1)/self.indepDelta :loss函數的微分 
                self.wba[i][1][r] =  v
                bTemp[r]  =  v- learnN
            wbaTemp = [wTemp, bTemp , self.wba[i][2] ]
            wbaTemps.append( wbaTemp ) 
        self.wba = wbaTemps
        #compute new error to return:
        ya = self.forward( xa )
        e = self.costAvg( y_a , ya )
        return e
    def pred( self , x ) :
        y = self.forwardOne( x )
        return y ;  
        

''' Main for  Leaned by myself '''
BATCH_SIZE = 4
TRAIN_STEPS = 100000
def directLearnMain() :
    xa = np.array( [[1,0,0,1],[0,1,0,0] , [0,1,0,1], [0,1,1,1]  ])
    y_a = np.array( [[1.,0.,0.],[0.,1.,0.] , [0.,0.,1.], [0.,1.0,0.]  ])
    if DEBUG01:
        print( 'xa:\n' , xa )
    act1 = signoid()
    #act1 = relu()
    wba=[[4,act1] , [3,act1] ]
    coste = quadraticCoster() 
    coste = cross_entropyCoster() 
    nn = neuNetworker( 4 , wba , coste )
    ya = nn.forward(xa)
    if DEBUG01:
        print( 'ya type:' , type(ya) , 'ya[0] type' , type( ya[0] ) )
        print('ya:\n' ,  ya )
    for step in range( TRAIN_STEPS ) :
        e = nn.learn( xa ,  y_a ,0.01 )
        if step % 100 == 0 :
            print( 'step ', step , ': ' , e )
    nn.dumpWeightBias()
        

if __name__ == '__main__':               
    directLearnMain()        

