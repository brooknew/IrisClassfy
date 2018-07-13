#coding:utf-8
import numpy as np

BATCH_SIZE = 4
TRAIN_STEPS = 150000

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

class neuNetwork() :
    def __init__(self ,  inDim , n_a ) :
        '''inDim =>Dimension of input
           n_a =>list of tuple ( nodes , activation ) of each layer
        '''
        self.indepDelta = 0.0001
        self.inDim = inDim
        self.n_a = n_a
        self.wba = []
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
                #print('w type' , type(w)  )
                print('w',lev , ':\n' , w )
                print('b',lev , ':\n' , b ) 
            a = i[1]
            wba = [w,b,a]
            self.wba.append( wba )
            lev += 1 
    def dumpWeightBias( self ) :
        i = 1 
        for wba in self.wba :
            w = wba[0]
            b = wba [1]
            print( 'layer ' ,  i , ':\n' ,'weight:\n' , w )
            print( 'bias:\n' , b )
            i = i + 1           
        
    
    def forward(self , xa ) :
        ya = [] 
        for  x in xa :
            xx = x 
            for wba in self.wba :
                xx = np.dot( xx , wba[0])
                xx +=  wba[1]
                #if DEBUG01 :
                    #print( 'y:' , xx ) 
                xx = wba[2].act(xx)
            ya.append (xx)
        return ya
    
    def squareErrorEach( self , y_ , y ) :
        ee = 0.
        for comp1 , comp2 in zip ( y_ , y ) :
            ee +=  (comp1 - comp2)**2
        return ee
    
    def squareErrorAvg( self , y_a , ya ) :
        e = 0.0
        for  y_  , y   in  zip( y_a , ya )  :
            ee = self.squareErrorEach( y_ , y )
            e += ee ;
            #yindex = yindex + 1
        e  /=  len( y_a )
        #print( 'len( y_a ) :' , len( y_a ) ) 
        return e

    def learn(self , xa ,  y_a , learnRate ) :
        ya = self.forward( xa )
        e1 = self.squareErrorAvg( y_a , ya )
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
                    e2 = self.squareErrorAvg( y_a , ya )
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
                e2 = self.squareErrorAvg( y_a , ya )
                learnN = (e2-e1)/self.indepDelta * learnRate # (e2-e1)/self.indepDelta :loss函數的微分 
                self.wba[i][1][r] =  v
                bTemp[r]  =  v- learnN
            wbaTemp = [wTemp, bTemp , self.wba[i][2] ]
            wbaTemps.append( wbaTemp ) 
        #print ( '1:' , self.wba[0][0] - wbaTemps[0][0] )
        self.wba = wbaTemps
        #compute new error to return:
        ya = self.forward( xa )
        e = self.squareErrorAvg( y_a , ya )
        return e 

''' Main for  Leaned by myself '''
def directLearnMain() :
    xa = np.array( [[1,0,0,1],[0,1,0,0] , [0,1,0,1], [0,1,1,1]  ])
    y_a = np.array( [[1.,0.,0.],[0.,1.,0.] , [0.,0.,1.], [0.,1.0,0.]  ])
    if DEBUG01:
        print( 'xa:\n' , xa )
    act1 = signoid()
    #act1 = relu()
    wba=[[3,act1]]
    nn = neuNetwork( 4 , wba )
    ya = nn.forward(xa)
    if DEBUG01:
        print( 'ya type:' , type(ya) , 'ya[0] type' , type( ya[0] ) )
        print('ya:\n' ,  ya )
    for step in range( TRAIN_STEPS ) :
        e = nn.learn( xa ,  y_a ,0.01 )
        if step % 100 == 0 :
            print( 'step ', step , ': ' , e )
    nn.dumpWeightBias()
        

               
directLearnMain()        

