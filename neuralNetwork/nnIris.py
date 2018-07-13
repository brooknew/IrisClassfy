#coding:utf-8
import numpy as np

BATCH_SIZE = 8
TRAIN_STEPS = 1000

class knnNeuNetClassify（） ：
    def __init__(self ,  inDim , w_b_a ) :
        '''inDim =>Dimension of input
           n_a =>list of tuple ( nodes , activation ) of each layer
        '''
        self.inDim = inDim
        self.n_a = n_a
        self.w = []
        self.b = []
        self.act = []
        lastDim = inDim
        for i in  n_a :
            self
    def matrixCopy( self , mSrc , mDst ) :
        for  r  in   range( len( mSrc ) )  :
            for c in  range ( len ( mSrc[r] ) ) :
                mDst[r][c] = mSrc[r][c]
    def forwardnn( self , xa , w1 , w2 ,b1 , b2 , ya) :
        i = 0 
        for  x in xa :
            a1 = np.dot( x , w1 )
            a1 += b1
            y = np.dot( a1 , w2 )
            y += b2
            ya[ i ] = y
            i = i + 1
    def squareErrorAvg( y_a , ya ) :
        e = 0.0
        yindex = 0
        for  y_    in   y_a  :
            y = ya[yindex]
            e1 = (y_[0]- y[0]) * (y_[0]- y[0])
            e  += e1
            yindex = yindex + 1
        e  /=  yindex
        return e 

    def  learn( layer ,  w1, w2 , b1 , b2 , wnew ,  xa , ya , y_a , learnRate , wDelta  ) :
        forwardnn( xa ,  w1 , w2 ,b1,b2, ya )
        e1 = squareErrorAvg( y_a , ya )
        if layer == 1 :
            w = w1
        else:
            w = w2 
        for  r  in   range( len( wnew ) )  :
            for c in  range ( len ( wnew[r] ) ) :
                v = w[r][c]
                w[r][c] += wDelta
                forwardnn( xa ,  w1 , w2 , b1 ,b2 , ya )
                e2 = squareErrorAvg( y_a , ya )
                learnN = (e2-e1)/wDelta * learnRate # (e2-e1)/wDelta :loss函數的微分 
                w[r][c] =  v
                wnew[r][c]  =  v- learnN
                
    def  learnB( layer ,  w1, w2 , b1 ,b2 , bnew ,  xa , ya , y_a , learnRate , wDelta  ) :
        forwardnn( xa ,  w1 , w2 ,b1,b2, ya )
        e1 = squareErrorAvg( y_a , ya )
        if layer == 1 :
            b = b1
        else:
            b = b2     
        for c in  range ( len ( bnew) ) :
            v = b[c]
            b[c] += wDelta
            forwardnn( xa ,  w1 , w2 , b1, b2 , ya )
            e2 = squareErrorAvg( y_a , ya )
            learnN = (e2-e1)/wDelta * learnRate # (e2-e1)/wDelta :loss函數的微分 
            b[c] =  v
            bnew[c]  =  v- learnN

    def calAccury(  ya ) : # ya is list to predicted result   
        ya = np.array( ya )
        y_a = Y_T
        y_a = np.array( y_a ) 
        accury =np.array(ya - y_a)
        a = ( abs( accury ) < TRAIN_THRESHOLD )
        af = a.astype( np.float32 )
        right = af.sum()
        per = right/VALIDATE_SIZE 
        return per , a 

    def directValidateMain(w1 , w2 , b1 ,b2 ):
        xa = XT[0:VALIDATE_SIZE]
        ya =  [[0.0]]* VALIDATE_SIZE
        forwardnn(xa, w1 , w2 , b1 ,b2 , ya)
        per  , a  = calAccury(  ya )
        for i in range( len( ya ) ):
            print ( xa[i] , '=>'  , ya[i] , '||', Y_T[i] , a[i]  )          
        return per 

''' Main for  Leaned by myself '''
def directLearnMain() :
    w1= np.array( [[1.0,1.0,1.0],[2.0,1.0,1.0]])
    w2= np.array([[2.0],[3.0],[5.0]])
    #w1= np.array(  [[0.1,0.5,0.0],[0.0,0.0,0.0]] ) 
    #w2= np.array( [[0.05],[0.01],[0.05]] ) 
    b1=[0.0,0.0,0.0]
    b2=[0.0]

    Y = [    [0.0] ]* SAMPLE_SIZE 
         
    STEPS =TRAIN_STEPS
    w1new = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    w2new= [[0.0],[0.0],[0.0]]
    b1new=[0.0,0.0,0.0]
    b2new=[0.0]

    for i in range(STEPS):
        start =  (i*BATCH_SIZE) % SAMPLE_SIZE 
        end =  start + BATCH_SIZE
        xa = X[start:end]
        ya = Y[start:end]
        y_a = Y_[start:end]
        learn( 1 ,  w1, w2 , b1,b2, w1new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learn( 2 ,  w1, w2 , b1,b2 , w2new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learnB( 1 ,  w1, w2 , b1,b2, b1new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learnB( 2 ,  w1, w2 , b1,b2, b2new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        matrixCopy( w1new ,    w1)
        matrixCopy( w2new ,    w2)
        #print('w1=' , w1 )
        #print('w2=' , w2 ) 
        b1= b1new[:]
        b2=b2new[:]
        if  (i +1)% 10000 == 0 :
            forwardnn( xa ,  w1 , w2 , b1,b2 , ya )
            e = squareErrorAvg( y_a  , ya )
            print ('w1 learned  by myself @time(s)  ' , i+1 , '  : ' ,  w1new )
            print ('w2 learned  by myself @time(s)  ' ,  i +1, '  : '  ,  w2new )
            print('loss  by myself @time(s)  ' ,  i +1, '  : '   , e) 
    print( "训练结果(myself)：" )
    print("w1:", w1 )
    print("w2:", w2 )
    print( "b1" , b1 )
    print("b2" ,b2)
    per  = directValidateMain( w1 , w2  , b1 ,b2)
    print("Accury by myslef:" , per ) 
    return w1,w2, b1 , b2
               
w11 , w12 , b11 , b12 = directLearnMain()        

