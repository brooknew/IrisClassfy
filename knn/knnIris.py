'''KNN classifaction '''

import sys
sys.path.append('../data')

#import __init__
from irisdata import *

class knn():#'''KNN classificator'''
    def __init__(self) :
        self.data = []
    def fit( self , data ) :# '''train , data is samples to be trained'''
        self.data = data
    def testData(self) :
        print( self.data )
    def pred( self ,k , item ) :#'''k is the neighbor(s) , item is the feature tuple to be predicted'''
        distIrisClass = []
        for t in self.data :
            dis = 0 
            for tt1 , tt2 in zip ( t[0:-1] , item ): 
                d = tt1 - tt2 
                d *= d
                dis += d
            dc = [ dis , t  ]
            distIrisClass.append( dc )
        distIrisClass.sort()
        subl = distIrisClass[0:k]
        kind = {0:0,1:0,2:0}
        for it in subl:
            kind[ int( it[1][4]) ] += 1
        max = 0
        maxK = -1
        for i , v  in kind.items() :
            if v  > max :
                max = v
                maxK = i
        return maxK          

def predSome( data , knns , k  ) :
    res = []
    for da in data :
        pred = knns.pred(  k , da )
        da.append( pred )
        flag = pred == da[4]
        da.append( flag )
        res.append( da )
    good = 0. 
    for da in res :
        if da[6] :
            good += 1
    good1 = good/len( res )*100
    print ( good, ' right ',  good1 , '%' )
    saveTest( res ) 
    return good1
    

def main() :
    data = getDataFromDatafile(0,112)
    knns = knn( )
    knns.fit( data ) 
    knns.pred( 5 , [5.7,3.1,2.6,1.2] )
    data = getDataFromDatafile(112,150)
    r = predSome( data , knns , 5  ) 
    print( 'knn successfull rate:' , r )


main()
