
def getDataFromDatafile(n1, n2):
    with open( '../data/irisRandom.csv' , 'rt') as fr :   
        for i  in range(66) :
            buf = fr.readline()
        dataGroup = []
        
        for i in range( n2 ) :
            line = fr.readline()
            if( i >= n1 ) : 
                line = line[0:-1].split( ',' )
                line = line[0:-1]
                items = [] 
                for item in line[:-1]:
                    items.append(  float( item ) )
                items.append( int( line[-1] ) )
                dataGroup.append( items )
        
        return dataGroup

class knn():
    def __init__(self) :
        self.data = []
    def fit( self , data ) :
        self.data = data
    def testData(self) :
        print( self.data )
    def pred( self ,k , item ) :
        distIrisClass = []
        i= 1
        for t in self.data :
            dis = 0 
            for tt1 , tt2 in zip ( t[0:-1] , item ): 
                d = tt1 - tt2 
                d *= d
                dis += d
            i = i + 1
            dc = [ dis , t  ]
            distIrisClass.append( dc )
        distIrisClass.sort()
        subl = distIrisClass[0:k]
        kind = {0:0,1:0,2:0}
        for it in subl:
            kind[ int(  it[1][4]) ] += 1
        max = 0
        maxK = -1
        for i , v  in kind.items() :
            if v  > max :
                max = v
                maxK = i
        return maxK 
         
def saveTest( record ) :
    ''' record format:
        sepal length,sepal width,petal length,petal width,target,pred target,pred right?
    '''
    with open( '../data/test.csv' , 'wt') as fw:
        fw.write('sepal length,sepal width,petal length,petal width,target,pred target,pred right?\n')
        for it in record:
            s = ''
            for i in it[:-1]:
                s += str(i)
                s += ','
            s += str(it[-1])
            s += '\n'
            fw.write( s ) 
            


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
    return good
    

def main() :
    data = getDataFromDatafile(0,112)
    knns = knn( )
    knns.fit( data ) 
    knns.pred( 5 , [5.7,3.1,2.6,1.2] )
    data = getDataFromDatafile(112,150)
    predSome( data , knns , 5  ) 
    


main()
