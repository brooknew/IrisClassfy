

def getDataFromDatafile():
    with open( '../data/iris.csv' , 'rt') as fr :   
        for i  in range(66) :
            buf = fr.readline()
            #print(i+1, ':' , buf ,end='' )

        dataGroup = []
        for i in range(150) :
            line = fr.readline()
            line = line[0:-1].split( ',' )
            line = line[0:-1]
            items = [] 
            for item in line:
                items.append(  float( item ) )
            #print( items )
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
                #print ( 'd ' , d ) 
                d *= d
                dis += d
            #print( i ,':', dis , t )
            i = i + 1
            dc = [ dis , t  ]
            distIrisClass.append( dc )
        distIrisClass.sort()
        for i in distIrisClass:
            print ( i )
        subl = distIrisClass[0:k]
        kind = {0:0,1:0,2:0}
        print( 'subl:' , subl )
        for it in subl:
            print( int( it[1][4]) )
            kind[ int(  it[1][4]) ] += 1
        #kind1 = [ v for v in sorted(kind.values())]
        print ( kind ) 
         
           
    
    

def main() :
    data = getDataFromDatafile()
    print( 'len of data:' , len( data  ))
    knns = knn( )
    knns.fit( data ) 
    knns.testData()
    knns.pred( 5 , [5.7,3.1,2.6,1.2] )


main()
