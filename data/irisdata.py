def getDataFromDatafile(n1, n2):
    '''out n2-n1 data items 
    each item of out data include:
    sepal length,sepal width,petal length,petal width,target . Example:
    5.4,         3.9,        1.3,         0.4,        0
    '''
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
def normalizeData( dataGroup ) :
    itemMax = [-1.,-1.,-1.,-1.]
    itemMin = [100.,100.,100.,100.]
    itemScale = [1.,1.,1.,1.]
    for  data in dataGroup :
        for  i in range( 4 )  :
            if itemMax[i] < data[i]  :
                itemMax[i] = data[i]
            if itemMin[i] < data[i]  :
                itemMin[i] = data[i]
    for  i in range( 4 ) :
        itemScale[i] = 1. /( itemMax[i] - itemMin[i]) 
    for  k in range( len(dataGroup) )  :
        for  i in range( 4 )  :
            dataGroup[k][i] -=  itemMin[i]
            dataGroup[k][i] *=  itemScale[i]
    #print( dataGroup )
    return dataGroup

         
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
            


