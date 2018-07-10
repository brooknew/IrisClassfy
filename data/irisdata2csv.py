from sklearn.datasets import load_iris

iris_dataset = load_iris()

with open( 'iris.csv' , 'wt') as fw :
    descr = iris_dataset['DESCR']
    t = descr.split('\n' )
    lines =  len( t )
    descr = descr + '------\n'
    content = descr 
    datasetItems = [ 'feature_names' , 'target_names' ]
    datasetItem_names = [ 'Features:' , 'Targets:' ]
    bFeat = True
    dataTitle = ''
    for datasetItem , datasetItem_name in zip ( datasetItems , datasetItem_names ) :
        items  = iris_dataset[ datasetItem ]
        t = ''  
        for item in items :
            t += item
            t += ','
        datasetItemLine = datasetItem_name + t[0:-1] + '\n'
        lines += 1
        if bFeat :
            dataTitle = t[0:-1]
            bFeat = False 
        content += datasetItemLine
        #print( datasetItemLine )
    dataTitle += ',target'
    dataTitle += ',target name\n'
    lines += 1 
    content = str( lines ) + ' lines :' + content + dataTitle 
    fw.write( content )
    feaD = iris_dataset['data']
    targD = iris_dataset['target']
    targetNames = iris_dataset['target_names']
    #print( targetNames )
    for f , t in zip( feaD, targD ) :
        linS= str(f[0])+','+ str(f[1])+','+ str(f[2])+',' + str(f[3])+','
        linS +=  str(t) + ',' +  targetNames[t]
        #print( linS )
        fw.write( linS+'\n' ) 



    
    
