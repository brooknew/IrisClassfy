'''neural network classifcation'''




def main() :
    data = getDataFromDatafile(0,112)
    knns = knn( )
    knns.fit( data ) 
    knns.pred( 5 , [5.7,3.1,2.6,1.2] )
    data = getDataFromDatafile(112,150)
    r = predSome( data , knns , 5  ) 
    print( 'knn successfull rate:' , r )


main()
