
# encode all dumby variables
def prepare_data():
    # load train data
    train    = pd.read_csv('train.csv')
    test     = pd.read_csv('test.csv')
    labels   = train.Hazard
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train_ind = train.ix[:,'Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    Dropcols = ['T2_V10', 'T2_V7','T1_V13', 'T1_V10']
    train.drop( Dropcols, axis = 1, inplace=True )
    test.drop( Dropcols, axis = 1, inplace=True )

    catCols=['T1_V4', 'T1_V5','T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15',
     'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']

    trainNumX=train.drop(catCols, axis=1)
    trainCatVecX = pd.get_dummies(train[catCols])
    trainX = np.hstack((trainCatVecX,trainNumX))

    testNumX=test.drop(catCols, axis=1)
    testCatVecX = pd.get_dummies(test[catCols])
    testX = np.hstack((testCatVecX,testNumX))

    return trainX, labels, train_ind, testX, test_ind
