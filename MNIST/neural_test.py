import numpy as np
import neural_net

def initializeTraining():
    zero = np.array([1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,1.,
                    1.,0.,0.,0.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])
    one = np.array([0.,0.,1.,0.,0.,
                    0.,0.,1.,0.,0.,
                    0.,0.,1.,0.,0.,
                    0.,0.,1.,0.,0.,
                    0.,0.,1.,0.,0.])
    two = np.array([1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,0.,
                    1.,1.,1.,1.,1.])
    three = np.array([1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])
    four = np.array([1.,0.,0.,0.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    0.,0.,0.,0.,1.])
    five = np.array([1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,0.,
                    1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])
    six = np.array([1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,0.,
                    1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])
    seven = np.array([1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    0.,0.,0.,0.,1.,
                    0.,0.,0.,0.,1.,
                    0.,0.,0.,0.,1.])
    eight = np.array([1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])
    nine = np.array([1.,1.,1.,1.,1.,
                    1.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.,
                    0.,0.,0.,0.,1.,
                    1.,1.,1.,1.,1.])

    return [zero,one,two,three,four,five,six,seven,eight,nine]
    

def initializeTesting():
    three = np.array([0.5,1.,1.,1.,0.7,
                    0.,0.,0.,0.5,1.,
                    0.,1.,1.,1.,1.,
                    0.,0.,0.,0.3,1.,
                    1.,1.,1.,1.,0.6])
    return three

if __name__ == '__main__':
    print "This is a test of the neural_net.py classes."
    numlist = initializeTraining()
    test = initializeTesting()
    training_data = []
    #print numlist
    for k in range(len(numlist)):
        train = numlist[k]
        output = np.zeros(10)
        output[k] = 1.
        training_data.append( (train,output) )
    N = neural_net.Network(3,25,10,0.1)

    print N.predict( training_data[0][1] )

    #N.train(training_data)

    test_data = test
    print test.shape
    print test
    print N.predict(test_data)
