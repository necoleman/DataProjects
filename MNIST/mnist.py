import numpy as np
from sklearn import linear_model
import neural_net
import os

# These routines load the MNIST data and learn various algorithms on it
# They then let you save the predictions to a csv file in the appropriate
# format to upload to Kaggle.

# Loads data from the csv files provided by Kaggle
def loadData():
    raw_train_data = np.loadtxt( 'train.csv', dtype='string',delimiter=',' )[1:,:].astype(np.float)
    raw_test_data = np.loadtxt( 'test.csv', dtype='string',delimiter=',' )[1:,:].astype(np.float)
    #print raw_train_data[:,:3]
    return raw_train_data, raw_test_data

# Normalizes data so the input vectors lie between 0 and 1.
def Normalize(training_data):
    return training_data / 255.

# Formats the training data: Splits the loaded data into
# the features and the labels, then rewrites each label
# as a basis vector, e.g., 3 ---> [0,0,0,1,0,0,0,0,0,0]
def formatLabelToVector(training_data):
    # take the loaded training_data and split it into
    # the pixel data and the labels
    # set up the image classification vector
    img_id = training_data[:,0]
    pixel_data = list(training_data[:,1:])
    img_vec = []
    for k in range(img_id.shape[0]):
        vec = np.zeros(10)
        vec[int(img_id[k])] = 1.
        img_vec.append(vec)
    img_vec = np.array(img_vec)
    for j in pixel_data:
        j = Normalize(np.array(j))

    return zip(pixel_data,img_vec)

# Formats the predictions
# reinterprets the basis vector as a label
# e.g. [0,0,0,1,0,0,0,0,0,0] ---> 3
def formatVectorToLabel(data):
    # format predicted data
    # from a max-values-predicted to a list of labels
    # (conversion: index of max prediction is the label)
    submission = []
    for k in range(data.shape[0]):
        value = np.argmax(data[k])
        submission.append(value)
    submission = np.array(submission)
    return submission   

# Writes the predicted, assumed formatted, data to the predictions.csv file
def write(data):
    # print predictions to csv
    #os.remove('predictions.csv')
    with open('predictions.csv', 'a+') as f:
        f.write('ImageId,Label\n')
        it = 1
        for j in data:
            f.write(str(it)+','+str(int(j))+'\n')
            it+=1

# Runs a straight-up honest-to-god linear regression on the training and test data
# Returns a prediction on the test data
def NaiveRegression(training_data,test_data):
    pixel_data,img_vec = formatTrainingData(training_data)
    ols = linear_model.LinearRegression()
    ols.fit(pixel_data,img_vec)
    prediction = ols.predict(test_data)
    return formatData(prediction)

# Runs a SGD on the data and returns a prediction
def StochasticGradientDescent(pixel_data,img_labels,test_data):
    classifier = linear_model.SGDClassifier()
    classifier.fit(pixel_data,img_labels)
    prediction = classifier.predict(test_data)
    return prediction

# Runs a Perceptron on the test data and returns a prediction    
def Perceptron(pixel_data,img_labels,test_data):
    p = linear_model.Perceptron()
    p.fit(pixel_data,img_labels)
    return p.predict(test_data)

# Someday this may turn into a K-clustering algorithm
# with 10 clusters
def KNeighbors():
    return

# Runs a log regression
# (Rather, tries to. It doesn't converge at the moment, for some reason.)
def LogRegression(pixel_data,img_vec,test_data):
    logreg = linear_model.LogisticRegression(max_iter=1000,solver='sag',verbose=1)
    logreg.fit(pixel_data,img_vec)
    raw_pred = logreg.predict(test_data)
    return raw_pred

# Algorithm of the day: Homebrew neural net!
if __name__ == '__main__':
    train,test = loadData()
    training_data = formatLabelToVector(train)
    pixel_length = training_data[0][0].shape[0]
    N = neural_net.Network(3,pixel_length,10,0.01)
    print N.profile()


    print "     Now training ..."
    for j in range(5):
        N.train(training_data[:500],1)
    print "     Done training! Now validating ..."

    print N.validate(training_data[500:510])

    test_list = list(test)
    predictions = [ N.predict( np.array(j) ) for j in test_list ]
    predictions = formatVectorToLabel( np.array(predictions) )
    write(predictions)
    
