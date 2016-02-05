import numpy as np
from sklearn import linear_model
import os

def loadData():
    raw_train_data = np.loadtxt( 'train.csv', dtype='string',delimiter=',' )[1:,:].astype(np.float)
    raw_test_data = np.loadtxt( 'test.csv', dtype='string',delimiter=',' )[1:,:].astype(np.float)
    #print raw_train_data[:,:3]
    return raw_train_data, raw_test_data

def Normalize(training_data):
    return training_data / 255.

def formatTrainingData(training_data):
    # take the loaded training_data and split it into
    # the pixel data and the labels

    # set up the image classification vector
    img_id = training_data[:,0]
    pixel_data = training_data[:,1:]
    img_vec = []
    for k in range(img_id.shape[0]):
        vec = np.zeros(10)
        vec[int(img_id[k])] = 1.
        img_vec.append(vec)
    img_vec = np.array(img_vec)
    return pixel_data,img_vec

def formatData(data):
    # format predicted data
    # from a max-values-predicted to a list of labels
    # (conversion: index of max prediction is the label)
    submission = []
    for k in range(data.shape[0]):
        value = np.argmax(data[k])
        submission.append(value)
    submission = np.array(submission)
    return submission   

def NaiveRegression(training_data,test_data):
    pixel_data,img_vec = formatTrainingData(training_data)
    ols = linear_model.LinearRegression()
    ols.fit(pixel_data,img_vec)
    prediction = ols.predict(test_data)
    return formatData(prediction)

def StochasticGradientDescent(pixel_data,img_labels,test_data):
    classifier = linear_model.SGDClassifier()
    classifier.fit(pixel_data,img_labels)
    prediction = classifier.predict(test_data)
    return prediction
    
def Perceptron(pixel_data,img_labels,test_data):
    p = linear_model.Perceptron()
    p.fit(pixel_data,img_labels)
    return p.predict(test_data)

def KNeighbors():
    return

def LogRegression(pixel_data,img_vec,test_data):
    logreg = linear_model.LogisticRegression(max_iter=1000,solver='sag',verbose=1)
    logreg.fit(pixel_data,img_vec)
    raw_pred = logreg.predict(test_data)
    return raw_pred

def write(data):
    # print predictions to csv
    #os.remove('predictions.csv')
    with open('predictions1.csv', 'a+') as f:
        f.write('ImageId,Label\n')
        it = 1
        for j in data:
            f.write(str(it)+','+str(int(j))+'\n')
            it+=1

if __name__ == '__main__':
    train,test = loadData()
    pixel_data,img_labels = train[:,1:],train[:,0]
    pixel_data = Normalize(pixel_data)
    prediction = Perceptron(pixel_data,img_labels,test)
    write(prediction)
