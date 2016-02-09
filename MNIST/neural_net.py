import numpy as np
import sys

# Here are some helper functions for what follows
def Sigmoid(t):
    x = 0.5
    if t < -40.:
        x = 0.0
    elif t > 40.:
        x = 1.0
    else:
        x = 1./(1.+np.exp(-t))
    return x

def DSigmoid(t):
    return Sigmoid(t)*(1-Sigmoid(t))

def DCost(t, outputs, trues):
    return (outputs - trues)

# Separating "activation function" from sigmoid
# To more easily allow for a different activation function
def ActivationFunction(t):
    return Sigmoid(t)

def DActivationFunction(t):
    return DSigmoid(t)

def InnerProduct(u,v):
    # take in two 1D arrays of the same shape
    U = np.reshape( u, (1,u.shape[0]) )
    V = np.reshape( v, (v.shape[0],1) )
    return (U.dot(V)).flatten()[0]

def OuterProduct(u,v):
    # take in two arbitrary 1D arrays
    U = np.reshape( u, (u.shape[0],1) )
    V = np.reshape( v, (1,v.shape[0]) )
    return U.dot(V)

# Backpropagation & error correction
# output and target are both 1d numpy arrays
def OutputDelta(output,target):
    return output * (np.ones(output.shape) - output) * (output - target)

def HiddenDelta(output,next_layer_delta):
    return output * (np.ones(output.shape) - output) * next_layer_delta.dot(  )

def RandomWeights(number):
    return np.random.rand(number)

# This class implements a simple neural net.
# Based on Ch 18 (p 217) of "Data Science from Scratch" by Joel Grus.
# The net can be instantiated and trained, then generates predictions.

class Neuron:
    # the neuron class knows itself, knows its inputs, and produces an output
    def __init__(self,weights,epsilon):
        self.weights = weights
        self.bias = 0.
        self.input_size = weights.shape[0]
        self.learning_rate = epsilon
        #print self.input_size
        return

    def adjustWeights(self,inputs,gradw,gradb):
        self.weights -= self.learning_rate * gradw
        self.weights / np.sum(self.weights)
        self.bias -= self.learning_rate * gradb
        return

    def fire(self,inputs):
        return ActivationFunction( InnerProduct(self.weights, inputs) ) + self.bias

class Layer:
    # this class comprises a layer of neurons and handles both input/output
    # and backpropagation
    def __init__(self,num_inputs,num_outputs,learning_rate, is_last):
        #print " Creating " + str(num_outputs) + " neurons!"
        self.neuron_list = [Neuron(np.ones(num_inputs)/num_inputs,learning_rate) for k in range(num_outputs)]
        self.last_output = np.zeros(num_outputs)
        self.last_input = np.zeros(num_inputs)
        self.is_last = is_last
        self.learning_rate = learning_rate
        return

    def __len__(self):
        return len(self.neuron_list)
    
    def weightArray(self):
        return np.array( [n.weights for n in self.neuron_list] )

    def fire(self,inputs):
        #print '     predicting!'
        self.last_input = inputs
        #print self.last_input
        self.last_output = np.reshape(np.array([n.fire(inputs) for n in self.neuron_list]),len(self.neuron_list))
        #print "         "+str(self.last_input.shape)
        #print "         "+str(self.last_output.shape)
        return self.last_output

    # perform backpropagation. return the critical values to calculate the error for the 
    # previous layer.
    # c.f. "the four key backpropagation equations" found here:
    # http://neuralnetworksanddeeplearning.com/chap2.html
    def backPropagate(self,next_errors):
        #print '         backpropagating through a layer ...'
        #next_errors = np.reshape( next_errors, (1,next_errors.shape[0]) )
        this_errors = next_errors * self.last_output
        if np.isnan(this_errors).any():
            print next_errors
            print self.last_output
            sys.exit()
        gradb = this_errors
        #print self.last_input.shape
        #print this_errors.shape
        gradw = OuterProduct(self.last_input,this_errors)
        error_to_hand_up = self.weightArray().T.dot(this_errors.T)
        for n in self.neuron_list:
            i = self.neuron_list.index(n)
            n.adjustWeights(self.last_input,gradw[:,i],gradb[i])
        return error_to_hand_up
        

class Network:

    def __init__(self,num_layers,input_size,output_size,learning_rate):
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layers = self.assembleNetwork()
        return

    def assembleNetwork(self):
        print self.input_size
        initial_layer = Layer(self.input_size,self.input_size,self.learning_rate,False)
        intermediate_num = (self.input_size + self.output_size)/2
        intermediate_layer = Layer(self.input_size,intermediate_num,self.learning_rate,False)
        final_layer = Layer(intermediate_num,self.output_size,self.learning_rate,True)
        layers = [initial_layer, intermediate_layer, final_layer]
        return layers

    def profile(self):
        return [len(l) for l in self.layers]

    def predict(self,inputs):
        prev_input = inputs
        for l in self.layers:
            #print "predicting in the " + str(l) + " layer"
            prev_input = l.fire(prev_input)
        return np.array(prev_input).reshape(self.output_size)

    # checks an individual input
    def checkPrediction(self,x,y):
        y_ = self.predict(x)
        return (np.argmax(y) == np.argmax(y_))

    def predictLabel(self,x):
        return np.argmax(self.predict(x))

    def stringValidation(self,x,y):
        y_ = self.predict(x)
        return str(np.argmax(y)) + ' = ' + str(np.argmax(y_))

    # checks a bunch of validation inputs for correctness
    def validate(self,validation_data):
        result_list = []
        for train,label in validation_data:
            if( self.checkPrediction(train,label) ):
                result_list.append(1.0)
            else:
                result_list.append(0.0)
        result_list = np.array(result_list)

        strlist = []
        for train,label in validation_data:
            strlist.append(self.stringValidation(train,label))
        print strlist
        return np.average(result_list)

    def backPropagate(self,error):
        #print "     backpropagating!"
        hand_back = error
        self.layers.reverse()
        for l in self.layers:
            hand_back = l.backPropagate(hand_back)
        self.layers.reverse()
        return

    def trainIndividually(self,trainer,label,num_it=0):
        #print "Training individually ..."
        X = self.predict(trainer)
        Y = label
        if num_it == 0:
            while( not self.checkPrediction(trainer,label) ):
                print "         predict: " + str(self.predictLabel(trainer))
                self.backPropagate( X - Y )
                X = self.predict(trainer)
                Y = label
                #print str(np.argmax(X)) + " = " + str(np.argmax(Y)) + "? " + "not there yet ..."
        else:
            for k in range(num_it):
                self.backPropagate( X-Y )
                X = self.predict(trainer)
                Y = label
        X = np.argmax(self.predict(trainer))
        Y = np.argmax(label)
        print str(X) + ' = ' + str(Y) + ': ' + str(np.argmax(self.predict(trainer)) == np.argmax(label) )
        return

    # trains the neural network with num_it iterations per training datum
    # (assumes that data are in this format:
    # { (vector,label) }
    def train(self,training_data,num_it):
        for trainer,label in training_data:
            #print "NEW TRAINING DATA"
            self.trainIndividually(trainer,label,num_it)
        return
