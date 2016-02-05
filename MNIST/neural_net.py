import numpy as np

# Here are some helper functions for what follows
def Sigmoid(t):
    return 1./(1.+np.exp(-t))

def DSigmoid(t):
    return Sigmoid(t)*(1-Sigmoid(t))

def DCost(t, outputs, trues):
    return (outputs - trues)

def ActivationFunction(t):
    return Sigmoid(t)

def DActivationFunction(t):
    return DSigmoid(t)

def InnerProduct(u,v):
    # take in two 1D arrays of the same shape
    U = np.reshape( u, (1,u.shape[0]) )
    V = np.reshape( v, (v.shape[0],1) )
    return U.dot(V)

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
        self.bias = 1.
        self.input_size = weights.shape
        self.learning_rate = epsilon
        return

    def adjustWeights(self,inputs,gradw,gradb):
        self.weights -= self.learning_rate * gradw
        self.bias -= self.learning_rate * gradb
        return

    def fire(self,inputs):
        return ActivationFunction( InnerProduct(self.weights, inputs) ) + self.bias

class Layer:
    # this class comprises a layer of neurons and handles both input/output
    # and backpropagation
    def __init__(self,num_inputs,num_outputs,learning_rate, is_last):
        self.neuron_list = [Neuron(RandomWeights(num_inputs),learning_rate) for k in range(num_outputs)]
        self.last_output = np.zeros(num_outputs)
        self.last_input = np.zeros(num_inputs)
        self.is_last = is_last
        self.learning_rate = learning_rate
        return
    
    def weightArray(self):
        return np.array( [n.weights for n in self.neuron_list] )

    def fire(self,inputs):
        self.last_input = inputs
        self.last_output = np.array([n.fire(inputs) for n in self.neuron_list])
        return self.last_output

    # perform backpropagation. return the critical values to calculate the error for the 
    # previous layer.
    # c.f. "the four key backpropagation equations" found here:
    # http://neuralnetworksanddeeplearning.com/chap2.html
    def backPropagate(self,next_errors):
        #print next_errors.shape
        #next_errors = np.reshape( next_errors, (1,next_errors.shape[0]) )
        this_errors = next_errors * self.last_output
        gradb = this_errors
        gradw = (self.last_input[np.newaxis].T).dot(this_errors[np.newaxis])
        error_to_hand_up = self.weightArray().T.dot(this_errors.T)
        for n in self.neuron_list:
            i = self.neuron_list.index(n)
            #print gradw.shape
            #print gradb.shape
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
        # network profile:
        # each hidden layer has (five?) neurons
        intermediate_num = 5
        initial_layer = Layer(self.input_size,intermediate_num,self.learning_rate,False)
        final_layer = Layer(intermediate_num,self.output_size,self.learning_rate,True)
        layers = [initial_layer]
        for k in range(self.num_layers - 1):
            layers.append(Layer(intermediate_num,intermediate_num,self.learning_rate,False))
        layers.append(final_layer)
        return layers

    def predict(self,inputs):
        prev_input = inputs
        for l in self.layers:
            print l
            prev_input = l.fire(prev_input)
        return prev_input

    def backPropagate(self,error):
        hand_back = error
        self.layers.reverse()
        for l in self.layers:
            hand_back = l.backPropagate(hand_back)
        self.layers.reverse()
        return

    def trainIndividually(self,trainer,label):
        for it in range(1000):
            print '\r'+str(it)
            self.backPropagate( self.predict(trainer) - label )
        return

    # trains the neural network
    # (assumes that data are in this format:
    # { (vector,label) }
    def train(self,training_data):
        for trainer,label in training_data:
            print "NEW TRAINING DATA"
            self.trainIndividually(trainer,label)
        return
