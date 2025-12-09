# //// MACHINE LEARNING ////

#From three blue one browns video series on deeplearning

#Basicly the formula for setting up a neural network is:
#   a^(1) = sigmoid( W * a^(0) + b)

# a^(1) : represents a vector of the next layer of neurals
# a^(0) : represents a vector of the starting neurals (e.x. all pixels in a picture)
# W     : A matrix where each row represents the weight of each a^(0) node and its
#         connection to 1 node in the next layer
# b     : Is the bias, it says how high the weighted sum needs to be for the neuron to
#         become meaningfuly active.

# To do the actual learning we need to implement a cost function, that
# tells us how of target the result is, and by taking the gradient of that cost function
# to find in which way we need to adjust the weights so that we minimize the result given by
# the cost function.
#
#  

# Will try it on the MNIST set, the pictures are 

import math
import random
import numpy as np


#Only want to retrive the MNIST dataset
from torchvision import datasets




#////////////////// RETRIVING TRAINING DATA /////////////////////////////

#Loads training data (60 000 samples)
mnist_train = datasets.MNIST(root="./MNIST", train=True, download=True)

#Loads data for testing (10 000 samples)
mnist_test = datasets.MNIST(root="./MNIST", train=False, download=True)
    

#Converting an image into an nparray

#Img is the image, and label is the number depicted in the image
#img, label = mnist_test[0]
#np.array(img)
# ------------------------------------------------------------------    

# Used to normalise the values
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Derivate used for back propagation
def sigmoidDerivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


def softMax(z):
    e = np.exp(z - np.max(z))  # subtract max for numerical stability
    return e / e.sum()



#Height will represent the amount of nodes in the hidden layer
#The number of colums will be 28*28=784
def createWeightMatrix(height, width):
    #creates a 2D numpy array with random float values
    return np.random.rand(height, width)


#Returns index containing largest value in the output vector
def findHighestVal(outputVector):
    return np.argmax(outputVector)

#Calculating the cost, each element in the output vector (output - 0)^2 execpt for the one on the correct index, then: (output- 1)^2
def cost(outputVector, answer):
    sum = 0
    for i in range(len(outputVector)):
        if(i == answer):
            sum += math.pow((outputVector[i] - 1),2)
        else:
            sum += math.pow((outputVector[i]), 2)
    return sum



#//////////// BACK PROPAGATION /////////////////////////

#This net has one hiden layer, so in totalt we will need to calculate 2 cost gradients

# ---------- FIRST LAYER -----------------------
# FROM OUTPUT LAYER -> HIDDEN LAYER
# dC/dW = e * (a)^T and dC/db = e           (e represents the error)
# where e = dC/dz = dC/da * sigmoid' (z)

# outputVec : a(l)
# zTerm     : z(l)
# desiredVec: y



def outPutLayerErrorTerm(outputVec, zTerm, desiredVec):
    return sigmoidDerivate(zTerm) * (2*(outputVec-desiredVec))

def hiddenLayerErrorTerm(nextLayerWeight, prevLayerError, zTerm):
    return ((nextLayerWeight).T @ prevLayerError) * sigmoidDerivate(zTerm)

# eTerm : d
# prevLayerVec : a(l-1)
#Returns the costGradient which contains the weight partial matrix and bias partial vector
def costGradient(eTerm, prevLayerVec):
    # Make both vectors column vectors for outer product
    eTerm_col = eTerm.reshape(-1, 1)           # shape: (neurons_current_layer, 1)
    prevLayerVec_col = prevLayerVec.reshape(1, -1)  # shape: (1, neurons_prev_layer)

    # Outer product to get weight gradient matrix
    weightPartial = eTerm_col @ prevLayerVec_col  # shape: (neurons_current_layer, neurons_prev_layer)

    # Bias gradient is just the error term
    biasPartial = eTerm_col.squeeze()  # keep as 1D vector

    return weightPartial, biasPartial




#Updates the weightMatrix, make sure to call in correct layer
def updateWeight(weightPartial, currentWeight, learningRate):
    return currentWeight - learningRate * weightPartial

def updateBias(biasPartial, currentBias, learningRate):
    return currentBias - learningRate * biasPartial



def main():


    LEARNING_RATE = 0.01
    HIDDEN_LAYER_SIZE = 128
    #Creating weights
    #Input layer
    W0 = np.random.rand(HIDDEN_LAYER_SIZE, 784) * np.sqrt(1 / 784) # 128 x 784     Will be multiplied with the input vector (784 x 1)

    #First hidden layer, starting out with 1
    W1 = np.random.rand(10, HIDDEN_LAYER_SIZE) * np.sqrt(1/128) # 10 x 128        Will be multiplied with the hidden layer vector (16x1) to reach the (9x1) output vector 

    #Don't really know what these should be
    L0_BIAS = np.full(HIDDEN_LAYER_SIZE, 0.1)
    L1_BIAS = np.full(10, 0.1)


    # ---- TRAINING LOOP ------------
    EPOCH = 5

    print("Training...\n")
    for epoch in range(EPOCH):

        print("Starting epoch: \n", epoch)
        hits = 0
        for i in range(len(mnist_train)):
            #Loads the input vector
            img, label = mnist_train[i]
            #Flatten out the array (when just loading the image it becomes a 28x28 matrix)
            input = np.array(img).reshape(-1) / 255

            #Creating the hidden layer
            Z0 = W0 @ input + L0_BIAS
            A1 = sigmoid(Z0)

            #Creating the output layer
            Z1 = W1 @ A1 + L1_BIAS
            A2 = softMax(Z1)

            #Creating desired vector:
            desiredVec = np.full(10, 0)
            desiredVec[label] = 1

            outputEterm = A2 - desiredVec
            hiddenEterm = hiddenLayerErrorTerm(W1, outputEterm, Z0)

            c2Gradient = costGradient(outputEterm, A1)
            c1Gradient = costGradient(hiddenEterm, input)

            W1 = updateWeight(c2Gradient[0], W1, LEARNING_RATE)
            W0 = updateWeight(c1Gradient[0], W0, LEARNING_RATE)

            L1_BIAS = updateBias(c2Gradient[1], L1_BIAS, LEARNING_RATE)
            L0_BIAS = updateBias(c1Gradient[1], L0_BIAS, LEARNING_RATE)

            
            if(findHighestVal(A2) == label):
                hits += 1
            print(f"Cost during training: {cost(A2, label):.4f}  Hitrate during training: {hits / (i+1):.4f}    ", end='\r', flush=True)


    
    

    # ----- TESTING ----------
    print("Done training! Testing: \n")
    hits = 0

    for i in range(len(mnist_test)):
        #Loads the input vector
        img, label = mnist_test[i]
        #Flatten out the array (when just loading the image it becomes a 28x28 matrix)
        input = np.array(img).reshape(-1)

        #Creating the hidden layer
        Z0 = W0 @ input + L0_BIAS
        A1 = sigmoid(Z0)

        #Creating the output layer
        Z1 = W1 @ A1 + L1_BIAS
        A2 = softMax(Z1)

        print(f"Final cost: {cost(A2, label):.4f} Final hit rate: {hits / (i+1):.4f}    ", end='\r', flush=True)
        
        if(findHighestVal(A2) == label):
            hits += 1



main()











