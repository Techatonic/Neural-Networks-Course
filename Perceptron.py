'''
Following course called "Training Neural Networks in Python" on LinkedIn Learning
'''

import numpy as np

class Perceptron:
    def __init__(self, inputs, bias = 1.0):
        self.weights = (np.random.rand(inputs+1) * 2 - 1)
        self.bias = bias

    def run(self, x):
        sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(sum)

    def set_weights(self, w_init):
        self.weights = np.array(w_init)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
'''
# OR Gate
neuron = Perceptron(2)
neuron.set_weights([10, 10, -5])

print("Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0,0])))
print("0 1 = {0:.10f}".format(neuron.run([0,1])))
print("1 0 = {0:.10f}".format(neuron.run([1,0])))
print("1 1 = {0:.10f}".format(neuron.run([1,1])))
'''

class MultilayerPerceptron:
    def __init__(self, layers, bias = 1.0, eta = 0.5):
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []       # The list of lists of error terms
        self.eta = eta

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.d.append([])

            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))


        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)
    
    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()
    
    def run(self, x):
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]
    
    def bp(self, x, y):
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        # STEP 1: Feed a sample to the network
        outputs = self.run(x)

        # STEP 2: Calculate the MSE
        mse = 0.0
        for outputNum in range(len(outputs)):
            mse += (y[outputNum] - outputs[outputNum])**2
        mse /= self.layers[-1]
        

        # Could've been done using numpy vectorised operations

        # STEP 3: Calculate the output error terms
        for outputNum in range(len(outputs)):
            self.d[-1][outputNum] = outputs[outputNum] * (1-outputs[outputNum]) * (y[outputNum] - outputs[outputNum])

        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1, len(self.network)-1)): # Iterates through layers
            for h in range(len(self.network[i])):         # Iterates through Perceptrons in each layer
                fwd_error = 0.0
                for k in range(self.layers[i+1]):         # Iterates through neurons in each layer
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # STEP 5 & 6: Calculate the deltas and update the weights
        for i in range(1, len(self.network)):                  # Goes through the layers
            for j in range(self.layers[i]):                    # Goes through the neurons
                for k in range(self.layers[i-1]+1):            # Goes through the inputs (in the previous layer)
                    # Couldn't do - had to follow solution
                    if k == self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
    
        return mse




#test code

mlp = MultilayerPerceptron(layers=[7,7,10])
epochs = 3000
print("\nTraining Neural Network for the display...\n")
for i in range(epochs):
    MSE = 0.0
    MSE += mlp.bp([1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #0
    MSE += mlp.bp([0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) #1
    MSE += mlp.bp([1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) #2
    MSE += mlp.bp([1, 1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) #3
    MSE += mlp.bp([0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) #4
    MSE += mlp.bp([1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) #5
    MSE += mlp.bp([1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) #6
    MSE += mlp.bp([1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) #7
    MSE += mlp.bp([1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) #8
    MSE += mlp.bp([1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) #9
    MSE = MSE / 10
    if(i%100 == 0):
        print (MSE)

mlp.printWeights()
    
print("MLP:")
print ("1 1 1 1 1 1 0 = " + str(mlp.run([0.66, 0.44, 0.47, 0.47, 0.01, 0.01, 0.28])))
#print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
#print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
#print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))
