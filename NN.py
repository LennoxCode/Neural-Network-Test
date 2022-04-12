#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy
import scipy.special


# In[29]:


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #initalising the size of the network
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        #learning rate of the network
        self.lr = learning_rate
        #initializing weight matrix
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train():
        pass
    
    def query(self, inputs_list):
        #calculating the hidden layer
        inputs = numpy.array(inputs_list, ndmin=2).T
        print(inputs)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        
        
    


# In[30]:


n = neuralNetwork(3,3,3,0.3)
print(n.inodes)
print(n.wih)
print(n.who)
n.query([1.0, 0.5, -1.5])


# In[ ]:




