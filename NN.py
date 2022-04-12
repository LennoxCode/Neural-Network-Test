#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy


# In[8]:


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #initalising the size of the network
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        #learning rate of the network
        self.lr = learning_rate
        #initializing weight matrix
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        pass
    
    def train():
        pass
    
    def query():
        pass
    


# In[9]:


n = neuralNetwork(3,3,3,0.3)
print(n.inodes)
print(n.wih)
print(n.who)


# In[ ]:




