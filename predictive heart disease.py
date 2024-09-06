# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:07:31 2024

@author: sbsbs
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/sbsbs/OneDrive/Desktop/Machine Learning model/trained_model.sav','rb'))

input_data = (50,1,0,144,200,0,0,126,1,0.9,1,0,3)

#input data as turneed into the numpy array
input_data_as_numpy_array = np.asarray(input_data)

#numpy array data as turned into the reshaped
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction to the reshaped data
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

# we using the if statement

if (prediction[0]==0):
  print('the person as no heart disease')
else:
  print('the person as heart disease')  