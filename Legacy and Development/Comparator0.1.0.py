#Project 3B Comparator VER 0.1.0
# Takes main framework from DuoNeuro Ver 1.1
# This is a custom-made, live learning capable, "back-driveable" neuron. It can either use minimally formatted data model or live data streams containing both 
# input and output data to train the weight and bias values, and is equipped with the function to generate both input and output data given the opposite half of the data model.
# Appends existing weight and bias values to a .txt file for easy access and lightweight storage.
# FUTURE UPDATE GENERAL NOTES: 
# Simplify the learning commands for both. Should be easy.
# Add node data deletion options
# Redundant calls for the same values as defined by variables needs to be replaced.

# Uses the sigmoid regression paradigm. Code from scratch. IN DEVELOPMENT. 

#The file version (this) is put in "toolbox" format, so once a user has access to this file, they can choose which portion of the program 
# they need and simply copy and paste over the necessary components as many times as needed.

#imports
import os
import numpy as np
#function definition
def normsigmoid(x):
    return 1/(1+np.exp(-x))

def normsigmoid_p(x):
    return normsigmoid(x,1) * (1-normsigmoid(x,1))

#exception class definition
class Error(Exception):
    pass

class shapeException(Error):
    pass

#variable definition
    cost = []
    cost_BD = []
    maximum_value_of_set = 0
    maximum_value_of_set_BD = 0

    number_of_weight_values = 0
    number_of_weight_values_and_bias = 0
    number_of_weight_bias_value_sets = 0
    index_of_bias_value = number_of_weight_values
    list_of_bias_and_weight_values = []

    number_of_weight_values_BD = 0
    number_of_weight_values_and_bias_BD = 0
    number_of_weight_bias_value_sets_BD = 0
    index_of_bias_value_BD = number_of_weight_values_BD
    list_of_bias_and_weight_values_BD = []

    alpha = 0.2

def Trainer(inputy, output, iterations):
    #filter 1: checks number of digits in all input and output values, finds the maximum figure. Makes the maximum figure also the maximum figure of the sigmoid function.
    maximum_value_of_set = max([sublist[-1] for sublist in output])
    maximum_value_of_set_BD = max([sublist[-1] for sublist in inputy])
    #filter 2: checks number of input and output values per point and how many points of data there are. 
    # If either matricies do not have the right length, throws exception as "DATA SHAPE INCONSISTENT" and terminates. 
    # If the shapes are consistent, it counts the number of output values per point and assigns new bias and weight values per output value.
    number_of_weight_values = len(inputy)
    number_of_weight_values_and_bias = number_of_weight_values + 1
    number_of_weight_bias_value_sets = len(inputy[0])

    index_of_bias_value = number_of_weight_values
    list_of_bias_and_weight_values = []

    #Tries to find a text file linked to the name of the neuron.
    #If it cannot, it will create a text file with the default name "Neuron-Data"
    #***FUTURE UPDATE***: Will account for multiple neurons without a txt datasheet activated all at once so as to not create different
    #instances of "Neuron-Data" in confusion.
    if(os.path.exists('Neuron-Data.txt')):
        datafile = open("Neuron-Data.txt","r")
        lineByLine = datafile.readlines()
        counter = 0
        for data in lineByLine:
            list_of_bias_and_weight_values[counter] = data
            counter += 1
        
    else:
        datafile = open("Neuron-Data.txt","w+")
        for a in (len(output[0])):
            for b in (number_of_weight_bias_value_sets + 1):
                list_of_bias_and_weight_values[b] = np.random.randn()
                datafile.write(list_of_bias_and_weight_values[b])
                datafile.write('\n')
    # Below is the same code from the above but used for the reverse process for back-driveability:
    # FUTURE UPDATE: Requires simplification and optimization. 
    # Preferrably have both paths done at the same time and have the data in the same file.
    number_of_weight_values_BD = len(output)
    number_of_weight_values_and_bias_BD = number_of_weight_values_BD + 1
    number_of_weight_bias_value_sets_BD = len(output[0])

    index_of_bias_value_BD = number_of_weight_values_BD
    list_of_bias_and_weight_values_BD = []

    if(os.path.exists('Neuron-Data_BD.txt')):
        datafile_BD = open("Neuron-Data_BD.txt","r")
        lineByLine = datafile.readlines()
        counter = 0
        for data in lineByLine:
            list_of_bias_and_weight_values_BD[counter] = data
            counter += 1
        
    else:
        datafile = open("Neuron-Data_BD.txt","w+")
        for a in (len(output[0])):
            for b in (number_of_weight_bias_value_sets_BD + 1):
                list_of_bias_and_weight_values_BD[b] = np.random.randn()
                datafile.write(list_of_bias_and_weight_values_BD[b])
                datafile.write('\n')


    #Training Loop: Per output value per point, the training loop sums the product of the weight and the input values with the bias and uses the output value as the actual value. 
    # The loop then uses each learning loop to change the bias and weight values within certain thresholds to provide accurate results when synthesizing information.
    # If "iterations" =! 0, it will only train the model for as many times it needs to.
    # If "iterations" == 0, it will automatically train the weights and biases until a sufficiently good "cost" value is reached.
    if(iterations != 0):
        for a in range(iterations):
            for b in len(output[0]):
                #finds a random data point from the input data
                ri = np.random.randint(len(inputy))
                point = inputy[ri]
                #z-sum: sum of the product of the weight values
                z = 0
                for i in (number_of_weight_bias_value_sets):
                    z += (point[i]*list_of_bias_and_weight_values[i]) 
                z += list_of_bias_and_weight_values[number_of_weight_values]
                #prediction & target value definition
                pred = max([sublist[-1] for sublist in output])*normsigmoid(z)
                target = output[ri][b]

                # cost for current random point
                cost[1] = np.square(pred - target)

                dcost_dpred = 2 * (pred - target)
                dpred_dz = maximum_value_of_set * normsigmoid_p(z)
                dcost_dz = dcost_dpred * dpred_dz
                
                #updates weight and bias values in the current runtime list
                for i in (number_of_weight_bias_value_sets + 1):
                    list_of_bias_and_weight_values[i] = list_of_bias_and_weight_values[i] - alpha * list_of_bias_and_weight_values[i] * dcost_dz

                #comparing cost values to auto-throttle the alpha value
                if(cost[1]>cost[0]):
                    alpha+=0.01
                if((cost[1]-cost[0])<-0.5):
                    alpha-=0.01
                cost[0] = cost[1]

                #***Start of the back-driveable portion***

            for a in len(inputy[0]):
                #finds a random data point from the output data
                ri = np.random.randint(len(output))
                point = output[ri]
                #z-sum: sum of the product of the weight values
                z = 0
                for i in (number_of_weight_bias_value_sets_BD):
                    z += (point[i]*list_of_bias_and_weight_values_BD[i]) 
                z += list_of_bias_and_weight_values_BD[number_of_weight_values_BD]
                #prediction & target value definition
                pred = max([sublist[-1] for sublist in inputy])*normsigmoid(z)
                target = inputy[ri][a]

                # cost for current random point
                cost[1] = np.square(pred - target)

                dcost_dpred = 2 * (pred - target)
                dpred_dz = maximum_value_of_set_BD * normsigmoid_p(z)
                dcost_dz = dcost_dpred * dpred_dz
                
                #updates weight and bias values in the current runtime list
                for i in (number_of_weight_bias_value_sets_BD + 1):
                    list_of_bias_and_weight_values_BD[i] = list_of_bias_and_weight_values_BD[i] - alpha * list_of_bias_and_weight_values_BD[i] * dcost_dz

                #comparing cost values to auto-throttle the alpha value
                if(cost[1]>cost[0]):
                    alpha+=0.01
                if((cost[1]-cost[0])<-0.5):
                    alpha-=0.01
                cost[0] = cost[1]
    if(iterations == 0):
        #FUTURE UPDATE: Make the cost limit based on the average cost of the last 100 to 200 iterations of the program.
        while (cost[0]>0.001 and cost_BD[0]>0.001):
            for b in len(output[0]):
                #finds a random data point from the input data
                ri = np.random.randint(len(inputy))
                point = inputy[ri]
                #z-sum: sum of the product of the weight values
                z = 0
                for i in (number_of_weight_bias_value_sets):
                    z += (point[i]*list_of_bias_and_weight_values[i]) 
                z += list_of_bias_and_weight_values[number_of_weight_values]
                #prediction & target value definition
                pred = max([sublist[-1] for sublist in output])*normsigmoid(z)
                target = output[ri][b]

                # cost for current random point
                cost[1] = np.square(pred - target)

                dcost_dpred = 2 * (pred - target)
                dpred_dz = maximum_value_of_set * normsigmoid_p(z)
                dcost_dz = dcost_dpred * dpred_dz
                
                #updates weight and bias values in the current runtime list
                for i in (number_of_weight_bias_value_sets + 1):
                    list_of_bias_and_weight_values[i] = list_of_bias_and_weight_values[i] - alpha * list_of_bias_and_weight_values[i] * dcost_dz

                #comparing cost values to auto-throttle the alpha value
                if(cost[1]>cost[0]):
                    alpha+=0.01
                if((cost[1]-cost[0])<-0.5):
                    alpha-=0.01
                cost[0] = cost[1]

            #***Start of the back-driveable portion***

            for a in len(output[0]):
                #finds a random data point from the output data
                ri = np.random.randint(len(output))
                point = output[ri]
                #z-sum: sum of the product of the weight values
                z = 0
                for i in (number_of_weight_bias_value_sets_BD):
                    z += (point[i]*list_of_bias_and_weight_values_BD[i]) 
                z += list_of_bias_and_weight_values_BD[number_of_weight_values_BD]
                #prediction & target value definition
                pred = max([sublist[-1] for sublist in inputy])*normsigmoid(z)
                target = inputy[ri][a]

                # cost for current random point
                cost_BD[1] = np.square(pred - target)

                dcost_dpred = 2 * (pred - target)
                dpred_dz = maximum_value_of_set_BD * normsigmoid_p(z)
                dcost_dz = dcost_dpred * dpred_dz
                
                #updates weight and bias values in the current runtime list
                for i in (number_of_weight_bias_value_sets_BD + 1):
                    list_of_bias_and_weight_values_BD[i] = list_of_bias_and_weight_values_BD[i] - alpha * list_of_bias_and_weight_values_BD[i] * dcost_dz

                #comparing cost values to auto-throttle the alpha value
                if(cost_BD[1]>cost_BD[0]):
                    alpha+=0.01
                if((cost_BD[1]-cost_BD[0])<-0.5):
                    alpha-=0.01
                cost_BD[0] = cost_BD[1]

#Influx: Uses a singular point of data in the same shape of the input points used in the model to synthesize an output value set.
def influx(inputy):
    for x in (number_of_weight_values):
        for i in (number_of_weight_bias_value_sets):
                z += (inputy[i]*list_of_bias_and_weight_values[i]) 
        z += list_of_bias_and_weight_values[number_of_weight_values]
        results[x] = maximum_value_of_set*normsigmoid(z)
    return results
#Efflux: Uses a singular point in the same shape of the original output data used in the model to synthesize a possible input value set.
def efflux(output):
    results = []
    for x in (number_of_weight_values_BD):
        for i in (number_of_weight_bias_value_sets_BD):
                z += (output[i]*list_of_bias_and_weight_values_BD[i]) 
        z += list_of_bias_and_weight_values_BD[number_of_weight_values_BD]
        results[x] = maximum_value_of_set_BD*normsigmoid(z)
    return results
