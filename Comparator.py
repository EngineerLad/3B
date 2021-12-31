'''Project 3B Comparator VER 1.0.0 RELEASE FILE Last Update: 12/30/2021
 Takes main framework from DuoNeuro Ver 1.1. Original Repository MACC.

 This is a custom-made, live-learning capable, "back-driveable" neuron. It can either use minimally formatted data models or live data streams containing both 
 input and output data to train the weight and bias values, and is equipped with the function to generate both input and output data given the opposite half of the data model.
 Appends existing weight and bias values to a .txt file for easy access and lightweight storage.

 DATA INPUTS

 DO NOT USE FOR EXPLICIT TIME DATA. (Project DeLorean in development for this purpose)

 Uses the sigmoid regression paradigm. Code from scratch. CURRENTLY IN DEVELOPMENT.'''

import os
import numpy as np
import random 
import ast

class ComparatorNode():
    def __init__(self,instanceNumber):
        self.instance = instanceNumber
    variables = {
    # List Variable Definitions
    #   Cost Lists: Keeps track of the dual cost points formed by the data shape.
    "cost" : [],
    "cost_BD" : [],
    #   Internal Bias and Weight Value lists: Internally kept list of coefficients used and moderated by the node's core, 
    #       written to and read from an externally kept .TXT file for safekeeping,
    "list_of_bias_and_weight_values" : [],
    "list_of_bias_and_weight_values_BD" : [],
    #   Maximum Values Lists: Lists of maximum values to help change the maximum of the sigmoid function to suit calculating data of varying magnitude.
    "maxValuesListi" : [],
    "maxValuesListo" : [],
    #   Alpha list: A list of alpha values, or "delta threshold constants," kept for each in and output data.
    "alpha" : [],
    "alpha_BD" : [],

    # Constant Variable Definitions
    #   In/Output count: Keeps track of the total number of variables that need to be dealt with.
    "number_of_inputs" : 0,
    "number_of_outputs" : 0,
    #   Maximum Z Sum values: Calculates the Z sum to cap the sigmoid function's input value 
    #       to shape the function to better handle a wider or shorter range of Z sum combinations.
    "maximumZ" : 0,
    "maximumZ_BD" : 0,
    }
    #   Trainer: The condensed version of the previous iteration of the Comparator Program. Takes input, output, and an iteration count(0 if training to threshold) 
    #   to change the weight and bias values and train.
    def Trainer(self, inputy, output, iterations):
        # Variable Call
        number_of_inputs = self.variables["number_of_inputs"]
        number_of_outputs = self.variables["number_of_inputs"]
        cost = self.variables["cost"]
        cost_BD = self.variables["cost_BD"]
        alpha = self.variables["alpha"]
        alpha_BD = self.variables["alpha_BD"]
        maxValuesListi = self.variables["maxValuesListi"]
        maxValuesListo = self.variables["maxValuesListo"]
        maximumZ = self.variables["maximumZ"]
        maximumZ_BD = self.variables["maximumZ_BD"]
        list_of_bias_and_weight_values = self.variables["list_of_bias_and_weight_values"]
        list_of_bias_and_weight_values_BD = self.variables["list_of_bias_and_weight_values_BD"]
        
        #   Novel Sigmoid Functions: Allows for input parameter range to be expanded.
        def normsigmoid(x,b):
            maximumDXvalue = (2*(np.exp(2)))/b
            return 1/(1+np.exp(-x*maximumDXvalue))

        def normsigmoid_p(x,b):
            maximumDXvalue = (2*(np.exp(2)))/b
            return (maximumDXvalue)*(normsigmoid(x,b) * (1-normsigmoid(x,b)))

        def TrainingLoopCORE(inputy,output):
            #Loops for each output variable
            for a in range(number_of_outputs):
                #z sum reset
                z = 0
                #summation of the input value and the associated weight value
                for i in range(number_of_inputs):
                    z += (float(inputy[i])*float(list_of_bias_and_weight_values[a][i])) 
                #adds bias value
                z += float(list_of_bias_and_weight_values[a][-1])

                pred = maxValuesListi[a] * normsigmoid(z,maximumZ)
                target = output[a]
                #the difference is divided by the maximum value to reduce noise
                cost[a][1] = (np.square((pred - target)/maxValuesListi[a]))

                dcost_dpred = 2 * ((pred - target)/maxValuesListi[a])
                dpred_dz = maxValuesListi[a]*normsigmoid_p(z,maximumZ)
                dcost_dz = dcost_dpred * dpred_dz
                for i in range(number_of_inputs):
                    list_of_bias_and_weight_values[a][i] = float(list_of_bias_and_weight_values[a][i]) - alpha[a] * dcost_dz * (inputy[i]/maxValuesListi[a])
                list_of_bias_and_weight_values[a][-1] = float(list_of_bias_and_weight_values[a][-1]) - alpha[a] * dcost_dz
                #internal alpha adjustment: if the cost is somehow larger than the previous, it increases the alpha value. 
                # If the average change in cost is more than the new cost, it decreases the alpha value.
                if(cost[a][1]>cost[a][0]):
                    alpha[a]+=0.01
                if(((cost[a][0]-cost[a][1])/2)>cost[a][1]):
                    alpha[a]-=0.01
                cost[a][0] = cost[a][1]
            #this is the same as the above but the reverse process to enable backdriveability
            for a in range(number_of_inputs):
                z = 0
                for i in range(number_of_outputs):
                    z += (float(output[i])*float(list_of_bias_and_weight_values_BD[a][i])) 
                z += float(list_of_bias_and_weight_values_BD[a][-1])
                pred = maxValuesListo[a] * normsigmoid(z,maximumZ_BD)
                target = inputy[a]
                cost_BD[a][1] = (np.square((pred - target)/maxValuesListo[a]))
                dcost_dpred = 2 * ((pred - target)/maxValuesListo[a])
                dpred_dz = maxValuesListo[a]*normsigmoid_p(z,maximumZ_BD)
                dcost_dz = dcost_dpred * dpred_dz
                for i in range(number_of_outputs):
                    list_of_bias_and_weight_values_BD[a][i] = float(list_of_bias_and_weight_values_BD[a][i]) - alpha_BD[a] * dcost_dz * (output[i]/maxValuesListo[a])
                list_of_bias_and_weight_values_BD[a][-1] = float(list_of_bias_and_weight_values_BD[a][-1]) - alpha_BD[a] * dcost_dz
                if(cost_BD[a][1]>cost_BD[a][0]):
                    alpha_BD[a]+=0.01
                if(((cost_BD[a][0]-cost_BD[a][1])/2)>cost_BD[a][1]):
                    alpha_BD[a]-=0.01
                cost_BD[a][0] = cost_BD[a][1]

        #helps differenciate input types from list and point and sets the value of some variables
        if isinstance(inputy[0], list):
            number_of_inputs = len(inputy[0])
            maxValuesListi = [max(elem) for elem in zip(*output)]
            maximumZ = sum([max(elem) for elem in zip(*output)])
            
        else:
            number_of_inputs = len(inputy)

        if isinstance(output[0], list):
            number_of_outputs = len(output[0])
            maxValuesListo = [max(elem) for elem in zip(*inputy)] 
            maximumZ_BD = sum([max(elem) for elem in zip(*inputy)])   
        else:
            number_of_outputs = len(output)


        templisti = []
        templisto = []

        if(os.path.exists('Neuron-DataTESTINGNODE.txt')):
            datafile = open("Neuron-DataTESTINGNODE.txt","r")
            lineByLine = datafile.readlines()
            for data in lineByLine:
                templisti.append(data.strip())
                if(len(templisti) == (number_of_inputs + 1)):
                    list_of_bias_and_weight_values.append(templisti)
                    templisti = []
            datafile.close()
            
        else:
            datafile = open("Neuron-DataTESTINGNODE.txt","w+")
            for a in range(number_of_outputs):
                templisti = []
                for b in range(number_of_inputs + 1): 
                    templisti.append(np.random.uniform(-1,1))
                    datafile.write(str(templisti[b])+"\n")
                list_of_bias_and_weight_values.append(templisti)
            datafile.close()

        if(os.path.exists('Neuron-Data_BDTESTINGNODE.txt')):
            datafile_BD = open("Neuron-Data_BDTESTINGNODE.txt","r")
            lineByLine2 = datafile_BD.readlines()
            for data in lineByLine2:
                templisto.append(data.strip())
                if(len(templisto) == (number_of_outputs + 1)):
                    list_of_bias_and_weight_values_BD.append(templisto)
                    templisto = []
            datafile_BD.close()
                        
        else:
            datafile_BD = open("Neuron-Data_BDTESTINGNODE.txt","w+")
            for a in range(number_of_inputs):
                templisto = []
                for b in range(number_of_outputs + 1): 
                    templisto.append(np.random.uniform(-1,1))
                    datafile_BD.write(str(templisto[b])+"\n")
                list_of_bias_and_weight_values_BD.append(templisto)
            datafile_BD.close()

        for j in range(number_of_inputs):
            cost.append([10,10])
            alpha.append(0.2)
        for k in range(number_of_outputs):
            cost_BD.append([10,10])
            alpha_BD.append(0.2)
        
        #This series of conditions makes sure the program can still train models where both are lists, where only the input(or output) is a point, 
        # and when both input and output are points.
        if (isinstance(inputy[0], list) and isinstance(output[0], list)): 
            if (iterations == 0):
                while(float(sum([sum(i) for i in zip(*cost)])) > 0.00001 and float(sum([sum(i) for i in zip(*cost_BD)])) > 0.00001):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy[ri],output[ri])
            else:
                for a in range(iterations):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy[ri],output[ri])
        elif (isinstance(inputy[0], list) and (isinstance(output[0], float) or isinstance(output[0], int))): 
            if (iterations == 0):
                while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy[ri],output)
            else:
                for a in range(iterations):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy[ri],output)
        elif (isinstance(output[0], list) and (isinstance(inputy[0], float) or isinstance(inputy[0], int))): 
            if (iterations == 0):
                while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and  float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy,output[ri])
            else:
                for a in range(iterations):
                    ri = random.randint(0,len(inputy)-1)
                    TrainingLoopCORE(inputy,output[ri])
        else:
            if (iterations == 0):
                while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and  float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                    TrainingLoopCORE(inputy,output)
            else:
                for a in range(iterations):
                    TrainingLoopCORE(inputy,output)
        #these variables are getting saved into each instance's values because they are persistent values necessary in defining the model that has been trained.
        self.variables["cost"] = []
        self.variables["cost_BD"] = []
        self.variables["number_of_inputs"] = number_of_inputs
        self.variables["number_of_outputs"] = number_of_outputs
        self.variables["maximumZ"] = maximumZ
        self.variables["maximumZ_BD"] = maximumZ_BD
        self.variables["maxValuesListi"] = maxValuesListi
        self.variables["maxValuesListo"] = maxValuesListo
        #this bit saves the bias and weight values onto txt files.
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        flat_bwlist = [item for sublist in list_of_bias_and_weight_values for item in sublist]
        for a in range(len(list_of_bias_and_weight_values) * len(list_of_bias_and_weight_values[0])):
            datafile.write(str(float(flat_bwlist[a]))+"\n")

        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        flat_bw_BDlist = [item for sublist in list_of_bias_and_weight_values_BD for item in sublist]
        for a in range(len(list_of_bias_and_weight_values_BD) * len(list_of_bias_and_weight_values_BD[0])):
            datafile.write(str(float(flat_bw_BDlist[a]))+"\n")

        
    #Influx: Uses a singular point of data in the same shape of the input points used in the model to synthesize an output value set.
    def influx(self, inputy):
        number_of_inputs = self.variables["number_of_inputs"]
        number_of_outputs = self.variables["number_of_outputs"]
        maxValuesListi = self.variables["maxValuesListi"]
        maximumZ = self.variables["maximumZ"]
        list_of_bias_and_weight_values = self.variables["list_of_bias_and_weight_values"]

        def normsigmoid(x,b):
            maximumDXvalue = (2*(np.exp(2)))/b
            return 1/(1+np.exp(-x*maximumDXvalue))
        z = 0
        results = []
        for a in range(number_of_outputs):
            z = 0
            for i in range(number_of_inputs):
                z += (float(inputy[i])*float(list_of_bias_and_weight_values[a][i])) 
            z += float(list_of_bias_and_weight_values[a][-1])
            pred = maxValuesListi[a] * normsigmoid(z,maximumZ)
            results.append(pred)
        return results
    #Efflux: Uses a singular point in the same shape of the original output data used in the model to synthesize a possible input value set.
    def efflux(self, output):
        number_of_inputs = self.variables["number_of_inputs"]
        number_of_outputs = self.variables["number_of_outputs"]
        maxValuesListo = self.variables["maxValuesListo"]
        maximumZ_BD = self.variables["maximumZ_BD"]
        list_of_bias_and_weight_values_BD = self.variables["list_of_bias_and_weight_values_BD"]
        def normsigmoid(x,b):
            maximumDXvalue = (2*(np.exp(2)))/b
            return 1/(1+np.exp(-x*maximumDXvalue))
        z = 0
        results = []
        for a in range(number_of_inputs):
            z = 0
            for i in range(number_of_outputs):
                z += (float(output[i])*float(list_of_bias_and_weight_values_BD[a][i])) 
            z += float(list_of_bias_and_weight_values_BD[a][-1])
            pred = maxValuesListo[a] * normsigmoid(z,maximumZ_BD)
            results.append(pred)
        return results
    #dataClear: Clears all data pertaining to the model that has been trained using this node. New values can be created using new data.
    def dataClear():
        number_of_inputs = self.variables["number_of_inputs"]
        number_of_outputs = self.variables["number_of_inputs"]
        cost = self.variables["cost"]
        cost_BD = self.variables["cost_BD"]
        maxValuesListi = self.variables["maxValuesListi"]
        maxValuesListo = self.variables["maxValuesListo"]
        maximumZ = self.variables["maximumZ"]
        maximumZ_BD = self.variables["maximumZ_BD"]
        list_of_bias_and_weight_values = self.variables["list_of_bias_and_weight_values"]
        list_of_bias_and_weight_values_BD = self.variables["list_of_bias_and_weight_values_BD"]
        
        if(os.path.exists('Neuron-DataTESTINGNODE.txt')):
            os.remove('Neuron-DataTESTINGNODE.txt')
        if(os.path.exists('Neuron-DataTESTINGNODE_BD.txt')):
            os.remove('Neuron-DataTESTINGNODE_BD.txt')
        
        number_of_inputs, number_of_outputs, maximumZ, maximumZ_BD = 0
        cost, cost_BD, maxValuesListi, maxValuesListo. list_of_bias_and_weight_values, list_of_bias_and_weight_values_BD = []

        self.variables["number_of_inputs"] = number_of_inputs
        self.variables["number_of_inputs"] = number_of_outputs
        self.variables["cost"] = cost
        self.variables["cost_BD"] = cost_BD
        self.variables["maxValuesListi"] = maxValuesListi
        self.variables["maxValuesListo"] = maxValuesListo
        self.variables["maximumZ"] = maximumZ
        self.variables["maximumZ_BD"] = maximumZ_BD
        self.variables["list_of_bias_and_weight_values"] = list_of_bias_and_weight_values
        self.variables["list_of_bias_and_weight_values_BD"] = list_of_bias_and_weight_values_BD
        
        
