#Comparator VER 0.1.0 Testing Node: Weather Predictor
# Uses temperature, barometric, and time figures from the last 30 minutes and the current figures to predict the figures for the next 30 minutes.
# IS DESIGNED TO TEST EACH DIFFERENT FUNCTION. PLEASE TAKE NOTES OF ANY NECESSARY CHANGES THAT NEED TO BE MADE TO THE COMPARATOR PROGRAM AS NEEDED.
# COMMENTS FROM THE ORIGINAL CODE HAVE BEEN REMOVED FOR EASIER NOTE-TAKING

#***BEGINNING OF COMPARATOR 0.1.0 SECTION***

import os
import numpy as np
import random 
import ast

cost = []
cost_BD = []

list_of_bias_and_weight_values = []
list_of_bias_and_weight_values_BD = []

maxValuesListi = []
maxValuesListo = []

number_of_inputs = 0
#notes on conversion:
# number of inputs = number of input variables = number of weight values needed
# number of inputs + 1 = number of weight and bias values needed
number_of_outputs = 0

maximumZ = 0
maximumZ_BD = 0

alpha = []
alpha_BD = []

def normsigmoid(x,b):
    maximumDXvalue = (2*(np.exp(2)))/b
    return 1/(1+np.exp(-x*maximumDXvalue))

def normsigmoid_p(x,b):
    maximumDXvalue = (2*(np.exp(2)))/b
    return (maximumDXvalue)*(normsigmoid(x,b) * (1-normsigmoid(x,b)))


class Error(Exception):
    pass

class shapeException(Error):
    pass

def clear():
    os.system( 'cls' )


def fileLoad():
    global list_of_bias_and_weight_values
    global list_of_bias_and_weight_values_BD

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

def wbUpdate(mode):
    if(mode == 0):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        for a in range():
            datafile.write(str(list_of_bias_and_weight_values[b])+"\n")
    if(mode == 1):
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range():
            datafile.write(float(list_of_bias_and_weight_values_BD[a].strip()))
    if(mode == 2):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        flat_bwlist = [item for sublist in list_of_bias_and_weight_values for item in sublist]
        for a in range(len(list_of_bias_and_weight_values) * len(list_of_bias_and_weight_values[0])):
            datafile.write(str(float(flat_bwlist[a]))+"\n")
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        flat_bw_BDlist = [item for sublist in list_of_bias_and_weight_values_BD for item in sublist]
        for a in range(len(list_of_bias_and_weight_values_BD) * len(list_of_bias_and_weight_values_BD[0])):
            datafile.write(str(float(flat_bw_BDlist[a]))+"\n")

#NEW: Created TRAININGLOOP to minimize the code clutter
def TrainingLoopCORE(inputy,output,mode):
    global alpha
    global alpha_BD
    global cost
    global cost_BD
    global number_of_inputs
    global number_of_outputs
    global maxValuesListi
    global maxValuesListo
    global list_of_bias_and_weight_values
    global list_of_bias_and_weight_values_BD
    global maximumZ
    global maximumZ_BD

    

    if(mode==0):
        for a in range(number_of_outputs):

            z = 0
            for i in range(number_of_inputs):
                z += (float(inputy[i])*float(list_of_bias_and_weight_values[a][i])) 
            z += float(list_of_bias_and_weight_values[a][-1])
            #print("value of z: "+str(z))
            pred = maxValuesListi[a] * normsigmoid(z,maximumZ)
            print("pred: "+str(pred))
            target = output[a]
            print("target: "+str(target))
            cost[a][1] = (np.square((pred - target)/maxValuesListi[a]))

            dcost_dpred = 2 * ((pred - target)/maxValuesListi[a])

            dpred_dz = maxValuesListi[a]*normsigmoid_p(z,maximumZ)

            dcost_dz = dcost_dpred * dpred_dz
            
            for i in range(number_of_inputs):
                list_of_bias_and_weight_values[a][i] = float(list_of_bias_and_weight_values[a][i]) - alpha[a] * dcost_dz * (inputy[i]/maxValuesListi[a])
                #print("weight " + str(i+1) +" of set " + str(a) + " value: " + str(list_of_bias_and_weight_values[a][i]))

            list_of_bias_and_weight_values[a][-1] = float(list_of_bias_and_weight_values[a][-1]) - alpha[a] * dcost_dz

            cost[a][0] = cost[a][1]
            
    if(mode==1):
        for a in range(number_of_inputs):
            z = 0
            for i in range(number_of_outputs):
                z += (float(inputy[i])*float(list_of_bias_and_weight_values_BD[a][i])) 
            z += float(list_of_bias_and_weight_values_BD[a][-1])

            pred = maxValuesListo[a] * normsigmoid(z,maximumZ_BD)
            #print("pred: "+str(pred))
            target = output[a]
            #print("target: "+str(target))
            cost_BD[a][1] = (np.square((pred - target)/maxValuesListo[a]))

            dcost_dpred = 2 * ((pred - target)/maxValuesListo[a])

            dpred_dz = maxValuesListo[a]*normsigmoid_p(z,maximumZ_BD)


            dcost_dz = dcost_dpred * dpred_dz
            
            for i in range(number_of_outputs):

                list_of_bias_and_weight_values_BD[a][i] = float(list_of_bias_and_weight_values_BD[a][i]) - alpha_BD[a] * dcost_dz * (inputy[i]/maxValuesListo[a])

            list_of_bias_and_weight_values_BD[a][-1] = float(list_of_bias_and_weight_values_BD[a][-1]) - alpha_BD[a] * dcost_dz

            cost_BD[a][0] = cost_BD[a][1]


    


 
def Trainer(inputy, output, iterations):
    global number_of_inputs
    global number_of_outputs
    global cost
    global cost_BD
    global maxValuesListi
    global maxValuesListo
    global maximumZ
    global maximumZ_BD
    


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


    fileLoad()

    for j in range(number_of_inputs):
        costi = []
        costi.append(10)
        costi.append(10)
        cost.append(costi)
        alpha.append(0.2)
    for k in range(number_of_outputs):
        costo = []
        costo.append(10)
        costo.append(10)
        cost_BD.append(costo)
        alpha_BD.append(0.2)
    
    print((sum([sum(i) for i in zip(*cost)])))
    if (isinstance(inputy[0], list) and isinstance(output[0], list)): 
        if (iterations == 0):
            while(float(sum([sum(i) for i in zip(*cost)])) > 0.00001 and float(sum([sum(i) for i in zip(*cost_BD)])) > 0.00001):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy[ri],output[ri],0)
                TrainingLoopCORE(output[ri],inputy[ri],1)
        else:
            for a in range(iterations):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy[ri],output[ri],0)
                TrainingLoopCORE(output[ri],inputy[ri],1)
    elif (isinstance(inputy[0], list) and (isinstance(output[0], float) or isinstance(output[0], int))): 
        if (iterations == 0):
            while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy[ri],output,0)
                TrainingLoopCORE(output,inputy[ri],1)
        else:
            for a in range(iterations):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy[ri],output,0)
                TrainingLoopCORE(output,inputy[ri],1)
    elif (isinstance(output[0], list) and (isinstance(inputy[0], float) or isinstance(inputy[0], int))): 
        if (iterations == 0):
            while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and  float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy,output[ri],0)
                TrainingLoopCORE(output[ri],inputy,1)
        else:
            for a in range(iterations):
                ri = random.randint(0,len(inputy)-1)
                TrainingLoopCORE(inputy,output[ri],0)
                TrainingLoopCORE(output[ri],inputy,1)
    else:
        if (iterations == 0):
            while(float(sum([sum(i) for i in zip(*cost)])) > 0.01 and  float(sum([sum(i) for i in zip(*cost_BD)])) > 0.01):
                TrainingLoopCORE(inputy,output,0)
                TrainingLoopCORE(output,inputy,1)
        else:
            for a in range(iterations):
                TrainingLoopCORE(inputy,output,0)
                TrainingLoopCORE(output,inputy,1)
    wbUpdate(2)
    cost=[]
    cost_BD=[]
    
            
    
def influx(inputy):
    global number_of_inputs
    global number_of_outputs
    global list_of_bias_and_weight_values
    global list_of_bias_and_weight_values_BD
    global maxValuesListi
    global maxValuesListo
    global maximumZ
    global maximumZ_BD

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

def efflux(output):
    global number_of_inputs
    global number_of_outputs
    global list_of_bias_and_weight_values
    global list_of_bias_and_weight_values_BD
    global maxValuesListi
    global maxValuesListo

    z = 0
    results = []
    for a in range(number_of_inputs):
            z = 0
            for i in range(number_of_outputs):
                z += (float(output[i])*float(list_of_bias_and_weight_values_BD[a][i])) 
            z += float(list_of_bias_and_weight_values_BD[a][-1])
            pred = maxValuesListo[a] * normsigmoid(z)
            results.append(pred)
    return results

#NEW; PUT THIS IN Basic Toolbox PROGRAM
#ADDED NEW FUNCTION: fauxDataGen: creates faux data for both input and output from provided figures and ranges.
#WARNING: SHOULD ONLY BE USED FOR PROOF OF CONCEPT OR GETTING THE MODEL UP TO SPEED USING EDUCATED GUESSES. DO NOT USE IF DATA IS UNKNOWN BEFORE USE.
#IT IS PREFERRABLE IF THE BOUNDS HAVE MINIMUM DIFFERENCE.
#Both inputs should only take in data formed as such: [[input type 1 lower bounds, it1 upper bounds], [input type 2 lower bounds, it2 upper bounds]...]
def fauxDataGen(iData, oData, numPoints):
    fauxData = []
    fauxDatai = []
    fauxDatai2 = []
    fauxDatao = []
    fauxDatao2 = []

    for x in range(numPoints):
        fauxDatai2 = []
        fauxDatao2 = []
        for a in range(len(iData)):            
            fauxDatai2.append(random.randint(iData[a][0], iData[a][1]))
        fauxDatai.append(fauxDatai2)
        for b in range(len(oData)):
            fauxDatao2.append(random.randint(oData[b][0], oData[b][1]))
        fauxDatao.append(fauxDatao2)

    fauxData.append(fauxDatai)
    fauxData.append(fauxDatao)
    return fauxData
            

#***END OF COMPARATOR 0.1.0 SECTION***

#***BEGINNING OF ORIGINAL SECTION***

import requests
import time
from urllib.request import urlopen
from datetime import datetime

def internet_on():
    try:
        urllib.request.urlopen('http://google.com', timeout=1)
        return "true"
    except: 
        return "false"


while True:
    temp = []
    timex = []
    date = []
    print("Warming up the model...")
    weatherModel =[[[49,1700,1229],[49,1800,1229],[50,1900,1229],[51,2000,1229],[51,2100,1229],[52,2200,1229],[52,2300,1229]],[[49,1800,1229],[50,1900,1229],[51,2000,1229],[51,2100,1229],[51,2200,1229],[52,2300,1229],[52,0,1230]]] 
    Trainer(weatherModel[0],weatherModel[1],0)
    print("Training complete!\n")
    print("Commencing weather prediction model...")
    while(internet_on()):
        now = datetime.now()
        #Problem: Link likes to time out for some reason. Need a more reliable connection. Free, if possible.
        url = 'http://api.openweathermap.org/data/2.5/weather?q=Isla%20Vista,us&mode=json&appid=bca9a5620f4d14d334e732bd8719286b&units=imperial'
        res = requests.get(url)
        data = res.json()
        currentTemperature = data['main']['temp']
        currentTime = int(now.strftime('%H%M'))
        currentDate = int(now.strftime("%m%d"))
        print("Current Temp, Date, and Time:" + str(currentTemperature) + ", " + str(currentTime) + ", " + str(currentDate))

        
        
        

        if(len(temp) == 2):
            temp[1] = currentTemperature
            timex[1] = currentTime
            date[1] = currentDate
            inputa = [temp[0],timex[0],date[0]]
            output = [temp[1],timex[1],date[1]]

            #Line below adjusts weights and bias live
            Trainer([temp[0],timex[0],date[0]], [currentTemperature,currentTime,currentDate] , 0)

            #Line below outputs from current data for future data
            #confusing yes, but it should work if everything else works
            currentResult = influx(inputa)

            
            print("Temp, Date and Time Guess based on current conditions:" + str(currentResult[0]) + ", " + str(currentResult[1])+", " + str(currentResult[2]))
            temp[0] = currentTemperature
            timex[0] = currentTime
            date[0] = currentDate
        else:
            print("Not enough data! Please wait for next data collection cycle.")
            temp.append(currentTemperature)
            timex.append(currentTime)
            date.append(currentDate)

            
        time.sleep(10)
