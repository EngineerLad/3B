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
    return 1/(1+np.exp(-x*(1/(b/10))))

def normsigmoid_p(x,b):
    return normsigmoid(x,b) * (1-normsigmoid(x,b))


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
        for a in range(len(list_of_bias_and_weight_values) * len(list_of_bias_and_weight_values[0])):
            datafile.write(float(list_of_bias_and_weight_values[a].strip()))
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range(len(list_of_bias_and_weight_values_BD) * len(list_of_bias_and_weight_values_BD[0])):
            datafile.write(float(list_of_bias_and_weight_values_BD[a].strip()))

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
            print("value of z" + str(z))
            print("max value; " + str(maxValuesListi[a]))
            print("norm sigmoid value: " + str(float(normsigmoid(float(z),maximumZ))))
            print("derivative of norm sigmoid value: " + str(normsigmoid_p(z,maximumZ)))
            pred = maxValuesListi[a] * normsigmoid(z,maximumZ)
            print("pred value; " + str(pred))
            target = output[a]
            print("target value; " + str(target))
            cost[a][1] = (np.square(pred - target))

            dcost_dpred = 2 * (pred - target)
            print("dcost_dpred: " + str(dcost_dpred))
            dpred_dz = maxValuesListi[a] * normsigmoid_p(z,maximumZ)

            print("dpred_dz: " + str(dpred_dz))
            dcost_dz = dcost_dpred * dpred_dz
            
            for i in range(number_of_inputs):
                print("alpha: " + str(alpha[a]))
                print("dcostdz: " + str(dcost_dz))
                print("input point: " + str(inputy[i]))
                print("weight delta :" + str(alpha[a] * dcost_dz * inputy[i]))
                list_of_bias_and_weight_values[a][i] = float(list_of_bias_and_weight_values[a][i]) - alpha[a] * dcost_dz * inputy[i]
                print("weight "+ str(i) + " value: " + str(list_of_bias_and_weight_values[a][i]))
            list_of_bias_and_weight_values[a][number_of_inputs] = float(list_of_bias_and_weight_values[a][number_of_inputs]) - alpha[a] * dcost_dz
            print("bias value: " + str(list_of_bias_and_weight_values[a][number_of_inputs]))
            print("previous cost:" + str(cost[a][0]))
            print("new cost:" + str(cost[a][1]))
            cost[a][0] = cost[a][1]
            
    if(mode==1):
        for a in range(number_of_inputs):

            z = 0
            for i in range(number_of_outputs):
                z += (float(inputy[i])*float(list_of_bias_and_weight_values_BD[a][i])) 
            z += float(list_of_bias_and_weight_values_BD[a][-1])

            pred = maxValuesListo[a] * normsigmoid(z,maximumZ_BD)
            target = output[a]

            cost_BD[a].append(np.square(pred - target))

            dcost_dpred = 2 * (pred - target)
            dpred_dz = maxValuesListo[a] * normsigmoid_p(z,maximumZ_BD)
            dcost_dz = dcost_dpred * dpred_dz
            
            for i in range(number_of_outputs):
                list_of_bias_and_weight_values_BD[a][i] = float(list_of_bias_and_weight_values_BD[a][i]) - alpha_BD[a] * float(list_of_bias_and_weight_values_BD[a][i]) * dcost_dz
            
            if(cost_BD[a][1]>cost_BD[a][0]):
                alpha_BD[a]+=0.01
            if((cost_BD[a][1]-cost_BD[a][0])<-0.5):
                alpha_BD[a]-=0.01
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
    

    maxValuesListi = [max(elem) for elem in zip(*output)]
    maxValuesListo = [max(elem) for elem in zip(*inputy)]


    if isinstance(inputy[0], list):
        number_of_inputs = len(inputy[0])
    else:
        number_of_inputs = len(inputy)

    if isinstance(output[0], list):
        number_of_outputs = len(output[0])    
    else:
        number_of_outputs = len(output)

    print("Number of inputs: " + str(number_of_inputs))
    print("Number of outputs: " + str(number_of_outputs))
    maximumZ = sum([max(elem) for elem in zip(*output)])
    maximumZ_BD = sum([max(elem) for elem in zip(*inputy)])

    fileLoad()

    for x in range(number_of_outputs):
        costi = []
        costi.append(10)
        costi.append(10)
        cost.append(costi)
        alpha.append(0.2)
    for y in range(number_of_outputs):
        costo = []
        costo.append(10)
        costo.append(10)
        cost_BD.append(costo)
        alpha_BD.append(0.2)

    if (iterations == 0):
        while(float(sum([sum(i) for i in zip(*cost)])) > 0.004 and  float(sum([sum(i) for i in zip(*cost_BD)])) > 0.004):
            ri = random.randint(0,len(inputy)-1)
            TrainingLoopCORE(inputy[ri],output[ri],0)
            TrainingLoopCORE(output[ri],inputy[ri],1)
    else:
        for a in range(iterations):
            TrainingLoopCORE(inputy[ri],output[ri],0)
            TrainingLoopCORE(output[ri],inputy[ri],1)
    wbUpdate(2)
    
            
    
def influx(inputy):
    global number_of_inputs
    global number_of_outputs
    global list_of_bias_and_weight_values
    global list_of_bias_and_weight_values_BD
    global maxValuesListi
    global maxValuesListo

    z = 0
    results = []
    for a in range(number_of_outputs):
            z = 0
            for i in range(number_of_inputs):
                z += (float(inputy[i])*float(list_of_bias_and_weight_values[a][i])) 
            z += float(list_of_bias_and_weight_values[a][-1])
            pred = maxValuesListi[a] * normsigmoid(z)
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
            fauxDatai2.append(random.uniform(iData[a][0], iData[a][1]))
        fauxDatai.append(fauxDatai2)
        for b in range(len(oData)):
            fauxDatao2.append(random.uniform(oData[b][0], oData[b][1]))
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
    pres = []
    timex = []
    date = []
    print("Warming up the model...")
    fauxDataWeather = fauxDataGen([[41,71],[1010,1040],[0,2359],[1201,1231]],[[41,71],[1010,1040],[0,2359],[1201,1231]],1000)
    Trainer(fauxDataWeather[0], fauxDataWeather[1] , 0)
    print("Training complete!\n")
    print("Commencing weather prediction model...")
    while(internet_on()):
        print("aw shit time to work again\n")
        now = datetime.now()
        #Problem: Link likes to time out for some reason. Need a more reliable connection. Free, if possible.
        url = 'http://api.openweathermap.org/data/2.5/weather?q=Isla%20Vista,us&mode=json&appid=bca9a5620f4d14d334e732bd8719286b&units=imperial'
        res = requests.get(url)
        data = res.json()
        currentTemperature = data['main']['temp']
        currentPressure = data['main']['pressure']
        currentTime = int(now.strftime('%H%M%S'))
        currentDate = int(now.strftime("%d%m"))
        print("Current Temp, Pres, Date, and Time:" + currentTemperature + ", " + currentPressure + ", " + currentDate + ", " + currentTime)

        
        
        

        if(len(temp) == 2):
            temp[1] = currentTemperature
            pres[1] = currentPressure
            timex[1] = currentTime
            date[1] = currentDate
            inputa = [temp[0],pres[0],timex[0],date[0]]
            output = [temp[1],pres[1],timex[1],date[1]]

            #Line below adjusts weights and bias live
            Trainer(inputa, output , 0)

            #Line below outputs from current data for future data
            #confusing yes, but it should work if everything else works
            currentResult = influx(output)

            
            print("Future Temp, Pres, Date, and Time:" + currentResult[0] + ", " + currentResult[1] + ", " + currentResult[3] + ", " + currentResult[2])
            temp[0] = currentTemperature
            pres[0] = currentPressure
            timex[0] = currentTime
            date[0] = currentDate
        else:
            print("Not enough data! Please wait for next data collection cycle.")
            temp.append(currentTemperature)
            pres.append(currentPressure)
            timex.append(currentTime)
            date.append(currentDate)

            
        time.sleep(3600)
