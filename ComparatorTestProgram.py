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

number_of_inputs = 0
number_of_outputs = 0

alpha = 0.2 
z = 0

def normsigmoid(x):
    return 1/(1+np.exp(-x))

def normsigmoid_p(x):
    return normsigmoid(x) * (1-normsigmoid(x))


class Error(Exception):
    pass

class shapeException(Error):
    pass




def fileLoad(aa,bb):

        if(os.path.exists('Neuron-DataTESTINGNODE.txt')):
            datafile = open("Neuron-DataTESTINGNODE.txt","r")
            lineByLine = datafile.readlines()
            counter = 0
            for data in lineByLine:
                print("getting stuck here 1")
                print(str(counter))
                print(str(data))
                list_of_bias_and_weight_values.append(data)
                counter += 1
            
        else:
            datafile = open("Neuron-DataTESTINGNODE.txt","w+")
            for a in range(bb):
                for b in range(aa+1):
                    list_of_bias_and_weight_values.append(np.random.randn())
                    datafile.write(str(list_of_bias_and_weight_values[b])+"\n")

        if(os.path.exists('Neuron-Data_BDTESTINGNODE.txt')):
            datafile_BD = open("Neuron-Data_BDTESTINGNODE.txt","r")
            lineByLine2 = datafile_BD.readlines()
            counter = 0
            for data in lineByLine2:
                print("getting stuck here 2")
                print(str(counter))
                print(str(data))
                list_of_bias_and_weight_values_BD.append(data)
                counter += 1
            
        else:
            datafile = open("Neuron-Data_BDTESTINGNODE.txt","w+")
            for a in range(aa):
                for b in range(bb+1):
                    list_of_bias_and_weight_values_BD.append(np.random.randn())
                    datafile.write(str(list_of_bias_and_weight_values_BD[b])+"\n")




def wbUpdate(mode):
    if(mode == 0):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values[a])
    if(mode == 1):
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values_BD[a])
    if(mode == 2):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values[a])
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values_BD[a])

#NEW: Created TRAININGLOOP to minimize the code clutter
def TrainingLoopCORE(inputy,output,maximum_value,mode):
    if(mode==0):
        z = 0
        for i in range(number_of_outputs):
            z += (inputy[i]*list_of_bias_and_weight_values[i]) 
        z += float(list_of_bias_and_weight_values[-1].strip())

        pred = maximum_value * normsigmoid(z)
        target = output

        cost.append(np.square(pred - target))

        dcost_dpred = 2 * (pred - target)
        dpred_dz = maximum_value * normsigmoid_p(z)
        dcost_dz = dcost_dpred * dpred_dz
        
        for i in range(len(inputy)):
            list_of_bias_and_weight_values[i] = list_of_bias_and_weight_values[i] - alpha * list_of_bias_and_weight_values[i] * dcost_dz

        if(cost[1]>cost[0]):
            alpha+=0.01
        if((cost[1]-cost[0])<-0.5):
            alpha-=0.01
        cost[0] = cost[1]
        print(str(cost[0]))
    if(mode==1):
        for i in range(number_of_inputs):
            z += (inputy[i]*list_of_bias_and_weight_values_BD[i]) 
        z += list_of_bias_and_weight_values_BD[-1]

        pred = maximum_value * normsigmoid(z)
        target = output

        cost[1] = np.square(pred - target)

        dcost_dpred = 2 * (pred - target)
        dpred_dz = maximum_value * normsigmoid_p(z)
        dcost_dz = dcost_dpred * dpred_dz
        
        for i in range(len(inputy)):
            list_of_bias_and_weight_values_BD[i] = list_of_bias_and_weight_values_BD[i] - alpha * list_of_bias_and_weight_values_BD[i] * dcost_dz

        if(cost[1]>cost[0]):
            alpha+=0.01
        if((cost[1]-cost[0])<-0.5):
            alpha-=0.01
        cost[0] = cost[1]
        print(str(cost[0]))
    

def TrainingLoopL1(inputy,output,maximum_value):
    z = 0
    TrainingLoopCORE(inputy,output,maximum_value,0)
    print("Forward Cost: " + cost[0])
    z = 0
    TrainingLoopCORE(inputy,output,maximum_value,1)
    print("Backward Cost: " + cost_BD[0])


 
def Trainer(inputy, output, iterations):
    number_of_input = 0
    number_of_output = 0

    if isinstance(inputy[0], list):
        number_of_input = len(inputy[0])
    else:
        number_of_input = len(inputy)

    if isinstance(output[0], list):
        number_of_output = len(output[0])    
    else:
        number_of_output = len(output)

    print("Number of inputs: " + str(number_of_input))
    print("Number of outputs: " + str(number_of_output))

    fileLoad(number_of_input , number_of_output)
    print("Finished Loading")

    number_of_weight_bias_value_sets = number_of_input + 1
    number_of_weight_bias_value_sets_BD = number_of_output + 1

    print("Assigned here")
    cost.append(10)
    cost_BD.append(10)
    print("Assigned here2")
    if (iterations == 0):
        print("entered loop")
        while(cost[0]>0.001):
            print("confirmation 1")
            if isinstance(output[0], list):
                print("confirmation 2")
                maximum_value = max([sublist[-1] for sublist in output])
                print("confirmation 3")
                ri = random.randrange(len(output))
                print("confirmation 4")
                for b in range(number_of_output):
                    
                    print("running...")
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)
                print("confirmation 5")
            else:
                maximum_value = max(output)
                for b in range(len(output)):
                    print("running...2")
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)

            if isinstance(inputy[0], list):
                maximum_value = max([sublist[-1] for sublist in inputy])
                ri = random.randrange(len(inputy))
                for b in range(number_of_input):
                    print("running...3")
                    TrainingLoopL1(output,inputy[ri][b],maximum_value)
            else:
                maximum_value = max(inputy)
                for b in range(len(inputy)):
                    print("running...4")
                    TrainingLoopL1(output,inputy[ri][b],maximum_value)
    else:
        for a in range(iterations):
            if isinstance(output[0], list):
                ri = random.randrangerandrange(len(output))
                for b in range(number_of_output):
                    print("running...5")
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)
            else:
                for b in range(len(output)):
                    print("running...6")
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)

            if isinstance(inputy[0], list):
                ri = random.randrangerandrange(len(inputy))
                for b in range(number_of_input):
                    print("running...7")
                    TrainingLoopL1(output,inputy[ri][b],maximum_value)
            else:
                for b in range(len(inputy)):
                    print("running...8")
                    TrainingLoopL1(output,inputy[ri][b],maximum_value)
    wbUpdate(2)
    
            
    
def influx(inputy):
    z = 0
    results = []
    for x in (number_of_weight_values):
        for i in (number_of_weight_bias_value_sets):
                z += (inputy[i]*list_of_bias_and_weight_values[i]) 
        z += list_of_bias_and_weight_values[number_of_weight_values]
        results[x] = maximum_value_of_set*normsigmoid(z)
    return results

def efflux(output):
    z = 0
    results = []
    for x in (number_of_weight_values_BD):
        for i in (number_of_weight_bias_value_sets_BD):
                z += (output[i]*list_of_bias_and_weight_values_BD[i]) 
        z += list_of_bias_and_weight_values_BD[number_of_weight_values_BD]
        results[x] = maximum_value_of_set_BD*normsigmoid(z)
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

            print("Current Temp, Pres, Date, and Time:" + currentTemperature + ", " + currentPressure + ", " + currentDate + ", " + currentTime)
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

            
        time.sleep(1)
