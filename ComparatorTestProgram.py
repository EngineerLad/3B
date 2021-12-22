#Comparator VER 0.1.0 Testing Node: Weather Predictor
# Uses temperature, barometric, and time figures from the last 30 minutes and the current figures to predict the figures for the next 30 minutes.
# IS DESIGNED TO TEST EACH DIFFERENT FUNCTION. PLEASE TAKE NOTES OF ANY NECESSARY CHANGES THAT NEED TO BE MADE TO THE COMPARATOR PROGRAM AS NEEDED.
# COMMENTS FROM THE ORIGINAL CODE HAVE BEEN REMOVED FOR EASIER NOTE-TAKING

#***BEGINNING OF COMPARATOR 0.1.0 SECTION***

import os
import numpy as np
import random 

def normsigmoid(x):
    return 1/(1+np.exp(-x))

def normsigmoid_p(x):
    return normsigmoid(x,1) * (1-normsigmoid(x,1))


class Error(Exception):
    pass

class shapeException(Error):
    pass


cost = []
cost_BD = []
maximum_value_of_set = 0
maximum_value_of_set_BD = 0
list_of_bias_and_weight_values = []
list_of_bias_and_weight_values_BD = []

number_of_inputs = 0
number_of_outputs = 0

alpha = 0.2 
z = 0

def fileLoad():
        if(os.path.exists('Neuron-DataTESTINGNODE.txt')):
            datafile = open("Neuron-DataTESTINGNODE.txt","r")
            lineByLine = datafile.readlines()
            counter = 0
            for a in range(number_of_outputs):
                for b in range(number_of_inputs):
                    for data in lineByLine:
                        list_of_bias_and_weight_values[counter] = data
                        counter += 1
            
        else:
            datafile = open("Neuron-DataTESTINGNODE.txt","w+")
            for a in range(number_of_outputs):
                for b in range(number_of_inputs):
                    list_of_bias_and_weight_values[b] = np.random.randn()
                    datafile.write(list_of_bias_and_weight_values[b]+'\n')

        if(os.path.exists('Neuron-Data_BDTESTINGNODE.txt')):
            datafile_BD = open("Neuron-Data_BDTESTINGNODE.txt","r")
            lineByLine = datafile.readlines()
            counter = 0
            for a in range(number_of_inputs):
                for b in range(number_of_outputs):
                    for data in lineByLine:
                        list_of_bias_and_weight_values_BD[counter] = data
                        counter += 1
            
        else:
            datafile = open("Neuron-Data_BDTESTINGNODE.txt","w+")
            for a in range(number_of_inputs):
                for b in range(number_of_outputs):
                    list_of_bias_and_weight_values_BD[b] = np.random.randn()
                    datafile.write(list_of_bias_and_weight_values_BD[b]+'\n')




def wbUpdate(mode):
    if(mode == 0):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values[a]+'\n')
    if(mode == 1):
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values_BD[a]+'\n')
    if(mode == 2):
        datafile = open("Neuron-DataTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values[a]+'\n')
        datafile = open("Neuron-Data_BDTESTINGNODE.txt","w")
        for a in range(len(inputy)):
            datafile.write(list_of_bias_and_weight_values_BD[a]+'\n')

#NEW: Created TRAININGLOOP to minimize the code clutter
def TrainingLoopCORE(inputy,output,maximum_value,mode):
    if(mode==0):
        for i in (number_of_weight_bias_value_sets):
            z += (inputy[i]*list_of_bias_and_weight_values[i]) 
        z += list_of_bias_and_weight_values[-1]

        pred = maximum_value * normsigmoid(z)
        target = output

        cost[1] = np.square(pred - target)

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
    if(mode==1):
        for i in (number_of_weight_bias_value_sets_BD):
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
    

def TrainingLoopL1(inputy,output,maximum_value):
    z = 0
    TrainingLoopCORE(inputy,output,maximum_value,0)
    print("Forward Cost: " + cost[0])
    z = 0
    TrainingLoopCORE(inputy,output,maximum_value,1)
    print("Backward Cost: " + cost_BD[0])


 
def Trainer(inputy, output, iterations):

    if(type(inputy[0]) == list):
        number_of_inputs = len(inputy[0])
    else:
        number_of_inputs = len(inputy)
    if(type(output[0]) == list):
        number_of_outputs = len(output[0])    
    else:
        number_of_outputs = len(output)
    
    for a in range(iterations):
        if(type(output[0]) == list):
            ri = randrange(len(output))
            ri2 = randrange(len(inputy))
            for b in range(number_of_outputs):
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)
        else:
            for b in range(len(output)):
                    TrainingLoopL1(inputy,output[ri][b],maximum_value)

        if(type(inputy[0]) == list):
            ri = randrange(len(inputy))
            for b in range(number_of_inputs):
                    TrainingLoopL1(output,inputy[ri][b],maximum_value)
        else:
            for b in range(len(inputy)):
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
    fauxData =[]
    fauxDatai = []
    fauxDatao = []
    x=0
    while (x!=(numPoints-1)):
        a=0
        b=0
        while(a!=len(iData)-1):
            fauxDatai[x][a] = random.uniform(iData[a][0], iData[a][1])
            a+=1
        while(b!=len(oData)-1):
            fauxDatao[x][b] = random.uniform(oData[b][0], oData[b][1])
            b+=1
        x+=1
    fauxData[0] = fauxDatai
    fauxData[1] = fauxDatao
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
    #fauxDataWeather = fauxDataGen([[41,71],[1010,1040],[0,2359],[1201,1231]],[[41,71],[1010,1040],[0,2359],[1201,1231]],1000)
    #Trainer(fauxDataWeather[0], fauxDataWeather[1] , 0)
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
            output = [temp[1],pres[1],timex[1],date[0]]

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
