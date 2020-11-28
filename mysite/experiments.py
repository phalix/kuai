from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404
from datetime import datetime
from django import forms
from mysite.models import Project,Experiment,Result,Sample,Metrics,Loss,LayerWeights
import mysite.neuralnetwork
import mysite.dataoperation
import asyncio
import threading
import tensorflow as tf

def createnewexperimentinproject(project):
    exp = Experiment()
    exp.project = project
    exp.status = 0
    exp.noofepochs = 100 
    exp.batchsize = 50
    exp.save()
    return exp

def getlatestexperiment(project_id):
    latestexp = 0
    experiments = Experiment.objects.filter(project=project_id).all()
    for e in experiments:
        latestexp = max(latestexp,e.id)
    
    if latestexp == 0:
        project = get_object_or_404(Project, pk=project_id)
        exp = createnewexperimentinproject(project)
        latestexp = exp.id
    return latestexp

def experimentsetuplastexperiment(request,project_id):
    latestexp = getlatestexperiment(project_id)
    return experimentsetup(request,project_id,latestexp)


def experimentsetup(request,project_id,experiment_id):
    #define earlystopping, after how many epochs
    #define in general how many epochs should run
    #define save val_loss, loss etc.

    #get callbacks etc.!

    template = loader.get_template('experiments/experiment.html')
    project = get_object_or_404(Project, pk=project_id)
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    
    
    metrics = Metrics.objects.filter(experiment=experiment_id).all()
    

    context = {
        "project" : project,
        "experiment" : experiment,
        "project_id" : project_id,
        "menuactive":5,
        "metrics":mysite.neuralnetwork.getkeraslayeroptions('keras.metrics'),
        "loss":mysite.neuralnetwork.getkeraslayeroptions('keras.losses'),
        "selloss":experiment.loss,
        "selmetrics":metrics,
        "callbacks":['earlystopping','...']

    }
    
    return HttpResponse(template.render(context, request))


def uploadexpsetup(request,project_id,experiment_id):
    project = get_object_or_404(Project, pk=project_id)
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    
    Metrics.objects.filter(experiment=experiment_id).all().delete()

    request.POST["loss"]
    loss = Loss(name=request.POST["loss"])
    loss.save()
    experiment.loss = loss
    
    experiment.batchsize = request.POST["batchsize"]
    experiment.noofepochs = request.POST["noofepochs"]
    
    experiment.save()
    if "metrics[]" in request.POST:
        metrics = request.POST.getlist("metrics[]")
        for m in metrics:
            metric = Metrics(name=m,experiment=experiment)
            metric.save()
            #experiment.metrics.add(metric)



    for parameter in request.POST:
        print(parameter)
        print(request.POST[parameter])
    return HttpResponseRedirect('/runexperiment/'+str(project_id))

def retrieveExperimentStats(project_id):
    
    expDict = {}
    currentEpoch = 0
    experiments = Experiment.objects.filter(project=project_id).all()
    for exp in experiments:
        exp.parameters
        results = Result.objects.filter(experiment=exp.id).all()
        resultsDict = {}
        for res in results:
            #res.configuration
            #res.epochno
            #res.id
            samp = Sample.objects.filter(result=res.id).all()
            samples = {}
            cur = {}
            for sa in samp:
                if res.id in samples.keys():
                    cur = samples[res.id]
                else:
                    cur = {}
                #provide max and min of source
                #MIN
                #identifier = sa.name+"_"+str(sa.source)+"_min"
                #if identifier in cur:
                #    if cur[identifier] > sa.number:
                #        cur[identifier] = sa.number
                #else:
                #    cur[identifier] = sa.number
                #MAX
                identifier = sa.name+"_"+str(sa.source)+"_max"
                if identifier in cur:
                    if cur[identifier] < sa.number:
                        cur[identifier] = sa.number
                else:
                    cur[identifier] = sa.number
                

                samples[res.id] = cur
            resDict = {}
            resDict["samples"] = cur
            resDict["epoch"] = res.epochno
            currentEpoch = max(currentEpoch,res.epochno)
            resultsDict[res.epochno] = resDict
        
        expDict[exp.id] = resultsDict
        expDict["currentEpoch"] = currentEpoch
    #print(expDict)
    return expDict


def prepareModel(project,experiment):
    cur_metrics = Metrics.objects.filter(experiment=experiment.id).all()
    metrics = []
    for m in cur_metrics:
        metrics.append(m.name)
    lossfunction = experiment.loss.name
    model = mysite.neuralnetwork.getcurrentmodel(project.id,lossfunction,metrics)
    #restore weights
    loadmodelweights(experiment.id,model)

    return model



def runexperiment(project_id,experiment_id):
    
    import sys, traceback
    project = get_object_or_404(Project, pk=project_id)
    exp = get_object_or_404(Experiment, pk=experiment_id)
    
    createProjectDessa(project.id)    

    cur_metrics = Metrics.objects.filter(experiment=experiment_id).all()
    metrics = []
    for m in cur_metrics:
        metrics.append(m.name)
    lossfunction = exp.loss.name

    from django.db.models import Max
    results = Result.objects.filter(experiment=experiment_id).all()
    maxepochno = results.aggregate(Max('epochno'))
    currentepoch = maxepochno['epochno__max']
    if not currentepoch:
        currentepoch = 0
    
    
    model = prepareModel(project,exp)
    saveModelInProjectFolder(model,project_id)
    
    
    train_df3 = mysite.dataoperation.getTransformedData(project_id,mysite.dataoperation.TRAIN_DATA_QUALIFIER)
    test_df3 = mysite.dataoperation.getTransformedData(project_id,mysite.dataoperation.TEST_DATA_QUALIFIER)
    saveParquetDataInProjectFolder(train_df3,project_id,qualifier=1)
    saveParquetDataInProjectFolder(test_df3,project_id,qualifier=2)


    submitDessaJob(project_id)

    #random split to required batch size
    # 1. get count of train_df3
    # 2. 1/train_df3/batch_size = share
    # 3. randomsplit with array of share and count 1/share
    
    def getArrayToSplitDFToBatch(count,batchsize):
        i = batchsize/count
        import math
        import numpy as np
        i_floor = math.floor(i)
        i_ceil = int(math.ceil(1/i))
        arrayforsplitting = np.zeros(i_ceil)
        arrayforsplitting[:] = i
        return arrayforsplitting
    
    def getMultipleDf(df,batchsize):
        counted = df.count()
        arraytosplit = getArrayToSplitDFToBatch(counted,batchsize)
        multiple_df = train_df3.randomSplit(arraytosplit)
        return multiple_df

    multiple_df = getMultipleDf(train_df3,exp.batchsize)
    multiple_df_test = getMultipleDf(test_df3,exp.batchsize)

    

    from collections.abc import Iterable

    
    NoOfEpochs = exp.noofepochs
    j = currentepoch
    while j <= NoOfEpochs+currentepoch and exp.status == 1:
    #for j in range(1,NoOfEpochs+1):
        samples = []
        

        currentEpoch = j
        batchcounter = 0
        for i in multiple_df:
            batchcounter = batchcounter+1
            try:
                currentdf = mysite.dataoperation.getXandYFromDataframe(i,project)
                if len(currentdf['y'])>0:
                    datafromtraining = model.train_on_batch(currentdf["x"],currentdf["y"])
                    if not isinstance(datafromtraining, Iterable):
                        datafromtraining = [datafromtraining]
                    
                    samp = Sample(name=lossfunction,number=datafromtraining[0],batch=batchcounter,source=mysite.dataoperation.TRAIN_DATA_QUALIFIER)
                    samples.append(samp)
                    for k in range(0,len(metrics)):
                        samp = Sample(name=metrics[k],number=datafromtraining[k+1],batch=batchcounter,source=mysite.dataoperation.TRAIN_DATA_QUALIFIER)
                        samples.append(samp)   
                else:
                    print("Batch with no Elements in Training")
            except Exception as e:
                print(e)
                traceback.print_exc(file=sys.stdout)
                print("Training not successfull")
            else:
                print("Training successfull")
            
                
        for i in multiple_df_test:
            try:
                currentdf = mysite.dataoperation.getXandYFromDataframe(i,project)
                if len(currentdf['y'])>0:
                    datafromtesting = model.test_on_batch(currentdf["x"],currentdf["y"])  
                    
                    if not isinstance(datafromtesting, Iterable):
                        datafromtesting = [datafromtesting]
                    
                    samp = Sample(name=lossfunction,number=datafromtesting[0],batch=batchcounter,source=mysite.dataoperation.TEST_DATA_QUALIFIER)
                    samples.append(samp)
                    for k in range(0,len(metrics)):
                        samp = Sample(name=metrics[k],number=datafromtesting[k+1],batch=batchcounter,source=mysite.dataoperation.TEST_DATA_QUALIFIER)
                        samples.append(samp)
                else:
                    print("Batch with no Elements in Testing")
            except:
                print("Testing not successfull") 
            else:
                print("Testing successfull")

        res = Result(epochno=j)
        res.experiment = exp
        res.save()
        for s in samples:
            s.result = res
            s.save()
        

        exp = get_object_or_404(Experiment, pk=experiment_id)
        j=j+1
    exp = get_object_or_404(Experiment, pk=experiment_id)
    exp.status = 3
    exp.save()
    savemodelweights(experiment_id,model)


def runlatestexperiment(request,project_id):
    experiment_id = getlatestexperiment(project_id)
    return run(request,project_id,experiment_id)


import threading
def run(request,project_id,experiment_id):

    #TODO: just add a button to run the experiments
    #TODO: make epochs configurable
    #TODO: make it possible to cancel experiment
    template = loader.get_template('experiments/runexperiment.html')
    project = get_object_or_404(Project, pk=project_id)
    experiment_id = getlatestexperiment(project_id)
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    expDict = retrieveExperimentStats(project_id)
    
    if not (experiment.noofepochs > 0 and experiment.batchsize > 0 and experiment.loss):
        return HttpResponseRedirect("/experimentsetup/"+str(project_id)+"/"+str(experiment_id))

    #startexperiment(project_id,experiment_id)
    metrics = Metrics.objects.filter(experiment=experiment_id).all()
    
    context = {
        "project" : project,
        "project_id" : project_id,
        "experiment" : experiment,
        "metrics": metrics,
        "menuactive":5,
        "experiments": expDict,

    }
    
    return HttpResponse(template.render(context, request))   

def startexperiment(project_id,experiment_id):
    thread = threading.Thread(target=runexperiment, args=(project_id,experiment_id))
    thread.daemon = True 
    thread.start()


def getexperimentsstasperproject(request,project_id,experiment_id):
    import json
    
    expDict = retrieveExperimentStats(project_id)
    if "currentEpoch" in expDict:
        currentEpoch = expDict["currentEpoch"]
    else:
        currentEpoch = 0
    
    expDict.pop("currentEpoch",None)
    
    response_data = {
        'state': expDict,
        'currentEpoch': currentEpoch,
        'active': active
    }
    return HttpResponse(json.dumps(response_data), content_type='application/json')

currentEpoch = 0
active = False

def startExperimentsPerProject(request,project_id,experiment_id):
    import json
    #active = True
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    if True or experiment.status in (0,3):
        startexperiment(project_id,experiment.id)
    experiment.status = 1
    experiment.save()
    
    return HttpResponse(json.dumps({}), content_type='application/json')

def stopExperimentsPerProject(request,project_id,experiment_id):
    import json
    #active = False
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    experiment.status = 2
    experiment.save()
    return HttpResponse(json.dumps({}), content_type='application/json')

def deleteExperimentData(request,project_id,experiment_id):
    import json
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    #experiments = Experiment.objects.filter(project=project_id).all()
    experiment.delete()
    return HttpResponse(json.dumps({}), content_type='application/json')

def parameter(request,project_id):
    template = loader.get_template('experiments/parameterfixing.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        "menuactive":5,
        "parameters":[(3,"kernel_size"),(5,"filter")]

    }
    
    return HttpResponse(template.render(context, request))

def setparameter(request,project_id):
    print(request.POST)
    return HttpResponseRedirect('/ai/'+str(project_id))

def savemodelweights(experiment_id,model):
    import pickle
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    LayerWeights.objects.filter(experiment=experiment_id).all().delete()
    for layer in model.layers:
        curName = layer.name
        if curName.startswith("tobesaved_"):
            weights = layer.get_weights()
            ser_weights = pickle.dumps(weights,protocol=0)
            lw = LayerWeights(name=layer.name,weights=ser_weights,experiment=experiment)
            lw.save()

def loadmodelweights(experiment_id,model):
    import pickle
    import sys, traceback
    lws = LayerWeights.objects.filter(experiment=experiment_id).all()
    for lw in lws:
        weights = pickle.loads(eval(lw.weights))
        try:
            modellayer = model.get_layer(lw.name)
            if modellayer:
                modellayer.set_weights(weights)
                
        except:
            print("Could not load model layer with name: "+str(lw.name))        
            for layer in model.layers:
                print(layer.name)
            traceback.print_exc(file=sys.stdout)


def writetodessa(request,project_id,experiment_id):
    import json
    project = get_object_or_404(Project, pk=project_id)
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    
    createProjectDessa(project_id)

    model = prepareModel(project,experiment)
    saveModelInProjectFolder(model,project_id)

    train_df3 = mysite.dataoperation.getTransformedData(project_id,mysite.dataoperation.TRAIN_DATA_QUALIFIER)
    test_df3 = mysite.dataoperation.getTransformedData(project_id,mysite.dataoperation.TEST_DATA_QUALIFIER)
    saveParquetDataInProjectFolder(train_df3,project_id,qualifier=1)
    saveParquetDataInProjectFolder(test_df3,project_id,qualifier=2)


    return HttpResponse(json.dumps({}), content_type='application/json')


def getMainDir():
    return "D:\\Entwicklung\kuai" #TODO: be configurable

def getProjectDir(project_id):
    return getMainDir()+"\\projects\\"+str(project_id)

def getProjectDataDir(project_id):
    return getProjectDir(project_id)+"\\"+"data"

def createProjectDessa(project_id):
    import os
    import subprocess
    import foundations
    maindir = getMainDir()
    projectdir = getMainDir()+"\\projects"
    os.chdir(projectdir)
    
    if str(project_id) not in os.listdir():
        #os.mkdir(str(project_id))
        #os.chdir(str(project_id))
        proc = subprocess.Popen(['foundations', 'init',str(project_id)], stdout=subprocess.PIPE, shell=True)
        proc.wait()
    os.chdir(str(project_id))
    if "data" not in os.listdir():
        os.mkdir("data")
    if "model" not in os.listdir():
        os.mkdir("model")
    os.chdir(maindir)


def saveModelInProjectFolder(model,project_id):
    model.save(getProjectDir(project_id)+"\\model\\")

def saveParquetDataInProjectFolder(dataframe,project_id,qualifier=1):
    import os
    dataframe.write.mode("overwrite").parquet(getProjectDataDir(project_id)+"\\"+str(qualifier)+".parquet")
    
def submitDessaJob(project_id):
    try:
        import foundations
        foundations.submit(job_directory=getProjectDir(project_id),command=["main.py"])
    except Exception as e:
        print(e)