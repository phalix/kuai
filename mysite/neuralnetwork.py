from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404

from datetime import datetime

from django import forms

from .models import Project,NeuralNetwork,Layer,Configuration,Optimizer,Experiment,Result,Sample

import mysite.dataoperation

#import tensorflow as tf
import numpy as np
import inspect
import keras
#librarystring = 'from tensorflow.python import keras'
#exec(librarystring)
librarystring_a = 'from tensorflow import python as pyt'
exec(librarystring_a)
librarystring2 = 'keras.layers'
librarystring3 = 'keras.optimizers'
librarystring4 = 'keras.metrics'

def modelrunwithspark(request,project_id):
    template = loader.get_template('modelsummary.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        "menuactive":4,
        #"layer_types" : getkeraslayeroptions(librarystring2),
        #"neuralnetwork": nn,
        #"layers": layers

    }
    
    df = mysite.dataoperation.readfromcassandra(project_id)
    import sparkdl
    from sparkdl import KerasTransformer


    # Parameters
    SIZE = (299, 299)  # Size accepted by Inception model
    IMAGES_PATH = 'datasets/image_classifier/test/'  # Images Path
    MODEL = '/tmp/model-full-tmp.h5'  # Model Path

    

    # Define Spark Transformer
    transformer = KerasTransformer(inputCol="uri", outputCol="predictions",
                                    modelFile=MODEL,
                                    outputMode="vector")

    #TODO: Write Module for importing exporting models with mongo
    #uri_df = dataoperation.readfromcassandra(project_id)
    # Get Output
    #labels_df = transformer.transform(uri_df)
    # Show Output
    #labels_df.show()


    return HttpResponse(template.render(context, request))


def modelsummary(request,project_id):
    #TODO: split this function, into
    # add feature transition!
    # feature extraction
    # model training
    template = loader.get_template('modelsummary.html')
    
    model = buildmodel(project_id,'mse',['accuracy'])
    project = get_object_or_404(Project, pk=project_id)

    import pandas as pd
    #from sklearn.linear_model import LogisticRegression
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    
    from pyspark.sql.types import StructField,StructType,StringType,DoubleType

    counter = 0
    
    from pyspark import SparkConf
    
    import io
    from contextlib import redirect_stdout
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        output = buf.getvalue()
    global currentmodel
    currentmodel = model
    context = {
        "project" : project,
        "project_id" : project_id,
        #"layer_types" : getkeraslayeroptions(librarystring2),
        #"menuactive":4,
        #"neuralnetwork": nn,
        #"layers": layers,
        "output":output,

    }
    
    return HttpResponse(template.render(context, request))




def index(request,project_id):
    template = loader.get_template('neuralnetworksetup.html')
    project = get_object_or_404(Project, pk=project_id)
    nn = None
    layers = None
    
    inputs = getinputshape(project_id)

    if project.neuralnetwork:
        nn = project.neuralnetwork
        if nn.layers:
            layers = nn.layers.filter(inputlayer=False)
    if layers:
        for l in layers:
            iputs = []
            for inp in l.inputlayers.filter(inputlayer=False):
                iputs.append(str(inp.index))
            for inp in l.inputlayers.filter(inputlayer=True):
                curIndex = inp.index
                shapes = inp.configuration.filter(fieldname="shape")
                curDimensions = ""
                for s in shapes:
                    curDimensions = s.option
                iputs.append("Input"+str(curIndex)+"_"+str(curDimensions))
            l.input = iputs
            confs = {}
            for conf in l.configuration.all():
                confs[conf.fieldname] = conf.option
            l.confs = l.configuration.all()

    
    
    
    context = {
        "project" : project,
        "project_id" : project_id,
        "layer_types" : getkeraslayeroptions(librarystring2),
        "menuactive":4,
        "neuralnetwork": nn,
        "layers": layers,
        "inputs": inputs,
    }

    
    
    
    return HttpResponse(template.render(context, request))

def aioptandoutupload(request,project_id):
    #print(request.POST)
    optiname = ""
    
    indices = []
    layersdict = {}
    layersoption = {}


    project = get_object_or_404(Project, pk=project_id)
    
    nn = project.neuralnetwork
    if not nn:
        nn = NeuralNetwork()
        nn.save()
        project.neuralnetwork = nn
    nn.layers.filter(outputlayer=True).delete()

    opticonfigs = []

    for para in request.POST:
        if para.startswith("layer"):
            indices.append(int(para[5:]))    
        if para.startswith("para"):
            splittedstring = para.split("$")
            index = int(splittedstring[0][4:])
            field = str(splittedstring[1].split("%")[0])
            
            value = str(request.POST[para])
            
            curConf = Configuration(fieldname=field,option=value)
            curConf.save()
            if index not in layersoption:
                layersoption[index] = []
            curOptions = layersoption[index]
            curOptions.append(curConf)
            layersoption[index] = curOptions
        if para.startswith("optpara"):
            optifieldname = para.split("$")[1]
            optivalue = request.POST[para]
            if optivalue and len(optivalue)>0:
                curopticonf = Configuration(fieldname=optifieldname,option=optivalue)
                curopticonf.save()
                opticonfigs.append(curopticonf)
        if para == "optimizerselect":
            optiname = request.POST[para]

    for index in indices:
        layerandindex = "layer"+str(index)
        curstates = request.POST.getlist("states[]"+str(index))
        curlayertype = request.POST["layer"+str(index)]
        curLayer = Layer(index=index,layertype=curlayertype,outputlayer=True)
        curLayer.save()
        layersdict[index] = curLayer
        nn.layers.add(curLayer)

      
    
    for index in indices:
        curLayer = layersdict[index]
        curstates = request.POST.getlist("states[]"+str(index))
        for state in curstates:
            if state.startswith('Input'):
                inpidx = int(state[5:].split("_")[0])
                iputlayers = nn.layers.filter(index=inpidx,outputlayer=False,inputlayer=True)
                for iputlayer in iputlayers:
                    curLayer.inputlayers.add(iputlayer)
            else:
                inmlayers = nn.layers.filter(index=state,outputlayer=False,inputlayer=False)
                for inmlayer in inmlayers:
                    curLayer.inputlayers.add(inmlayer)
        
        if index in layersoption:
            for option in layersoption[index]:
                #print(option)
                #print(option.fieldname)
                #print(option.option)
                curLayer.configuration.add(option)
                option.save()
        curLayer.save()



    selopti = Optimizer(name=optiname)
    selopti.save()
    for oc in opticonfigs:
        selopti.configuration.add(oc)
        oc.save()
    selopti.save()    
    
    nn.optimizer = selopti
    nn.save()
    project.save()

    # TODO: save to database
    # TODO: save output layer
    # TODO: save optimizer parameters
    # TODO: add semantics to parameters
    return HttpResponseRedirect('/optimizer/'+str(project_id))

def aiupload(request,project_id):
    project = get_object_or_404(Project, pk=project_id)
    if project.neuralnetwork:
        nn = project.neuralnetwork
        nn.layers.all().delete()
    else:
        nn = NeuralNetwork()
    nn.save()
    project.neuralnetwork = nn
    project.save()

    #input layer needs to be defined
    inputs = getinputshape(project_id)
    inputlayers = {}
    for (inp,value) in inputs.items():
        inplayer = Layer(index=inp,inputlayer=True)
        inplayer.save()
        curConf = Configuration(fieldname="shape",option=value.shape)
        curConf.save()
        inplayer.configuration.add(curConf)
        inplayer.save()
        curConf = Configuration(fieldname="name",option=value.name)
        curConf.save()
        inplayer.configuration.add(curConf)
        inplayer.save()
        nn.layers.add(inplayer)
        nn.save()
        inputlayers[inp] = inplayer



    indices = []
    layersdict = {}
    layersoption = {}
    
    for para in request.POST:
        if para.startswith("layer"):
            indices.append(int(para[5:]))    
        if para.startswith("para"):
            splittedstring = para.split("$")
            index = int(splittedstring[0][4:])
            field = str(splittedstring[1].split("%")[0])
            
            value = str(request.POST[para])
            
            curConf = Configuration(fieldname=field,option=value)
            curConf.save()
            if index not in layersoption:
                layersoption[index] = []
            curOptions = layersoption[index]
            curOptions.append(curConf)
            layersoption[index] = curOptions




    for index in indices:
        layerandindex = "layer"+str(index)
        curstates = request.POST.getlist("states[]"+str(index))
        curlayertype = request.POST["layer"+str(index)]
        curLayer = Layer(index=index,layertype=curlayertype)
        curLayer.save()
        layersdict[index] = curLayer
        nn.layers.add(curLayer)

      
    
    for index in indices:
        curLayer = layersdict[index]
        curstates = request.POST.getlist("states[]"+str(index))
        for state in curstates:
            if state.startswith('Input'):
                inpidx = int(state[5:].split("_")[0])
                print("input_________")
                print(inpidx)
                inplayer = inputlayers[inpidx]
                curLayer.inputlayers.add(inplayer)
            else:
                if int(state) in layersdict:
                    curState = layersdict[int(state)]
                    #print(curState)
                    curLayer.inputlayers.add(curState)
        
        if index in layersoption:
            for option in layersoption[index]:
                #print(option)
                #print(option.fieldname)
                #print(option.option)
                curLayer.configuration.add(option)
                option.save()
        curLayer.save()

    nn.save()
    return HttpResponseRedirect('/ai/'+str(project_id))

def parameter(request,project_id):
    template = loader.get_template('optimizer.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        #"layer_types" : getkeraslayeroptions(),
        "menuactive":4,

    }
    
    return HttpResponse(template.render(context, request))

def optimizer(request,project_id):
    template = loader.get_template('neuralnetworkoptandout.html')
    project = get_object_or_404(Project, pk=project_id)
    optimizername = ""
    availablelayers = []
    inputs = getinputshape(project_id)
    optconfs = []
    if project.neuralnetwork:
        for l in project.neuralnetwork.layers.filter(inputlayer=False,outputlayer=False):
            availablelayers.append(l.index)
        if project.neuralnetwork.optimizer:
            optimizername = project.neuralnetwork.optimizer.name
            optconfs = project.neuralnetwork.optimizer.configuration.all()


   
    layers = None
    if project.neuralnetwork:
        nn = project.neuralnetwork
        if nn.layers:
            layers = nn.layers.filter(inputlayer=False,outputlayer=True)
    if layers:
        for l in layers:
            iputs = []
            for inp in l.inputlayers.filter(inputlayer=False):
                iputs.append(str(inp.index))
            for inp in l.inputlayers.filter(inputlayer=True):
                print(inp.index)
                curIndex = inp.index
                shapes = inp.configuration.filter(fieldname="shape")
                curDimensions = ""
                for s in shapes:
                    curDimensions = s.option
                print(curDimensions)
                iputs.append("Input"+str(curIndex)+"_"+str(curDimensions))
            l.input = iputs
            confs = {}
            for conf in l.configuration.all():
                confs[conf.fieldname] = conf.option
            l.confs = l.configuration.all()
    


    context = {
        "project" : project,
        "project_id" : project_id,
        "layer_types" : getkeraslayeroptions(librarystring2),
        "optimizers" : getkeraslayeroptions(librarystring3),
        "metrics" : getkeraslayeroptions(librarystring4),
        "availablelayers" : availablelayers,
        "menuactive":4,
        "optimizername":optimizername,
        "optimizerconfiguration":optconfs,
        "inputs":inputs,
        "layers":layers,

    }
    
    return HttpResponse(template.render(context, request))

### helper functions to get json from python structure

def getkeraslayeroptions(libstr):
    result = []
    layers = inspect.getmembers(eval(libstr))
    for l in layers:
        arguments = []
        try:
            nameofclass = l[1]
            args = []
            if inspect.isclass(nameofclass):
                nameofclass = nameofclass.__init__
            parameter = inspect.signature(nameofclass).parameters
            arguments = list(parameter.keys())
            for argkey in arguments:
                arg = parameter[argkey]
                if(arg.name != "kwargs" and arg.name != "self"):
                    if(arg.default == inspect._empty):
                        o = (arg.name,None)
                    else:
                        o = (arg.name,arg.default)
                    args.append(o)

            result.append((l[0],args))
        except Exception as e: 
            print(e)
            arguments = []
            
    return result




currentmodel = None

def getcurrentmodel(project_id,loss,metrics):
    #global currentmodel
    #if not currentmodel:
    currentmodel = buildmodel(project_id,loss,metrics)
    return currentmodel


def getinputshape(project_id):
    inputs = {}
    project = get_object_or_404(Project, pk=project_id)
    cur_features = project.features.all()
    inputshapes = mysite.dataoperation.getfeaturedimensionbyproject(cur_features)
    idx = 1
    for (key,value) in inputshapes.items():
        
        for (k,v) in value.items():
            name=("Input"+str(idx)+"_"+str(k)).replace(" ","")
            inputs[idx] = keras.Input(shape=tuple(v),name=name)
            idx = idx + 1
    
    print(inputs)
    return inputs


## generate keras model from database definition
def buildmodel(project_id,loss,metrics):
    project = get_object_or_404(Project, pk=project_id)
    nn = project.neuralnetwork
    myfeatures = project.features.all()
    
    
    target_array = []
    target_array.append(project.target.fieldname)
    
    #Build Tensorflow Model
    inputs = getinputshape(project_id)
    previouslayers = {}
    for layer in nn.layers.filter(inputlayer=False):
        calldict = {}
        for attr in layer.configuration.all():
            #TODO: dynamictiy of parameters
            option = None
            try:
                option = eval(attr.option)
                calldict[attr.fieldname] = option
            except:
                option = attr.option
                if len(option)>0:
                    calldict[attr.fieldname] = option
        calldict["name"]= "tobesaved_"+str(layer.id)  
        #y.name = "tobesaved_"+str(layer.id)    
            
        #initiate by actual predecessor
        y = getattr(keras.layers, layer.layertype)(**calldict)
        inputsinput = []
        for inpim in layer.inputlayers.filter(inputlayer=True):
            inputsinput.append(inpim.index)
        intermedinputs = []
        for inpim in layer.inputlayers.filter(inputlayer=False):
            intermedinputs.append(inpim.index)
        
        previouslayers[layer.index] = (y,inputsinput,intermedinputs,layer.id)
    
    previouslayer_inst = {}
    removed = True
    while len(previouslayers)>0 and removed ==True:
        removed = False
        for layidx in previouslayers:
            cursel = []
            value = previouslayers[layidx]
            okay = True
            for x in value[2]:
                if x not in previouslayer_inst:
                    okay = False
                else:
                    cursel.append(previouslayer_inst[x])
            for x in value[1]:
                cursel.append(inputs[x])
            if len(cursel) == 1:
                previouslayer_inst[layidx] = value[0](cursel[0])
            else:
                previouslayer_inst[layidx] = value[0](cursel)
            
            y = previouslayer_inst[layidx]


    model = keras.Model(inputs=list(inputs.values()),outputs=y)
    
    configuration = nn.optimizer.configuration.all()
    configdict = {}
    for conf in configuration:
        try:
            option = eval(conf.option)
            configdict[conf.fieldname] = option
        except:
            option = conf.option
            if len(option)>0:
                configdict[conf.fieldname] = option
        

    optimizer = eval(librarystring3+"."+nn.optimizer.name)(**configdict)

    
    

    model_compilation = model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    

    return model