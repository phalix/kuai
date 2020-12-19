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

#librarystring = 'from tensorflow.python import keras'
#exec(librarystring)
librarystring_a = 'from tensorflow import python as pyt'
exec(librarystring_a)
librarystring2 = 'keras.layers'
librarystring3 = 'keras.optimizers'
librarystring4 = 'keras.metrics'




def modelsummary(request,project_id):
    #TODO: split this function, into
    # add feature transition!
    # feature extraction
    # model training
    template = loader.get_template('ai/modelsummary.html')
    project = get_object_or_404(Project, pk=project_id)
    inputs = mysite.neuralnetwork.getinputshape(project)
    neuralnetwork = getNeuralNetworkStructureAsPlainPython(project)
    optimizer = getOptimizerAsPlainPython(project)
    model = buildmodel(project,neuralnetwork,optimizer,list(project.features.all()),'mse',['accuracy'],project.target.fieldname,inputs)
    
    iputs = model.inputs
    for iput in iputs:
        print(iput.shape)
    oputs = model.outputs
    for oput in oputs:
        print(oput.shape)

    datframeshapes = mysite.dataoperation.getDimensionsByProject(project)
    
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
        "menuactive":4,
        #"neuralnetwork": nn,
        #"layers": layers,
        "output":output,
        "modelinputs":iputs,
        "modeloutputs":oputs,
        "datainputs":datframeshapes[0],
        "dataoutputs":datframeshapes[1],



    }
    
    return HttpResponse(template.render(context, request))




def index(request,project_id):
    template = loader.get_template('ai/neuralnetworksetup.html')
    project = get_object_or_404(Project, pk=project_id)
    nn = None
    layers = None
    
    inputs = getinputshape(project)

    if project.neuralnetwork:
        nn = project.neuralnetwork
        if nn.layers:
            layers = nn.layers.filter(inputlayer=False).filter(outputlayer=False)
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

    # TODO: add semantics to parameters
    return HttpResponseRedirect('/optimizer/'+str(project_id))

def aiupload(request,project_id):
    #do not delete inputlayer, change on data classification
    project = get_object_or_404(Project, pk=project_id)
    if project.neuralnetwork:
        nn = project.neuralnetwork
        nn.layers.filter(outputlayer=False).delete()
    else:
        nn = NeuralNetwork()
    nn.save()
    project.neuralnetwork = nn
    project.save()

    #input layer needs to be defined
    inputs = getinputshape(project)
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
    template = loader.get_template('ai/optimizer.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        #"layer_types" : getkeraslayeroptions(),
        "menuactive":4,

    }
    
    return HttpResponse(template.render(context, request))

def optimizer(request,project_id):
    template = loader.get_template('ai/neuralnetworkoptandout.html')
    project = get_object_or_404(Project, pk=project_id)
    optimizername = ""
    availablelayers = []
    inputs = getinputshape(project)
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
    import keras
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

def getcurrentmodel(project,loss,metrics,neuralnetwork,optimizer,features,target,inputschema):
    currentmodel = buildmodel(project,neuralnetwork,optimizer,features,loss,metrics,target,inputschema)
    return currentmodel


def getinputshape(project):
    import keras
    inputs = {}
    project_id = project.id
    #cur_features = project.features.all()
    inputshapes = mysite.dataoperation.getinputschema(project_id)
    idx = 1
    for (key,value) in inputshapes.items():
        name=("Input"+str(idx)+"_"+str(key)).replace(" ","")
        shape = None
        if type(value) == list and type(value[0]) == list:
            shape = value[0]
        else:
            shape = value
        inputs[idx] = keras.Input(shape=shape,name=name)
        idx = idx + 1
    return inputs


def getNeuralNetworkStructureAsPlainPython(project):
    return list(map(lambda x: {\
        'idx':x.index,
        'isinput':x.inputlayer,
        'isoutput':x.outputlayer,
        'type':x.layertype,
        'inputs':list(map(lambda y:y.index,list(x.inputlayers.filter(inputlayer=True).all()))), 
        'intermediates':list(map(lambda y:y.index,list(x.inputlayers.filter(inputlayer=False).all()))), 
        'conf': dict(map(lambda y: [y.fieldname,y.option] ,list(x.configuration.all()))),
        },list(project.neuralnetwork.layers.all())))

def getOptimizerAsPlainPython(project):
    a = {}
    if project.neuralnetwork.optimizer and project.neuralnetwork.optimizer.configuration:
        a['conf'] = dict(map(lambda x: [x.fieldname,x.option],list(project.neuralnetwork.optimizer.configuration.all())))
        a['name'] = project.neuralnetwork.optimizer.name
    else:
        a['name'] = 'SGD'
        a['conf'] = {}
    return a

## generate keras model from database definition
def buildmodel(project,neuralnetwork,optimizer,features,loss,metrics,target,inputschema):
    import keras
    nn = neuralnetwork
    
    #Build Tensorflow Model
    inputs = inputschema#getinputshape(project)
    previouslayers = {}
    
    alllayersnoinput = list(filter(lambda x:x['isinput']==False,neuralnetwork))

    #for layer in nn.layers.filter(inputlayer=False):
    for layer in alllayersnoinput:
        calldict = {}
        configuration = layer['conf']
        #for attr in layer.configuration.all():
        for key,value in configuration.items():
            option = None
            try:
                option = eval(value)
                calldict[key] = option
            except:
                option = value
                if len(option)>0:
                    calldict[key] = option
        calldict["name"]= "tobesaved_"+str(layer['idx'])  
        #y.name = "tobesaved_"+str(layer.id)    
            
        #initiate by actual predecessor
        y = getattr(keras.layers, layer['type'])(**calldict)
        '''inputsinput = []
        for inpim in layer.inputlayers.filter(inputlayer=True):
            inputsinput.append(inpim.index)
        intermedinputs = []
        for inpim in layer.inputlayers.filter(inputlayer=False):
            intermedinputs.append(inpim.index)
        
        previouslayers[layer.index] = (y,inputsinput,intermedinputs,layer.id)'''
        previouslayers[layer['idx']] = (y,layer['inputs'],layer['intermediates'],layer['idx'])
    
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
    
    #configuration = nn.optimizer.configuration.all()
    configuration = optimizer['conf']
    configdict = {}
    for key,value in configuration.items():
        try:
            option = eval(value)
            configdict[key] = option
        except:
            option = value
            if len(option)>0:
                configdict[key] = value
    #optimizer = eval(librarystring3+"."+nn.optimizer.name)(**configdict)
    optimizer = eval(librarystring3+"."+optimizer['name'])(**configdict)

    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    

    return model