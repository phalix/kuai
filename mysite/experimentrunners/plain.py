class PlainPythonExperiment:
    
    projectfolder = None
    experimentfile = None
    executionText = None
    logfile = None
    configuration = None
    

    def __init__(self,projectfolder,experimentfile,logfile):
        self.projectfolder = projectfolder
        self.experimentfile = experimentfile
        self.logfile = logfile

    def setupExperiment(self,configuration):
        required = {'epochs':10,
                    'BatchSize':10,
                    'Optimizer':{"selection":"Adam",\
                        "options":{"learning_rate":0.01}
                        },
                    "loss":"MAE",
                    "metrics":["MSE","Accuracy"]
                    }
        self.configuration = configuration

        for requirement in required.keys():
            if requirement not in self.configuration:
                self.configuration[requirement] = required[requirement]
        
        print("setup")
    
    def showExperiment(self):
        print(self.executionText)

    def getExperiment(self):
        return self.executionText

    def writeExperiment(self):

        import os
        import subprocess
        path = os.path.abspath(self.projectfolder)
        os.makedirs(path,exist_ok=True)
        
        olddir = os.getcwd()
        maindir = olddir
        projectdir = path
        os.chdir(path)
        import shutil
        
        if "data" not in os.listdir():
            os.mkdir("data")
        if "model" not in os.listdir():
            os.mkdir("model")
        
        shutil.copyfile(maindir+"/DessaWorker/PyArrowDataExtraction.py",projectdir+"/PyArrowDataExtraction.py")
        shutil.copyfile(maindir+"/DessaWorker/DessaCallback.py",projectdir+"/DessaCallback.py")
        shutil.copyfile(maindir+"/DessaWorker/requirements.txt",projectdir+"/requirements.txt")

        os.chdir(olddir)

        f = open("./DessaWorker/DefaultWorker.py", "r")
        defaultworker = f.read()
        
        import re
        re.compile("prjdir = ''#prjdir")
        #defaultworker = re.sub("prjdir = '.'#prjdir","prjdir = '"+prjdir+"'#prjdir",defaultworker)
        defaultworker = re.sub("epochs = 5 #epochs","epochs = "+str(self.configuration['epochs'])+" #epochs",defaultworker)
        defaultworker = re.sub("batch_size = 1 #batchsize","batch_size = "+str(self.configuration['BatchSize'])+" #batchsize",defaultworker)
        
        confString = ""
        from functools import reduce
        optionsString = ""
        
        myMapWithOptions = list(map(lambda x:str(x[0])+"="+str(x[1]),self.configuration['Optimizer']['options'].items()))
        if len(myMapWithOptions) > 0:
            optionsString = reduce(lambda x,y:x+";"+y,myMapWithOptions)
            
        
        metricsString = reduce(lambda x,y:x+","+y,map(lambda x:"'"+x+"'",self.configuration['metrics']))
        

        defaultworker = re.sub("#overwrite model if needed optimizer = keras.optimizer.Adam\(\);model.compile\(optimizer=optimizer,loss=loss,metrics=metrics\)#change optimizer",
            "optimizer = keras.optimizers."+self.configuration['Optimizer']['selection']+"("+optionsString+");model.compile(optimizer=optimizer,loss='"+self.configuration['loss']+"',metrics=["+metricsString+"])",
            defaultworker)

        

        withoutDessa = ["foundations.set_tensorboard_logdir(logdir)#foundations",
                        "import foundations#foundations",
                        "callbacks.append(dc.CustomDessaCallback())#foundations",
                        "callbacksEvaluate.append(dc.CustomDessaCallback())#foundations",
                        "tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)#tensorboard",
                        "foundations.set_tensorboard_logdir(logdir)#foundations",
                        "callbacks.append(tensorboard_callback)#tensorboard",
                        'callbacksEvaluate.append(dc.CustomDessaCallback("test"))#foundations',
                        'callbacks.append(dc.CustomDessaCallback("train"))#foundations']

        for wd in withoutDessa:
            #defaultworker = re.sub(wd,"",defaultworker)
            defaultworker = defaultworker.replace(wd,"")
        
        f.close()
        f2 = open(self.projectfolder+"/"+self.experimentfile, "w")
        f2.write(defaultworker)
        f2.close()
        self.executionText = defaultworker
        print("write")
    
    def executeExperiment(self):
        print("execute")
        import subprocess
        f = open(self.logfile,"a")
        proc = subprocess.Popen(['python', self.projectfolder+"/"+self.experimentfile], stdout=f,stderr=f, shell=True,cwd=self.projectfolder)
        return proc
        
    def getLog(self):
        f = open(self.logfile, "r")
        curExperimentLog = f.read()
        return curExperimentLog

    def getResults(self):
        return None