class PlainPythonExperiment:
    
    projectfolder = None
    experimentfile = None
    executionText = None
    logfile = None
    configuration = None
    csvLoggingFile = None
    

    def __init__(self,projectfolder,experimentfile,logfile,executionText=None):
        self.projectfolder = projectfolder
        self.experimentfile = experimentfile
        self.logfile = logfile
        self.executionText = executionText
        self.csvLoggingFile = experimentfile+".csv"
        self.csvLoggingFileTest = experimentfile+"test.csv"


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
            
        try:
            metricsString = reduce(lambda x,y:x+","+y,map(lambda x:"'"+x+"'",self.configuration['metrics']))
        except Exception as e:
            print(e)
            metricsString = ""

        defaultworker = re.sub("#overwrite model if needed optimizer = keras.optimizer.Adam\(\);model.compile\(optimizer=optimizer,loss=loss,metrics=metrics\)#change optimizer",
            "optimizer = keras.optimizers."+self.configuration['Optimizer']['selection']+"("+optionsString+");model.compile(optimizer=optimizer,loss='"+self.configuration['loss']+"',metrics=["+metricsString+"])",
            defaultworker)

        defaultworker = re.sub(r'.*csvlogger\b','csv_callback = keras.callbacks.CSVLogger("'+self.csvLoggingFile+'", separator=",", append=False)#csvlogger',defaultworker)
        defaultworker = re.sub(r'.*csvloggertesting\b','csv_callback_test = keras.callbacks.CSVLogger("'+self.csvLoggingFileTest+'", separator=",", append=False)#csvloggertesting',defaultworker)


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
        self.executionText = defaultworker
        print("write")
    
    def executeExperiment(self):
        print("execute")
        assert self.executionText != None
        assert self.projectfolder != None
        assert self.experimentfile != None
        assert self.logfile != None
        
        if self.executionText:
            f2 = open(self.projectfolder+"/"+self.experimentfile, "w")
            f2.write(self.executionText)
            f2.close()
            import subprocess
            f = open(self.logfile,"a")
            proc = subprocess.Popen(['python', self.projectfolder+"/"+self.experimentfile], stdout=f,stderr=f, shell=True,cwd=self.projectfolder)
            proc.wait()
            return proc
        else:
            print("not complied yet")
            return None
    
    def getLog(self):
        try:
            f = open(self.logfile, "r")
            curExperimentLog = f.read()
        except Exception as e:
            curExperimentLog = ""
        return curExperimentLog

    def getResults(self):
        import pandas as pd
        try:
            dataframe = pd.read_csv(self.projectfolder+"/"+self.csvLoggingFile)
            output_metric_names = list(map(lambda x:{"name":"train"+x, "type": "array number"},list(filter(lambda x:x!="epoch",list(dataframe.columns)))))
            
            output_metrics = []
            for a in output_metric_names:
                element = {"name":a['name']}
                element['value'] = list(dataframe[a['name'][5:]])
                output_metrics.append(element)
        except Exception as e:
            output_metrics = []
            output_metric_names = []
        
        try:   
            dataframe_test = pd.read_csv(self.projectfolder+"/"+self.csvLoggingFileTest)
            output_metric_names_test = list(map(lambda x:{"name":"test"+x, "type": "array number"},list(filter(lambda x:x!="epoch",list(dataframe_test.columns)))))
            output_metrics_test = []
            for a in output_metric_names_test:
                element = {"name":a['name']}
                element['value'] = list(dataframe_test[a['name'][5:]])
                output_metrics_test.append(element)
        except Exception as e:
            output_metrics_test = []
            output_metric_names_test = []
            
        result = {
            "name":"1",
            "parameters":[],
            "jobs":[{
                "job_id":"1",
                "user":"",
                "project":"",
                "job_parameters":"",
                "output_metrics":output_metrics+output_metrics_test,
            }],
            "output_metric_names":output_metric_names+output_metric_names_test

        }
            
        return [result,""]
        
    def serialize(self):
        result = {}
        result['executionText'] = self.executionText
        result['projectfolder'] = self.projectfolder
        result['experimentfile'] = self.experimentfile
        result['logfile'] = self.logfile
        import json
        return json.dumps(result)
    
    @classmethod
    def deserialize(cls,json_dict):
        import json
        para = json.loads(json_dict)
        return cls(para['projectfolder'],para['experimentfile'],para['logfile'],para['executionText'])

    def update(self,newexecutiontext):
        self.executionText = newexecutiontext

    