from mysite.experimentrunners.plain import PlainPythonExperiment

class AtlasDessaExperiment(PlainPythonExperiment):
    
    projectfolder = None
    experimentfile = None
    executionText = None
    logfile = None
    configuration = None
    projectid = None
    

    def __init__(self,projectfolder,experimentfile,logfile,project_id,executionText=None):
        self.projectfolder = projectfolder
        self.experimentfile = experimentfile
        self.logfile = logfile
        self.executionText = executionText
        self.projectid = project_id

    def show(self):
        print("show")


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
        
        base,current = os.path.split(path)
        os.chdir(base)
        proc = subprocess.Popen(['foundations', 'init',current], stdout=subprocess.PIPE, shell=True)
        proc.wait()
        os.chdir(path)
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

        f.close()
        self.executionText = defaultworker
        f2 = open(self.projectfolder+"/"+self.experimentfile, "w")
        f2.write(defaultworker)
        f2.close()
        print("write")
    
    def executeExperiment(self):
        print("execute")
        try:
            if self.executionText:
                f2 = open(self.projectfolder+"/"+self.experimentfile, "w")
                f2.write(self.executionText)
                f2.close()
                import foundations
                foundations.submit(scheduler_config='scheduler',job_directory=self.projectfolder,command=[self.experimentfile])
                print("started")
            else:
                print("not complied yet")
                return None
        except Exception as e:
            print(e)
        return None
        
    @classmethod
    def deserialize(cls,json_dict):
        import json
        para = json.loads(json_dict)
        return cls(para['projectfolder'],para['experimentfile'],para['logfile'],para['executionText'])

    def getResults():
        import mysite.project as mp
        urlatlasdessa = mp.getSetting(self.projectid,'DessaServer')[0]
        
        import requests
        results_dessa = requests.get(urlatlasdessa+"/foundations_rest_api/api/v2beta/projects/"+str(self.projectid)+"/job_listing")
        import json
        if results_dessa.text == '"This project was not found"\n':
            result_json = {"error":results_dessa.text}
            name = []
            jobs = []
            metrics = []
            parameters = []
        else:
            results_json = json.loads(results_dessa.text)
            name = results_json['name']
            jobs = results_json['jobs']
            metrics = results_json['output_metric_names']
            parameters = results_json['parameters']
        
        return [result_json,urlatlasdessa]