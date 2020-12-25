from django.db import models

class Statistics(models.Model):
    best_loss = models.DecimalField(decimal_places=4,max_digits=99)

class DataModels(models.Model):
    traindata = models.BinaryField(max_length=None,null=True)
    testdata = models.BinaryField(max_length=None,null=True)
    cvdata = models.BinaryField(max_length=None,null=True)
    originaldataset = models.BinaryField(max_length=None,null=True)
    

class Feature(models.Model):
    fieldname = models.CharField(max_length=99,null=True)
    transition = models.IntegerField(null=True)
    reformat = models.TextField(null=True)
    type = models.CharField(max_length=15,null=False)
    dimension = models.CharField(max_length=50,null=True)

class Configuration(models.Model):
    fieldname = models.CharField(max_length=99,null=False)
    option = models.CharField(max_length=99,null=False)
    parameterfixed = models.BooleanField(default=False)

class Layer(models.Model):
    index = models.IntegerField()
    inputlayer = models.BooleanField(default=False)
    outputlayer = models.BooleanField(default=False)
    layertype = models.CharField(max_length=50)
    inputlayers = models.ManyToManyField('self')
    configuration = models.ManyToManyField(Configuration)



class Optimizer(models.Model):
    name = models.CharField(max_length=50)
    configuration = models.ManyToManyField(Configuration)

class NeuralNetwork(models.Model):
    layers = models.ManyToManyField(Layer)
    optimizer = models.ForeignKey(Optimizer, on_delete=models.DO_NOTHING,null=True)
    

class ParameterFixation(models.Model):
    parameter = models.ForeignKey(Configuration, null=True,on_delete=models.CASCADE)
    configuration = models.ManyToManyField(Configuration,related_name="configuration")

class CallbackSetup(models.Model):
    name = models.CharField(max_length=50)
    configuration = models.ManyToManyField(Configuration)

class Project(models.Model):
    execution_text = models.CharField(max_length=9999,null=True)
    author = models.CharField(max_length=100)
    datacreated = models.DateField()
    projectname = models.CharField(max_length=200)
    projectstatistics = models.ForeignKey(Statistics, on_delete=models.DO_NOTHING,null=True)
    datamodel = models.ForeignKey(DataModels, on_delete=models.DO_NOTHING,null=True)
    features = models.ManyToManyField(Feature,related_name="features")
    targets = models.ManyToManyField(Feature,related_name="targets")
    target = models.ForeignKey(Feature, on_delete=models.DO_NOTHING,null=True,related_name="target")
    neuralnetwork = models.ForeignKey(NeuralNetwork, on_delete=models.DO_NOTHING,null=True)
    appendFeature = models.ForeignKey(Feature,on_delete=models.CASCADE,null=True)
    input = models.CharField(max_length=250)
    selectstatement = models.TextField(null=True)
    udfclasses = models.TextField(null=True)
    configuration = models.ManyToManyField(Configuration)
    

class UDF(models.Model):
    input = models.CharField(max_length=350)
    udfexecutiontext = models.TextField(null=False)
    outputtype = models.CharField(max_length=350)
    project = models.ForeignKey(Project,on_delete=models.CASCADE,null=False)


class Loss(models.Model):
    name = models.CharField(max_length=150)
    

class Experiment(models.Model):
    parameters = models.ManyToManyField(ParameterFixation)
    setup = models.ManyToManyField(CallbackSetup)
    project = models.ForeignKey(Project,on_delete=models.CASCADE,null=True)
    status = models.IntegerField() #0 = init, 1 = running, 2 = to stop, 3 = stopped
    loss = models.ForeignKey(Loss,on_delete=models.DO_NOTHING,null=True)
    noofepochs = models.IntegerField(null=True)
    batchsize = models.IntegerField(null=True)
    optimizer = models.ForeignKey(Optimizer, on_delete=models.DO_NOTHING,null=True)

class LayerWeights(models.Model):
    name = models.CharField(max_length=150)
    weights = models.CharField(max_length=999)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,null=True)

class Metrics(models.Model):
    name = models.CharField(max_length=150)
    experiment = models.ForeignKey(Experiment,on_delete=models.CASCADE,null=True)



class Result(models.Model):
    configuration = models.ManyToManyField(Configuration)
    epochno = models.IntegerField()
    experiment = models.ForeignKey(Experiment,on_delete=models.CASCADE,null=True)

class Sample(models.Model):
    name = models.CharField(max_length=50)
    number = models.FloatField(max_length=50)
    batch = models.IntegerField()
    source = models.IntegerField() #1 = train, #2 = test, #3 = cross validation
    result = models.ForeignKey(Result,on_delete=models.CASCADE,null=True)    

