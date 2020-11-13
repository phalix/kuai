
from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404
from mysite.models import Project,Feature,NeuralNetwork
import pandas as pd

pd.set_option("display.max_colwidth",20)

import numpy as np
from datetime import datetime

import traceback

from django import forms

import mysite.featurepipelines as fp
import mysite.neuralnetwork
from pyspark.sql.functions import udf,col,lit
from pyspark.sql.types import StringType,DoubleType,DateType,ArrayType,FloatType,IntegerType




TRAIN_DATA_QUALIFIER = 1
TEST_DATA_QUALIFIER = 2
CV_DATA_QUALIFIER = 3

displaylimit = 10


def index(request,project_id):
    template = loader.get_template('datasetup.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3
    }


    return HttpResponse(template.render(context, request))

def transformdataframe(project,dataframe):
    from pyspark.sql import SQLContext
    from pyspark import StorageLevel

    spark = getsparksession(project.id,1)
    sqlContext = SQLContext(spark)
    try:
        df2 = sqlContext.sql("select * from transform_temp_table")

    except:
        traceback.print_exc()
        
        
        if project.selectstatement:
            tobeeval = "dataframe."+project.selectstatement
            print(tobeeval)
            
            try:
                exec(project.udfclasses)
                df2 = eval(tobeeval)
                df2.registerTempTable("transform_temp_table")
                #sqlContext.cacheTable("transform_temp_table",StorageLevel.DISK_ONLY)
                sqlContext.sql("CACHE TABLE transform_temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")

            except:
                traceback.print_exc()
                #project.selectstatement = ""
                #project.save()
                df2 = dataframe
        else:
            df2 = dataframe
    return df2

def createDataFrameHTMLPreview(dataframe):
    from pyspark.sql.functions import substring
    #dataframe.limit(displaylimit).toPandas().to_html()
    #remove id
    #cast to string, limit to 50
    removeid = filter(lambda x: x != "_id", dataframe.columns)
    limitto50 = map(lambda x:substring(col(x).cast("String").alias(x),0,100),removeid) 
    pandas = dataframe.limit(displaylimit).select(list(limitto50)).toPandas()
    html = pandas.to_html()
    

    return html

def setuptransformdata(request,project_id):
    template = loader.get_template('datatransformation.html')
    project = get_object_or_404(Project, pk=project_id)
    df = readfromcassandra(project_id,1).limit(displaylimit).cache()
    
    df2 = transformdataframe(project,df)
    
    b = createDataFrameHTMLPreview(df)
    try:
        a = createDataFrameHTMLPreview(df2)
    except:
        a = b
    
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3,
        "dataframe":b,
        "dataframe2":a, 
        "selectstatement":project.selectstatement,
        "udfclasses":project.udfclasses,
    }


    return HttpResponse(template.render(context, request))

def transformdata(request,project_id):
    selectstatement = request.POST["selectstatement"]
    udfclasses = request.POST["udfclasses"]

    if len(selectstatement)==0:
        selectstatement = 'select("*")'

    project = get_object_or_404(Project, pk=project_id)
    project.selectstatement = selectstatement
    project.udfclasses = udfclasses
    project.save()

    print(selectstatement)
    print(udfclasses)
    
    from pyspark.sql import SQLContext
    spark = getsparksession(project.id,1)
    sqlContext = SQLContext(spark)
    try:
        sqlContext.uncacheTable("transform_temp_table")
        df2 = sqlContext.sql("drop table transform_temp_table")
    except:
        print("nothing to drop")

    return HttpResponseRedirect('/transform/'+str(project_id))



def existingdatasetselection(request):
    authorname = request.POST['authorNameInput']
    projectname = request.POST['projectNameInput']
    p = Project(author=authorname,projectname=projectname,datacreated=datetime.now())
    p.save()
    return HttpResponseRedirect('/datasetup/'+str(p.pk))

def uploaddata(request,project_id):
    #subm_file = request.FILES['filewithdata']
    print(request.POST['folderfile'])
    shuffle = request.POST['shuffledata']
    trainshare = float(request.POST['trainshare'])
    testshare = float(request.POST['testshare'])
    cvshare = float(request.POST['cvshare'])
    subm_file = request.POST['folderfile']
    if request.POST['datatype'] == "csv":
        ###TODO: seperator must be configurable
        
        df = pd.read_csv(subm_file,sep=";")
        
        
        if shuffle:
            df = df.sample(frac=1)
        lendf = len(df)
        

        msk1 = np.random.rand(lendf) < trainshare
        msk2 = np.random.rand(lendf) < testshare/(1-trainshare) 
        msk3 = np.random.rand(lendf) < cvshare/(1-trainshare+testshare)
        
        traindata = df[msk1]
        testdata = df[msk2]
        cvdata = df[msk3]
        
        traindata.to_pickle('./'+str(project_id)+"tr.pickle")
        testdata.to_pickle('./'+str(project_id)+"te.pickle")
        cvdata.to_pickle('./'+str(project_id)+"cv.pickle")

        df.to_pickle('./'+str(project_id)+".pickle")
        
        savetocassandra(project_id,traindata,1)
        savetocassandra(project_id,testdata,2)
        savetocassandra(project_id,cvdata,3)
    elif request.POST['datatype'] == "img":
        from pyspark.ml.image import ImageSchema 
        
        spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
        df = ImageSchema.readImages(subm_file) 
        #if shuffle:
        #    df = df.sample(frac=1)
        splitted = df.randomSplit([trainshare,testshare,cvshare])
        #splitted[0].config(conf=createSparkConfig(1,TRAIN_DATA_QUALIFIER))
        #splitted[1].config(conf=createSparkConfig(1,TEST_DATA_QUALIFIER))
        #splitted[2].config(conf=createSparkConfig(1,CV_DATA_QUALIFIER))
        
        
        
        savetocassandra_writesparkdataframe(splitted[0])
        spark = getsparksession(project_id,TEST_DATA_QUALIFIER)
        
        savetocassandra_writesparkdataframe(splitted[1])
        spark = getsparksession(project_id,CV_DATA_QUALIFIER)
        savetocassandra_writesparkdataframe(splitted[2])

    from pyspark.sql import SQLContext
    spark = getsparksession(project_id,1)
    sqlContext = SQLContext(spark)
    try:
        sqlContext.uncacheTable("temp_table")
        df2 = sqlContext.sql("drop table temp_table")
    except:
        print("nothing to drop")

    return HttpResponseRedirect('/dataclassification/'+str(project_id))

def dataclassification(request,project_id):
    
    template = loader.get_template('dataclassification.html')
    project = get_object_or_404(Project, pk=project_id)
    nn = None
    if project.neuralnetwork:
        nn = project.neuralnetwork
    if not nn:
        nn = NeuralNetwork()
        nn.save()
        project.neuralnetwork = nn

    
    # TODO: Enable Text Preprocessing from Keras
    r_features = []
    cur_features = project.features.all()
    

    for feat in cur_features:
        if feat != None:
            r_features.append(feat.fieldname)

    if project.target != None:
        target = project.target.fieldname
    else:
        target = ""
    
    df = readfromcassandra(project_id,1)
    df = transformdataframe(project,df).cache()

    print(df.schema)
    
    from pyspark.sql import functions as F
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorIndexer
    print("Imports Done")
    indexed = applyfeaturetransition(df,cur_features,project.target)#.limit(displaylimit)
    #mytypes_filter = filter(lambda x:x[0]!='_id' and not x[0].endswith("Pre"),indexed.dtypes)
    mytypes_filter = filter(lambda x:x[0]!='_id',indexed.dtypes)
    mytypes = list(mytypes_filter)
    mytypes_dict  = dict(mytypes)
    
    mytypes = list(filter(lambda x:not (x[0]+"Piped" in mytypes_dict.keys()),mytypes ))
    def reformatPipedPre(x):
        if x[0].endswith("Piped"):
            a= x[0][:(len(x[0])-5)]
        else:
            a = x[0]
        return (a,x[1])

    mytypes = list(map(lambda x:reformatPipedPre(x),mytypes))

    print("Types done")
    
    
    try:
        firstrow = indexed.first().asDict()
        print("First row done")
        
        featurevector_df = buildFeatureVector(indexed,cur_features,project.target).limit(displaylimit)
        print("featurevector_df")
        featurevector = ""#featurevector_df.first().asDict()['features_array']
        print("featurevector")
        featurevector_df_html = ""
        featurevector_df_html = ""#createDataFrameHTMLPreview(mysite.neuralnetwork.generateTrainingDataFrame(project,featurevector_df))
        print("featurevector_df_html")
        dataframe_html = ""
        dataframe_html = createDataFrameHTMLPreview(indexed) 
        print("dataframe_html preview done")
        
            
    
    except:
        firstrow = df.first().asDict()
        featurevector = []
        featurevector_df_html = ""
        dataframe_html = ""
    
    for key in firstrow:
        firstrow[key] = str(firstrow[key])[0:50]

    print("provision done")

    context = {
        "project" : project,
        "project_id" : project_id,
        "featurenames": r_features,
        "features": cur_features,
        "target" : target,
        "targetfeature" : project.target,
        "columns" : df.columns,
        "menuactive":3,
        "types" : mytypes,
        "firstrow": firstrow,
        "pipelines": fp.indexPipelines(),
        "featurevector": featurevector,
        "dataframe":dataframe_html,#
        "dataframe_fv_tar": featurevector_df_html,
    }
    
    
    return HttpResponse(template.render(context, request))



def getinputschema(project_id):
    import pyspark
    import json
    #TODO: we need to add other dimensions in order to use pictures etc.
    #Right now, it is only possible to use flat object like string, integer, etc.
    project = get_object_or_404(Project, pk=project_id)
    
    if not project.input or len(project.input)==0:
        cur_features = project.features.all()
        df = readfromcassandra(project_id)

        indexed = applyfeaturetransition(df,cur_features,project.target)
        featurevector_df = buildFeatureVector(indexed,cur_features,project.target)
        features_array = featurevector_df.first().asDict()['features_array']
        result = {
            1:[len(features_array),],
            #2:(0,0),
            #3:(0,0,0),
        }

        #for element in df.schema:
        #    for feat in cur_features:
        #        if feat.fieldname == element.name:
        #            if type(element) == pyspark.sql.types.StructField:
        #                result[1] = (result[1][0]+1,)
                        
        #TODO: create code for 2 dim, 3 dim and 4 dim

        res_json_str = json.dumps(result)
        project.input = res_json_str
        project.save()
    else:
        result = json.loads(project.input)


    return result






def setupdataclassifcation(request,project_id):
    project = get_object_or_404(Project, pk=project_id)
    project.features.all().delete()
    project.input = ""
    project.save()
    #project.target.delete()
    
    targetselection = request.POST['targetselection']
    targettransition = request.POST['fttransition_'+targetselection]
    reformattransition = request.POST['ftreformat_'+targetselection]
    targettype = request.POST['fttype_'+targetselection]
    target = Feature(fieldname=targetselection,transition=targettransition,reformat=reformattransition,type=targettype)
    target.save()
    project.target = target
    
    for x in request.POST:
        if x.startswith('feature_') and x[8:]!=targetselection:
            print(x[8:])
            transition = request.POST['fttransition_'+x[8:]]
            reformat = request.POST['ftreformat_'+x[8:]]
            type = request.POST['fttype_'+x[8:]]
            curfeat = Feature(fieldname=x[8:],transition=transition,reformat=reformat,type=type)
            curfeat.save()
            project.features.add(curfeat)
            curfeat.save()
        if x.startswith('fttransition_') and x[8:]!=targetselection:
            print("transition setup")
            print(x)
            print(request.POST[x])
        if x.startswith('addfieldname_'):
            
            print(x[13:])
            print(request.POST["addfieldname_"+x[13:]])
            print(request.POST["addfieldrule_"+x[13:]])
            print(request.POST["addfieldtransition_"+x[13:]])
            print(request.POST["addfieldtype_1"+x[13:]])
            appFeat = Feature(fieldname=x[13:],reformat=request.POST["addfieldrule_"+x[13:]],transition=request.POST["addfieldtransition_"+x[13:]],type=request.POST["addfieldtype_1"+x[13:]])
            appFeat.save()
            project.appendFeature=appFeat

    
    project.save()
    return HttpResponseRedirect('/dataclassification/'+str(project_id))



#########################################
### non-django but Spark related things##
### TODO: move to extra class!!!       ##
#########################################





def savetocassandra(project_id,df,type=TRAIN_DATA_QUALIFIER):
    spark = getsparksession(project_id,type)
    sparkdf = spark.createDataFrame(df)
    savetocassandra_writesparkdataframe(sparkdf)
    

def savetocassandra_writesparkdataframe(sparkdf):
    sparkdf.write.format("mongo").mode("append").save()   
    


def readfromcassandra(project_id,type=TRAIN_DATA_QUALIFIER):
    from pyspark.sql import SQLContext
    from pyspark import StorageLevel
    spark = getsparksession(project_id,type)
    sqlContext = SQLContext(spark)
    try:
        df = sqlContext.sql("select * from temp_table")

    except:
        traceback.print_exc()
        df = spark.read.format("mongo").load()
        df.registerTempTable("temp_table")
        sqlContext.sql("CACHE TABLE temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")
    return df




def getsparksession(project_id,type):
    #type should be
    # 1 = train
    # 2 = test
    # 3 = cross validation
    if type in [1,2,3]:

        import os
        #relevant for selecting right version
        #TODO: must be configurable
        os.environ['PYSPARK_PYTHON'] = '/usr/local/Frameworks/Python.framework/Versions/3.7/bin/python3.7'
        os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/Frameworks/Python.framework/Versions/3.7/bin/python3.7'

        conf = createSparkConfig(project_id,type)
        
        from django.conf import settings
        db_path = settings.DATABASES['default']['NAME']


        import pyspark
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName('kuai') \
            .master("local[8]")\
            .config(conf=conf)\
            .getOrCreate()
        return spark
    raise TypeError("type")

def createSparkConfig(project_id,type):
    import pyspark
    conf = pyspark.SparkConf()
    conf.set("spark.app.name", 'kuai')
    #conf.set("spark.executor.cores", 1)
    #conf.set("spark.executor.instances", 1)
    #TODO: must be configurable
    conf.set('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.0')
    url = "mongodb://localhost/test.test"
    detailedurl = url+str(project_id)+"_"+str(type)
    conf.set("spark.mongodb.input.uri", detailedurl)
    conf.set("spark.mongodb.output.uri", detailedurl)

    #conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #conf.set("spark.memory.offHeap.size","16g")
    #conf.set("spark.memory.offHeap.enabled",True)
    
    #conf.set("spark.driver.core", "4")
    #conf.set("spark.driver.memory", "2g")
    #conf.set("spark.driver.memoryOverhead","512mb")
    
    #conf.set("spark.executor.core", "4")
    #conf.set("spark.executor.instances", "1")
    conf.set("spark.executor.memory", "6g")
    #conf.set("spark.executor.memoryOverhead","512mb")
    
    #conf.set("spark.default.parallelism","3")
    conf.set("spark.driver.maxResultSize","4096mb")
    conf.set("spark.driver.extraJavaOptions","-XX:+UseG1GC -XX:+UseCompressedOops -XX:-UseGCOverheadLimit")
    
    
    return conf

def applyfeaturetransition(dataframe,features,target):
    indexed = dataframe
    
    for feat in features:
        try:
            if(feat.transition>0):
                indexed = fp.applytransition(feat.transition,feat.fieldname,indexed)
            else:
                #indexed = fp.applytransition(0,feat.fieldname,indexed)
                print("no transformation needed")
            
        except Exception as e:
            print("error on transformation")
            print(e)
    #target
    
    try:
        indexed = fp.applytransition(target.transition,target.fieldname,indexed)
    except Exception as e:
            print("error on transformation")
            print(e)
    return indexed


def buildFeatureVector(dataframe,features,target):
    #Transform Data
    train_df = dataframe
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler
    
    feature_dict = {}
    #TODO: sum up features by dimension

    feature_array = []
    
    for myf in features:
        feature_array.append(myf.fieldname+"Pre")

    vector_assembler = VectorAssembler(inputCols=feature_array, outputCol="features")
    
    train_df3 = vector_assembler.transform(train_df)
    from pyspark.sql import types as T
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import DenseVector
    def convertSparseVectortoDenseVector(v):
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array
    toDenseVectorUdf = F.udf(convertSparseVectortoDenseVector, T.ArrayType(T.FloatType()))
    
    train_df3 = train_df3.withColumn('features_array', toDenseVectorUdf('features'))
    return train_df3


def analysis(request,project_id):
    template = loader.get_template('dataanalysis.html')
    project = get_object_or_404(Project, pk=project_id)
    
    
    df = readfromcassandra(project_id,1)
    df2 = transformdataframe(project,df)

    dfdescription = createDataFrameHTMLPreview(df2.describe())
    
    
    distributionbytarget = createDataFrameHTMLPreview(df2.groupby(project.target).count())
    
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3,
        "dfdescription":dfdescription,
        "distribution":distributionbytarget,
    }


    return HttpResponse(template.render(context, request))


def loadimagesexample():
    import mysite.dataoperation as dt
    ### ImageSchema.toNDArray
    from pyspark.ml.image import ImageSchema
    spark = dt.getsparksession(1,1)
    #zero = ImageSchema.readImages("/Users/sehrbastian/Entwicklung/Bilderkennung/standardisiert2/*") 
    #print(zero.count())
    df = spark.read.format("image").option("dropInvalid", True).load("/Users/sehrbastian/Entwicklung/Bilderkennung/standardisiert2/1/") 
    df = df.limit(1).cache()
    return df
    #test = udf(lambda vector: pyspark.ml.image.ImageSchema.toNDArray(vector)) 
    from django.shortcuts import render,get_object_or_404
    from mysite.models import Project,Feature,NeuralNetwork
    project_id = 1
    project = get_object_or_404(Project, pk=1)
    df2 = dt.transformdataframe(project,df).cache()
    print("transform done")
    from pyspark.sql.functions import substring,col
    removeid = filter(lambda x: x != "_id", df2.columns)
    limitto50 = map(lambda x:substring(col(x).cast("String").alias(x),0,100),removeid) 
    df3 = df2.select(list(limitto50))
    return df3
    #pandas = df3.toPandas()
    #return pandas

    
    #df2.limit(1).show()
    #return df2.take(1)#show(1, truncate=True)