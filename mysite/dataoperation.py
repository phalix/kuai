
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


def analysis(request,project_id):
    template = loader.get_template('dataanalysis.html')
    project = get_object_or_404(Project, pk=project_id)
    
    df = readfromcassandra(project_id,1)
    df2 = transformdataframe(project,df)

    dfdescription = df2.describe().toPandas().to_html()
    target = project.target.fieldname
    distributionbytarget = df2.groupby(target).count()
    distributionbytarget_html = distributionbytarget.toPandas().to_html()
    
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3,
        "dfdescription":dfdescription,
        "distribution":distributionbytarget_html,
    }


    return HttpResponse(template.render(context, request))


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
    print(selectstatement)
    print(udfclasses)
    
    if len(selectstatement)==0:
        selectstatement = 'select("*")'

    project = get_object_or_404(Project, pk=project_id)
    project.selectstatement = selectstatement
    project.udfclasses = udfclasses
    project.save()

    
    
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
        df = spark.read.format("image").option("dropInvalid", True).load(subm_file) 
        #df = ImageSchema.readImages(subm_file) 
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

    return HttpResponseRedirect('/transform/'+str(project_id)+"/")

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
    
    
    from pyspark.sql import functions as F
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorIndexer
    print("Imports Done")
    indexed = applyfeaturetransition(df,cur_features,project.target)
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

    print("Types sdone")
    
    firstrow_transformed = df.first().asDict()
    try:
        firstrow_indexed = indexed.first().asDict()
        print("First row done")
        from pyspark.sql.functions import size
        indexed.select(size(col("image"))).show()
        featurevector_df = buildFeatureVector(indexed,cur_features,project.target).limit(displaylimit)
        print("featurevector_df")
        featurevector = featurevector_df.first().asDict()['features_array']
        print("featurevector")
        featurevector_df_html = ""
        featurevector_df_html = createDataFrameHTMLPreview(featurevector_df)
        print("featurevector_df_html")
        dataframe_html = ""
        dataframe_html = createDataFrameHTMLPreview(indexed) 
        print("dataframe_html preview done")
        
            
    
    except Exception as e:
        print(e)
        traceback.print_exc()
        firstrow_indexed = firstrow_transformed
        featurevector = []
        featurevector_df_html = ""
        dataframe_html = ""
    
    for key in firstrow_indexed:
        firstrow_indexed[key] = str(firstrow_indexed[key])[0:50]

    for key in firstrow_transformed:
        firstrow_transformed[key] = str(firstrow_transformed[key])[0:50]

    print(firstrow_transformed)
    print(firstrow_indexed)
    
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
        "firstrow_indexed": firstrow_indexed,
        "firstrow_transformed": firstrow_transformed,
        "pipelines": fp.indexPipelines(),
        "featurevector": featurevector,
        "dataframe":dataframe_html,#
        "dataframe_fv_tar": featurevector_df_html,
    }
    
    
    return HttpResponse(template.render(context, request))



def setupdataclassifcation(request,project_id):
    project = get_object_or_404(Project, pk=project_id)
    project.features.all().delete()
    project.input = ""
    project.save()
    #project.target.delete()
    print(request.POST)
    targetselection = request.POST['targetselection']
    targettransition = request.POST['fttransition_'+targetselection]
    reformattransition = request.POST['ftreformat_'+targetselection]
    targetdimension = request.POST['dimension_'+targetselection]
    targettype = request.POST['fttype_'+targetselection]
    target = Feature(fieldname=targetselection,transition=targettransition,reformat=reformattransition,type=targettype,dimension=targetdimension)
    target.save()
    project.target = target
    
    for x in request.POST:
        if x.startswith('feature_') and x[8:]!=targetselection:
            print(x[8:])
            transition = request.POST['fttransition_'+x[8:]]
            reformat = request.POST['ftreformat_'+x[8:]]
            dimension = request.POST['dimension_'+x[8:]]
            type = request.POST['fttype_'+x[8:]]
            curfeat = Feature(fieldname=x[8:],transition=transition,reformat=reformat,type=type,dimension=dimension)
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
    sparkdf.write.format("mongo").mode("overwrite").save()   
    


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
        #os.environ['PYSPARK_PYTHON'] = '/usr/local/Frameworks/Python.framework/Versions/3.7/bin/python3.7'
        #os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/Frameworks/Python.framework/Versions/3.7/bin/python3.7'

        #os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\sebas\\AppData\\Local\\Programs\\Python\\Python38'
        #os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\\Users\\sebas\\AppData\\Local\\Programs\\Python\\Python38'

        conf = createSparkConfig(project_id,type)
        
        from django.conf import settings
        db_path = settings.DATABASES['default']['NAME']


        import pyspark
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName('kuai') \
            .master("local[16]")\
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
    conf.set("spark.driver.memory", "16g")
    #conf.set("spark.driver.memoryOverhead","512mb")
    
    #conf.set("spark.executor.core", "4")
    #conf.set("spark.executor.instances", "1")
    conf.set("spark.executor.memory", "16g")
    #conf.set("spark.executor.memoryOverhead","512mb")
    
    #conf.set("spark.default.parallelism","3")
    #conf.set("spark.driver.maxResultSize","4096mb")
    #conf.set("spark.driver.extraJavaOptions","-XX:+UseG1GC -XX:+UseCompressedOops")
    
    
    return conf

def transformdataframe(project,dataframe):
    #apply project.selectstatement
    #apply project.udfclasses
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
    limitto50 = map(lambda x:(substring(col(x).cast("String"),0,100)).alias(x),removeid) 
    pandas = dataframe.limit(displaylimit).select(list(limitto50)).toPandas()
    html = pandas.to_html()
    

    return html


def applyfeaturetransition(dataframe,features,target):
    indexed = dataframe
    assert features != None
    assert target != None

    for feat in features:
        try:
            if(feat.transition>0):
                indexed = fp.applytransition(feat.transition,feat.fieldname,indexed)
            else:
                #indexed = fp.applytransition(0,feat.fieldname,isndexed)
                print("no transformation needed")
            
        except Exception as e:
            print("error on transformation")
            print(e)
    #target
    
    try:
        if(target.transition>0):
            indexed = fp.applytransition(target.transition,target.fieldname,indexed)
    except Exception as e:
            print("error on transformation")
            print(e)
    return indexed




### Get Features and Dimensions

def getfeaturedimensionbyproject(features):
    feature_dict = {}
    
    for myf in features:
        a = myf.dimension
        b = list(map(lambda x:int(x),a.split(",")))
        if len(b) in feature_dict:
            arr = feature_dict[len(b)]    
        else:
            arr = {}
        arr[myf.fieldname] = b
        feature_dict[len(b)] = arr

    return feature_dict

### concats number of dimensions

def getinputschema(project_id):
    import pyspark
    import json
    from functools import reduce
    project = get_object_or_404(Project, pk=project_id)
    
    cur_features = project.features.all()
    feature_dict = getfeaturedimensionbyproject(cur_features)
    
    #Seperate Treatment of one-dimensional features, due to the fact that they can be joined
    if 1 in feature_dict:
        onedimension = feature_dict[1] 
        inonedim = reduce(lambda x,y:x+y,onedimension.values())

        result = {
            1:[len(inonedim),],
        }
    else:
        result = {}

    for key in feature_dict.keys():
        if key>1:
            result[key] = []
    
    for (key,value) in feature_dict.items():
        if key>1:
            for (key2,value2) in value.items():
                result[key].append(value2)
    res_json_str = json.dumps(result)
    project.input = res_json_str
    project.save()
    
    #get by without calling the function
    #result = json.loads(project.input)
    return result

### return dataframe by features and target

def buildFeatureVector(dataframe,features,target):
    ###
    assert target != None
    assert features != None
    assert features[0] != None
    assert target.fieldname != None
    assert features[0].fieldname != None
    
    
    #Transform Data
    train_df = dataframe
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler
    
    
    feature_dict = getfeaturedimensionbyproject(features)
    if 1 in feature_dict:
        feature_array = list(feature_dict[1].keys())
        print("vector assembler")
        print(feature_array)
        vector_assembler = VectorAssembler(inputCols=feature_array, outputCol="features")
        train_df.show()
        train_df3 = vector_assembler.transform(train_df)
        train_df3.show()
        from pyspark.sql import types as T
        from pyspark.sql import functions as F
        from pyspark.ml.linalg import DenseVector
        def convertSparseVectortoDenseVector(v):
            v = DenseVector(v)
            new_array = list([float(x) for x in v])
            return new_array
        toDenseVectorUdf = F.udf(convertSparseVectortoDenseVector, T.ArrayType(T.FloatType()))
        train_df3 = train_df3.withColumn('features_array', toDenseVectorUdf('features'))
        train_df = train_df3
    print("*************")
    print(feature_dict)
    print("*************")
    train_df3 = train_df
    for (key,value) in feature_dict.items():
        if key > 1:
            for (key2,value2) in value.items():
                train_df3 = train_df3.withColumn('features_array'+str(value2),train_df[key2])
                print(key2)
                print(value2)
    
    train_df3 = train_df3.withColumn("target",train_df3[target.fieldname])

    return train_df3