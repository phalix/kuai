
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

import pyspark
from pyspark.sql.functions import udf,col,lit
from pyspark.sql.types import StringType,DoubleType,DateType,ArrayType,FloatType,IntegerType
import inspect



TRAIN_DATA_QUALIFIER = 1
TEST_DATA_QUALIFIER = 2
CV_DATA_QUALIFIER = 3

displaylimit = 10

def createOrUpdateUDF(request,project_id):
    from pyspark.sql.functions import udf

    columns = request.POST['columns']
    udfcode = request.POST['udfcode1']
    outputtype = request.POST['outputtype']
    
    print(columns)
    print(udfcode)
    print(outputtype)
    left = ""
    right = ""
    for col in outputtype.split(','):
        left = left + 'pyspark.sql.types.'+col+'('
        right = ')'+right
    
    outputtypeeval = eval(left+right)
    exec('def a(output): '+udfcode)
    a_udf = eval('udf(a, outputtypeeval)')
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    result = df.select(a_udf(columns)).limit(1).collect()
    

    

    import json
    return HttpResponse(json.dumps({
        'result':result,
        'works':'True'
    }))



def evalUDFFunction(request,project_id):
    func = request.GET['func']
    column = request.GET['column']

    response = {
        'type':type("hello"),
        'value':"hello"
    }



    return HttpResponse(json.dumps(response), content_type='application/json')




def analysisbychart(request,project_id):
    ### TODO: Enable saving of diagrams to revisit analysis
    ### TODO: Enable creation of multiple diagrams

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    from inspect import signature 
    
    
    assert ('x' in request.GET and 'y' in request.GET) or 'hue' in request.GET
    assert 'func' in request.GET
    assert 'columns' in request.GET

    columns = request.GET['columns'].split(",")
    func = request.GET['func']

    assert len(columns)>0
    assert len(func)>0

    a = eval("sns."+func)
    parameters = list(signature(a).parameters)

    kwargs = {}
    if 'x' in request.GET and 'y' in request.GET:
        x = request.GET['x'].split(",")[0]
        if 'x' in parameters:
            kwargs['x'] = x
        y = request.GET['y'].split(",")[0]
        if 'y' in parameters:
            kwargs['y'] = y
    else:
        x = None
        y = None
    
    if 'hue' in request.GET:
        hue = request.GET['hue']
        if len(hue)>0 and 'hue' in parameters:
            kwargs['hue'] = hue
    else:
        hue = None    

    
    

    matplotlib.pyplot.ioff()
    sns.set_theme(style="darkgrid")
    
    project = get_object_or_404(Project, pk=project_id)
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    df2 = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)

    #assert x in df2.columns
    #assert y in df2.columns
    
    response = HttpResponse(content_type="image/jpeg")
    dfxandy = df2.select(columns).toPandas()
    kwargs['data'] = dfxandy
    c = a(**kwargs)


    if hasattr(c,'figure'):
        c.figure.savefig(response, format="png")
    else:
        c.savefig(response, format="png")
    
    return response


def analysis(request,project_id):
    template = loader.get_template('data/dataanalysis.html')
    project = get_object_or_404(Project, pk=project_id)
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    df_test = readfromcassandra(project_id,TEST_DATA_QUALIFIER)
    df_cv = readfromcassandra(project_id,CV_DATA_QUALIFIER)
    print(df.count())
    print(df_test.count())
    print(df_cv.count())
    df2 = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)
    df2_test = transformdataframe(project,df_test,TEST_DATA_QUALIFIER)
    df2_cv = transformdataframe(project,df_cv,CV_DATA_QUALIFIER)
    
    dfdescription = df2.describe().toPandas().to_html()
    dfdescription_test = df2_test.describe().toPandas().to_html()
    dfdescription_cv = df2_cv.describe().toPandas().to_html()

    if project and project.target and project.target.fieldname:
        target = project.target.fieldname
    else:
        target = df2.columns[0]
    
    distributionbytarget = df2.groupby(target).count()
    distributionbytarget_html = distributionbytarget.toPandas().to_html()
    ### restrict columns to simple types
    columns = list(dict(filter(lambda x:isTypeSimpleType(x[1]),df2.dtypes)).keys())
    
    ### TODO: provide plot types of seaborn for selection
    
    seabornfunctions = {}
    seabornfunctions['Relational Plots'] = ['relplot','scatterplot','lineplot']
    seabornfunctions['Distribution Plots'] = ['displot','histplot','kdeplot','ecdfplot','rugplot','distplot']
    seabornfunctions['Categorical Plots'] = ['catplot','stripplot','swarmplot','boxplot','violinplot','boxenplot','pointplot','barplot','countplot']
    seabornfunctions['Regression Plots'] = ['lmplot','regplot','residplot']
    seabornfunctions['Matrix Plots'] = ['heatmap','clustermap']
    seabornfunctions['Multi Plot Grids - Pair'] = ['pairplot']
    seabornfunctions['Multi Plot Grids - Joint '] = ['jointplot']
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3,
        "dfdescription":dfdescription,
        "dfdescription_test":dfdescription_test,
        "dfdescription_cv":dfdescription_cv,
        "distribution":distributionbytarget_html,
        "columns":columns,
        "plotfunctions":seabornfunctions,
    }


    return HttpResponse(template.render(context, request))


def index(request,project_id):
    template = loader.get_template('data/datasetup.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        "projects": Project.objects.all(),
        "menuactive":3
    }


    return HttpResponse(template.render(context, request))



def setuptransformdata(request,project_id):
    template = loader.get_template('data/datatransformation.html')
    project = get_object_or_404(Project, pk=project_id)
    
    #Spark ResultTypes:
    relevanttypes = ['ArrayType','StringType','IntegerType','FloatType']
    #sparktypes = list(map(lambda x: x[0],filter(lambda x: (x[0].endswith("Type") and x[0] in relevanttypes,inspect.getmembers(pyspark.sql.types)))))
    
    sparktypes = dict(map(lambda x: (x[0],len(inspect.signature(x[1]).parameters)>0),filter(lambda x: x[0].endswith("Type") and x[0] in relevanttypes,inspect.getmembers(pyspark.sql.types))))
    

    print(len(inspect.signature(pyspark.sql.types.ArrayType).parameters))
    print(len(inspect.signature(pyspark.sql.types.StringType).parameters))
    print(len(inspect.signature(pyspark.sql.types.FloatType).parameters))
    print(len(inspect.signature(pyspark.sql.types.FloatType).parameters))
    print(len(inspect.signature(pyspark.sql.types.StructType).parameters))
    
    
    print(sparktypes)
    
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    
    df2 = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)
    df = df.drop('type')
    b = createDataFrameHTMLPreview(df)
    
    fields = df.schema

    def getFields(prefix,input):
        result = []
        fields = input.fields
        
        for f in fields:
            result.append(prefix+f.name)
            if hasattr(f.dataType,'fields'):
                result = result + getFields(prefix+f.name+'.',f.dataType)
        return result

    allfields = getFields('',fields)
    


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
        "columns":allfields,
        "types":sparktypes,
        
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
    spark = getsparksession(project.id,TRAIN_DATA_QUALIFIER)
    sqlContext = SQLContext(spark)
    try:
        sqlContext.uncacheTable("transform_temp_table")
        sqlContext.sql("drop table transform_temp_table")
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
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    
    if request.POST['datatype'] == "csv":
        
        df = spark.read.format("csv").option("delimiter",";").option("header","true").load(subm_file)
        
    elif request.POST['datatype'] == "img":
        from pyspark.ml.image import ImageSchema 
        
        df = spark.read.format("image").option("dropInvalid", True).load(subm_file) 
        
    if shuffle:
        df = df.sample(1.0)
    #lendf = df.count()    
    
    from pyspark.sql.functions import col,lit
    traindata,testdata,cvdata = df.randomSplit([trainshare,testshare,cvshare])
    traindata = traindata.withColumn('type',lit(TRAIN_DATA_QUALIFIER))
    testdata = testdata.withColumn('type',lit(TEST_DATA_QUALIFIER))
    cvdata = cvdata.withColumn('type',lit(CV_DATA_QUALIFIER))

    data = traindata.union(testdata).union(cvdata)

    savetocassandra(project_id,data,TRAIN_DATA_QUALIFIER)
    
    from pyspark.sql import SQLContext
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    sqlContext = SQLContext(spark)
    try:
        sqlContext.uncacheTable("temp_table")
        df2 = sqlContext.sql("drop table temp_table")
    except:
        print("nothing to drop")

    return HttpResponseRedirect('/transform/'+str(project_id)+"/")

def dataclassification(request,project_id):
    
    template = loader.get_template('data/dataclassification.html')
    project = get_object_or_404(Project, pk=project_id)
    nn = None
    if project.neuralnetwork:
        nn = project.neuralnetwork
    if not nn:
        nn = NeuralNetwork()
        nn.save()
        project.neuralnetwork = nn

    
    # TODO: Enable Text Preprocessing from Keras
    
    cur_features = project.features.all()
    r_features = list(map(lambda x:x.fieldname,cur_features))
    
    if project.target != None:
        target = project.target.fieldname
    else:
        target = ""
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    df = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)
    
    
    from pyspark.sql import functions as F
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorIndexer
    print("Imports Done")
    
    
    mytypes = gettypesanddimensionsofdataframe(df)

    print("Types sdone")
    
    firstrow_transformed = df.first().asDict()
    try:
        indexed = applyfeaturetransition(df,cur_features,project.target)
        firstrow_indexed = indexed.first().asDict()
        print("First row done")
        featurevector_df = buildFeatureVector(indexed,cur_features,project.target).limit(displaylimit)
        print("featurevector_df")
        featurevector = dict(filter(lambda x:x[0]!='target',featurevector_df.first().asDict().items()))
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





def savetocassandra(project_id,df,type):
    savetocassandra_writesparkdataframe(df)
    

def savetocassandra_writesparkdataframe(sparkdf):
    sparkdf.write.format("mongo").mode("overwrite").save()   
    


def readfromcassandra(project_id,type):
    from pyspark.sql import SQLContext
    
    spark = getsparksession(project_id,1)
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


        
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName('kuai') \
            .master("local[16]")\
            .config(conf=conf)\
            .getOrCreate()
        return spark
    raise TypeError("type")

def createSparkConfig(project_id,type):
    
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

def transformdataframe(project,dataframe,type):
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

def gettypesanddimensionsofdataframe(df):
    #Remove id, since sometimes mongodb adds this
    mytypes_filter = filter(lambda x:x[0]!='_id',df.dtypes)
    mytypes_nodim = list(mytypes_filter)
    
    #mytypes_dict  = dict(mytypes)
    
    #mytypes = list(filter(lambda x:not (x[0]+"Piped" in mytypes_dict.keys()),mytypes ))
    #def reformatPipedPre(x):
    #    if x[0].endswith("Piped"):
    #        a= x[0][:(len(x[0])-5)]
    #    else:
    #        a = x[0]
    #    return (a,x[1])

    mytypes =[]
    #Figure out dimension
    for item in mytypes_nodim:
        if isTypeSimpleType(item[1]):
           cur = item + (1,)
        else:
            cur = item + (0,)
        mytypes.append(cur)

    #mytypes = list(map(lambda x:reformatPipedPre(x),mytypes))

    return mytypes

def isTypeSimpleType(type):
    if type in sparkSimpleTypes():
        return True
    else:
        return False

def sparkSimpleTypes():
    return ('float','string','double','int','short','long','byte','decimal','timestamp','date','boolean','null','data')

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
    import pyspark.sql.functions as psf
    
    feature_dict = getfeaturedimensionbyproject(features)
    selector = []
    if 1 in feature_dict:
        feature_array = list(feature_dict[1].keys())
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
        selector.append(psf.col('features_array'))
    
    
    train_df3 = train_df
    for (key,value) in feature_dict.items():
        if key > 1:
            for (key2,value2) in value.items():
                selector.append(psf.col(key2).alias('feature_'+str(key2)))
                #train_df3 = train_df3.withColumn('feature_'+str(key2),train_df[key2])
                
    selector.append(psf.col(target.fieldname).alias("target"))

    #train_df3 = train_df3.withColumn("target",train_df3[target.fieldname])
    train_df3 = train_df3.select(selector)
    return train_df3


def getXandYFromDataframe(df,project):
    import numpy as np
    featurevalues = list(filter(lambda x:x.startswith("feature"),df.columns))
    x = []
    for feature in featurevalues:
        list_x_3 = df.select(feature).rdd.map(lambda r : r[0]).collect()
        x_3 = np.array(list_x_3)
        x.append(x_3)
    #Remove array structure if input is not multiple!
    if len(x) == 1:
        x = x[0]
    targetvalues = list(filter(lambda x:x.startswith("target"),df.columns))
    y = df.select("target").rdd.map(lambda r : r[0]).collect()
    y = np.array(y)
    return {"x":x,"y":y}


def getTransformedData(project_id,qualifier):
    project = get_object_or_404(Project, pk=project_id)
    #1 = mysite.dataoperation.TRAIN_DATA_QUALIFIER
    train_df = mysite.dataoperation.readfromcassandra(project_id,qualifier)
    train_df = train_df.filter(train_df.type == qualifier)
    train_df = train_df.drop('type')
    train_df = mysite.dataoperation.transformdataframe(project,train_df)
    train_df1 = mysite.dataoperation.buildFeatureVector(train_df,project.features.all(),project.target)
    train_df3 = train_df1
    return train_df3