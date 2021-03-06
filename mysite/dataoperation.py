
from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404
from mysite.models import Project,Feature,NeuralNetwork,UDF
import pandas as pd


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

import mysite.data.dataudftransform


TRAIN_DATA_QUALIFIER = 1
TEST_DATA_QUALIFIER = 2
CV_DATA_QUALIFIER = 3

displaylimit = 10

def customLoadProcedure(request,project_id):
    import json
    value = request.POST['value']
    trainshare = float(request.POST['trainshare'])
    testshare = float(request.POST['testshare'])
    cvshare = float(request.POST['cvshare'])
    #valueIndented = value.replace("\n","\n\t")
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    #valueExecutable = valueIndented +"\n\tdir()"
    #valueExecutableFunction = "def customloadprocudure(spark):\n\t" +valueIndented +"\n\treturn dir()"

    exec(value)
    variablenames = dir()
    variables = []
    for name in variablenames:
        variables.append(eval(name))
    #variables = list(map(lambda x:eval(x),variablenames))
    ## find pandas variable and do the following:
    sparkvariables = list(filter(lambda x : type(x) == pyspark.sql.dataframe.DataFrame,variables))
    if len(sparkvariables)>0:
        df = sparkvariables[len(sparkvariables)-1]
        savetocassandra(project_id,df,TRAIN_DATA_QUALIFIER)
    import pandas
    pandasvariables = list(filter(lambda x : type(x) == pandas.core.frame.DataFrame,variables))
    if len(pandasvariables)>0:
        pdf = pandasvariables[len(pandasvariables)-1]
        df = spark.createDataFrame(pdf)
        generateDataFromUpload(project_id,df,trainshare,testshare,cvshare)
    

    
    return HttpResponse(json.dumps({
        'value':value
    })) 

def createOrUpdateUDF(request,project_id):
    from pyspark.sql.functions import udf
    import json

    columns = request.POST['columns']
    udfcode = request.POST['udfcode1']
    outputtype = request.POST['outputtype']
    udfpk = request.POST['id']
    action = request.POST['action']
    
    clearData(project_id,'transform_temp_table')

    if action == "REMOVE":
        result = ""
        try:
            newudf = get_object_or_404(UDF, pk=udfpk)
            newudf.delete()
            result = "Deletion successfull"
        except:
            result = "Nothing to Delete"
    
        return HttpResponse(json.dumps({
            'result':result,
            'works':'True'
        }))


    project = get_object_or_404(Project, pk=project_id)
    
    if udfpk == '0':
        newudf = UDF(input=columns,udfexecutiontext=udfcode,outputtype=outputtype,project=project)
    else:
        try:
            newudf = get_object_or_404(UDF, pk=udfpk)
            newudf.input = columns
            newudf.udfexecutiontext = udfcode
            newudf.outputtype = outputtype
        except:
            newudf = UDF(input=columns,udfexecutiontext=udfcode,outputtype=outputtype,project=project)

    
    newudf.save()

    errorlog = ""
    result = ""
    try:
        udfpara = {}
        udfpara['outputtype'] = newudf.outputtype
        udfpara['input'] = newudf.input
        udfpara['udfexecutiontext'] = newudf.udfexecutiontext
        udfpara['functionname'] = 'testexecution'
        
        a_udf = generateUDFonUDF(udfpara)
        df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
        result = df.limit(1).select(a_udf[1].alias(str(newudf.pk))).limit(1).collect()
        result = result[0][0]
    except Exception as e:
        import traceback
        import io
        from contextlib import redirect_stdout
        with io.StringIO() as buf, redirect_stdout(buf):
            traceback.print_exc()
            errorlog = str(e)+"\n"+buf.getvalue()
    
    return HttpResponse(json.dumps({
        'result':str(result)[0:5000],
        'works':'True',
        'errorlog':str(errorlog)[0:5000],
        'id':newudf.pk
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

    response = HttpResponse(content_type="image/jpeg")
    dfxandy = df2.select(columns).toPandas()
    kwargs['data'] = dfxandy
    c = a(**kwargs)


    if hasattr(c,'figure'):
        c.figure.savefig(response, format="png")
    else:
        c.savefig(response, format="png")
    
    return response


def correlation(request,project_id):
    template = loader.get_template('data/datacorrelation.html')
    project = get_object_or_404(Project, pk=project_id)
    #df = getTransformedData(project_id,0)
    df = mysite.dataoperation.readfromcassandra(project_id,0)
    df = mysite.dataoperation.transformdataframe(project,df,0)
    df = applyfeaturetransition(project,df,project.features.all(),project.target,project.targets.all())

    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    from pyspark.ml.stat import ChiSquareTest
    feat = list(map(lambda x:x.fieldname,project.features.all()))
    targ = list(map(lambda x:x.fieldname,project.targets.all()))

    features = feat+targ
    assembler = VectorAssembler(
        inputCols=features,
        outputCol="correlation")
    
    output = assembler.transform(df)
    corr_mat=Correlation.corr(output,"correlation", method="pearson")
    corr_html = corr_mat.toPandas().iloc[0]['pearson(correlation)']
    #chi_sqr = ChiSquareTest.test(output, "correlation", "label").head()
    context = {
        "project" : project,
        "project_id" : project_id,
        "menuactive":3,
        "correlation":corr_html
        
    }
    return HttpResponse(template.render(context, request))
    

def analysis(request,project_id):
    template = loader.get_template('data/dataanalysis.html')
    project = get_object_or_404(Project, pk=project_id)
    df2 = mysite.dataoperation.readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    df2 = mysite.dataoperation.transformdataframe(project,df2,TRAIN_DATA_QUALIFIER)
    df2 = df2.drop('type')
    dfhead5 = df2.limit(5).toPandas().to_html()
    
    dfdescription = df2.describe().toPandas().to_html()
    columns = list(dict(filter(lambda x:isTypeSimpleType(x[1]),df2.dtypes)).keys())
    
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
        #"dfdescription_test":dfdescription_test,
        #"dfdescription_cv":dfdescription_cv,
        "columns":columns,
        "plotfunctions":seabornfunctions,
        "df2head":dfhead5
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
    relevanttypes = ['ArrayType','StringType','IntegerType','FloatType','DoubleType','DateType']
    #sparktypes = list(map(lambda x: x[0],filter(lambda x: (x[0].endswith("Type") and x[0] in relevanttypes,inspect.getmembers(pyspark.sql.types)))))
    
    sparktypes = dict(map(lambda x: (x[0],len(inspect.signature(x[1]).parameters)>0),filter(lambda x: x[0].endswith("Type") and x[0] in relevanttypes,inspect.getmembers(pyspark.sql.types))))
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    
    df2 = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)
    df = df.drop('type')
    df2 = df2.drop('type')

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
    
    ##GET Already Defined udfs
    from django.core.serializers import serialize
    alludfsperproject = UDF.objects.filter(project=project_id).all()
    udfsPerJson = serialize('json', alludfsperproject)

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
        "udfs": udfsPerJson,
        
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

    clearData(project_id,'transform_temp_table')
    
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
    import uuid 
    tempfile = "temp/upload/temp"+str(uuid.uuid1())

    if request.POST['datatype'] == "csv":
        if 'filewithdata' in request.FILES:
            content = request.FILES['filewithdata'].file.read()
            open(tempfile, 'wb').write(content)
        elif not subm_file.startswith("http"):
            tempfile = subm_file
        else:
            import requests
            url = subm_file
            r = requests.get(url, allow_redirects=True)
            open(tempfile, 'wb').write(r.content)
        
        import mysite.project as mp
        delimeter = mp.getSetting(project_id,'delimeter')[0]
        
        df = spark.read.format("csv").option("delimiter",delimeter).option("header","true").option("inferSchema", "true").load(tempfile)
        
    elif request.POST['datatype'] == "img":
        if 'filewithdata' in request.FILES:
            content = request.FILES['filewithdata'].file.read()
            open(tempfile, 'wb').write(content)
            from zipfile import ZipFile
            with ZipFile(tempfile, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                subm_file = "temp/extract/temp"+str(uuid.uuid1())
                zipObj.extractall(subm_file)

        from pyspark.ml.image import ImageSchema 
        df = spark.read.format("image").option("dropInvalid", True).option("recursiveFileLookup","true").load(subm_file) 
    elif request.POST['datatype'] == "json":
        df  = spark.read.json(subm_file, multiLine = "true")
    
    if shuffle:
        df = df.sample(1.0)
    #lendf = df.count()    
    
    generateDataFromUpload(project_id,df,trainshare,testshare,cvshare)

    return HttpResponseRedirect('/transform/'+str(project_id)+"/")

def generateDataFromUpload(project_id,df,trainshare=0.6,testshare=0.2,cvshare=0.2):
    from pyspark.sql.functions import col,lit
    traindata,testdata,cvdata = df.randomSplit([trainshare,testshare,cvshare])
    traindata = traindata.withColumn('type',lit(TRAIN_DATA_QUALIFIER))
    testdata = testdata.withColumn('type',lit(TEST_DATA_QUALIFIER))
    cvdata = cvdata.withColumn('type',lit(CV_DATA_QUALIFIER))

    data = traindata.union(testdata).union(cvdata)
    data.persist(pyspark.StorageLevel.DISK_ONLY)
    
    
    clearData(project_id,'temp_table')
    
    #alias column names for forbidden characters

    from functools import reduce

    oldColumns = data.schema.names
    import functools

    substituedColumns = list(filter(lambda x: len(x)>0 or x.startswith('feature') or x.startswith('target'),oldColumns))
    
    newColumns = list(map(lambda x:col(x).alias(x
        .replace(" ","")
        .replace(",","")
        .replace(";","")
        .replace("{","")
        .replace("}","")
        .replace("(","")
        .replace(")","")
        .replace("\n","")
        .replace("\t","")),
        substituedColumns))
    
    data = data.select(newColumns)

    savetocassandra(project_id,data,TRAIN_DATA_QUALIFIER)

def clearData(project_id,stage):
    from pyspark.sql import SQLContext
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    sqlContext = SQLContext(spark)
    tables = ['temp_table','transform_temp_table','indexed_temp_table']
    startwith = tables.index(stage)
    deleteThis = tables[startwith:]
    for table in deleteThis:
        if table in sqlContext.tableNames():
            sqlContext.uncacheTable(table)
            sqlContext.sql("drop table "+table)

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
    cur_targets = project.targets.all()
    r_features = list(map(lambda x:x.fieldname,cur_features))
    r_targets = list(map(lambda x:x.fieldname,cur_targets))
    if project.target != None:
        target = project.target.fieldname
    else:
        target = ""
    
    df = readfromcassandra(project_id,TRAIN_DATA_QUALIFIER)
    df = transformdataframe(project,df,TRAIN_DATA_QUALIFIER)
    firstrow_transformed = df.first().asDict()
    try:
        
        indexed = applyfeaturetransition(project,df,cur_features,project.target,cur_targets)
        distributionbytarget = df.groupby(project.target.fieldname).count()
        distributionbytarget_html = distributionbytarget.limit(100).toPandas().to_html()
    except Exception as e:
        
        traceback.print_exc()
        print(e)

        print("feature transition not successfull")
        distributionbytarget_html = ""
        indexed = df
    
    from pyspark.sql import functions as F
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorIndexer
    print("Imports Done")
    
    
    mytypes = gettypesanddimensionsofdataframe(indexed)

    print("Types sdone")
    
    
    try:
        
        firstrow_indexed = indexed.first().asDict()
        print("First row done")
        featurevector_df = buildFeatureVector(indexed,cur_features,project.target,cur_targets).limit(displaylimit)
        
        df = df.drop('type')
        indexed = indexed.drop('type')
        featurevector_df = featurevector_df.drop('type')
        
        print("featurevector_df")
        featurevector = dict(filter(lambda x:x[0]!='target',featurevector_df.first().asDict().items()))
        print("featurevector")
        featurevector_df_html = ""
        featurevector_df_html = createDataFrameHTMLPreview(featurevector_df)
        print("featurevector_df_html")
        dataframe_html = ""
        dataframe_html = createDataFrameHTMLPreview(indexed) 
        print("dataframe_html preview done")
        for key in firstrow_indexed:
            firstrow_indexed[key] = str(firstrow_indexed[key])[0:50]

        for key in firstrow_transformed:
            firstrow_transformed[key] = str(firstrow_transformed[key])[0:50]
            
    
    except Exception as e:
        print(e)
        traceback.print_exc()
        firstrow_indexed = []
        featurevector = []
        featurevector_df_html = ""
        dataframe_html = ""
    


    context = {
        "project" : project,
        "project_id" : project_id,
        "featurenames": r_features,
        "features": cur_features,
        "targets" : cur_targets,
        "targetnames" : r_targets,
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
        "distribution":distributionbytarget_html,
        "transformators":mysite.featurepipelines.getAllSparkFeatureModifiers()
    }
    
    
    return HttpResponse(template.render(context, request))



def setupdataclassifcation(request,project_id):
    project = get_object_or_404(Project, pk=project_id)
    project.features.all().delete()
    
    try:
        project.target.delete()
    except Exception as e:
        print(e)
    
    try:
        project.targets.all().delete()
    except Exception as e:
        print(e)
        for target in project.targets.all():
            project.targets.remove(target.pk)
    
    project.input = ""
    project.save()
    #project.target.delete()
    
    #targetselection = request.POST['targetselection']
    targetselection = request.POST.getlist('targetselection')
    if type(targetselection) != list:
        targetselection = [targetselection]
    for targetname in targetselection:
        targettransition = request.POST['fttransition_'+targetname]
        reformattransition = request.POST['ftreformat_'+targetname]
        targetdimension = request.POST['dimension_'+targetname]
        targettype = request.POST['fttype_'+targetname]
        target = Feature(fieldname=targetname,transition=targettransition,reformat=reformattransition,type=targettype,dimension=targetdimension)
        target.save()
        project.targets.add(target)
        project.target = target
    
    for x in request.POST:
        if x.startswith('feature_') and x[8:] not in targetselection:
            print(x[8:])
            transition = request.POST['fttransition_'+x[8:]]
            reformat = request.POST['ftreformat_'+x[8:]]
            dimension = request.POST['dimension_'+x[8:]]
            fttype = request.POST['fttype_'+x[8:]]
            curfeat = Feature(fieldname=x[8:],transition=transition,reformat=reformat,type=fttype,dimension=dimension)
            curfeat.save()
            project.features.add(curfeat)
            curfeat.save()
        if x.startswith('fttransition_') and x[8:] not in targetselection:
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
    
    from pyspark.sql import SQLContext
    
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    sqlContext = SQLContext(spark)
    if 'indexed_temp_table' in sqlContext.tableNames():
        sqlContext.uncacheTable("indexed_temp_table")
        sqlContext.sql("drop table indexed_temp_table")
    
    
    project.save()
    return HttpResponseRedirect('/dataclassification/'+str(project_id))



#########################################
### non-django but Spark related things##
### TODO: move to extra class!!!       ##
#########################################





def savetocassandra(project_id,df,type):
    savetocassandra_writesparkdataframe(project_id,df)
    

def savetocassandra_writesparkdataframe(project_id,sparkdf):
    sparkdf.write.format("mongo").mode("overwrite").save()  
    #sparkdf.write.format("parquet").mode("overwrite").save(str(project_id)+".parquet")
    from pyspark.sql import SQLContext
    spark = getsparksession(project_id,TRAIN_DATA_QUALIFIER)
    sqlContext = SQLContext(spark)
    if 'temp_table' in sqlContext.tableNames():
        sqlContext.uncacheTable("temp_table")
        sqlContext.sql("drop table temp_table")
    


def readfromcassandra(project_id,type):
    from pyspark.sql import SQLContext
    
    spark = getsparksession(project_id,type)
    sqlContext = SQLContext(spark)
    if 'temp_table' in sqlContext.tableNames():
        df = sqlContext.sql("select * from temp_table")
    else:
        df = spark.read.format("mongo").load()
        #df = spark.read.parquet(str(project_id)+".parquet")
        df.registerTempTable("temp_table")
        sqlContext.sql("CACHE TABLE temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")
    return df



def getsparkcontext(project_id,type):
    import os
    conf = createSparkConfig(project_id,type)
    from django.conf import settings
    from pyspark.sql import SparkSession
    
    appName = 'kuai_'+str(project_id)

    from pyspark import SparkContext
    sc = SparkContext.getOrCreate(conf)
    if sc.appName != appName:
        try:
            sc.stop()
        except:
            print("Could not delete Context")
        sc = SparkContext.getOrCreate(conf)
    return sc

def getsparksession(project_id,type):
    #type should be
    # 1 = train
    # 2 = test
    # 3 = cross validation
    import os
    conf = createSparkConfig(project_id,type)
    from django.conf import settings
    from pyspark.sql import SparkSession
    
    appName = 'kuai_'+str(project_id)

    sc = getsparkcontext(project_id,type)
    
    
    spark = SparkSession(sc).builder.appName(appName) \
        .master("local[*]")\
        .config(conf=conf)\
        .getOrCreate()
    if spark.builder._options['spark.app.name'] != appName:
        spark = SparkSession(sc).builder\
            .config(conf=conf)\
            .getOrCreate()
    return spark

def createSparkConfig(project_id,type):
    
    appName = 'kuai_'+str(project_id)
    conf = pyspark.SparkConf().setMaster("local[*]").setAppName(appName)
    
    conf.set("spark.app.name", appName)
    conf.set("spark.driver.allowMultipleContexts", "true")
    #### MONGO
    conf.set('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.0')
    import mysite.project as mp
    url = mp.getSetting(project_id,'mongoconnection')[0]
    
    detailedurl = url+"/test.test"+str(project_id)+"_"+"1"#str(type)
    conf.set("spark.mongodb.input.uri", detailedurl)
    conf.set("spark.mongodb.output.uri", detailedurl)
    
    
    #conf.set("spark.driver.memory", "8g")
    #conf.set("spark.executor.memory", "8g")

    #conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    #conf.set("spark.sql.inMemoryColumnarStorage.batchSize", 5)
    #conf.set("spark.sql.inMemoryColumnarStorage.compressed",False)
    return conf



def createDataFrameHTMLPreview(dataframe):
    from pyspark.sql.functions import substring
    #remove id
    #cast to string, limit to 50
    removeid = filter(lambda x: x != "_id", dataframe.columns)
    limitto50 = list(map(lambda x:(substring(col(x).cast("String"),0,100)).alias(x),removeid))
    pandas = dataframe.limit(displaylimit).select(limitto50).toPandas()
    html = pandas.to_html()
    

    return html


def applyfeaturetransition(project,dataframe,features,target,targets):

    from pyspark.sql import SQLContext
    from pyspark import StorageLevel

    spark = getsparksession(project.id,1)
    sqlContext = SQLContext(spark)
    if 'indexed_temp_table' in sqlContext.tableNames():
        indexed = sqlContext.sql("select * from indexed_temp_table")
    else:
        traceback.print_exc()
        
        try:
            indexed = dataframe
            assert features != None
            assert target != None

            for feat in features:
                try:
                    if(len(feat.reformat.strip())>0):
                        indexed = fp.applySetOfFeatureModifiers(feat.reformat.strip(),feat.fieldname,indexed)
                    if(feat.transition>0):
                        indexed = fp.applytransition(feat.transition,feat.fieldname,indexed)
                    else:
                        #indexed = fp.applytransition(0,feat.fieldname,isndexed)
                        print("no transformation needed for "+str(feat.fieldname))
                    
                except Exception as e:
                    print("error on transformation")
                    print(e)
                    traceback.print_exc()
            #target
            
            for feat in targets:
                try:
                    if(len(feat.reformat.strip())>0):
                        indexed = fp.applySetOfFeatureModifiers(feat.reformat.strip(),feat.fieldname,indexed)
                    if(feat.transition>0):
                        indexed = fp.applytransition(feat.transition,feat.fieldname,indexed)
                    else:
                        #indexed = fp.applytransition(0,feat.fieldname,isndexed)
                        print("no transformation needed for "+str(feat.fieldname))
                    
                except Exception as e:
                    print("error on transformation")
                    print(e)
                    traceback.print_exc()
            #target
            '''
            try:
                if(len(target.reformat.strip())>0):
                    indexed = fp.applySetOfFeatureModifiers(target.reformat.strip(),target.fieldname,indexed)
                if(target.transition>0):
                    indexed = fp.applytransition(target.transition,target.fieldname,indexed)
            except Exception as e:
                print("error on transformation")
                print(e)
                traceback.print_exc()
            '''
            

            
            indexed.registerTempTable("indexed_temp_table")
            #sqlContext.cacheTable("transform_temp_table",StorageLevel.DISK_ONLY)
            sqlContext.sql("CACHE TABLE indexed_temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")

        except Exception as e:
            print(e)
            traceback.print_exc()
            #project.selectstatement = ""
            #project.save()
            indexed = dataframe
        
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
        inonedim = reduce(lambda x,y:[x[0]+y[0]],list(onedimension.values()))

        result = {
            1:[inonedim[0],],
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


def getoutputschema(project_id):
    
    import json
    from functools import reduce
    project = get_object_or_404(Project, pk=project_id)
    
    cur_features = project.targets.all()
    feature_dict = getfeaturedimensionbyproject(cur_features)
    
    
    result = {}
    result2 = {}
    for key in feature_dict.keys():
        result[key] = []
    
    for (key,value) in feature_dict.items():
        for (key2,value2) in value.items():
            result[key].append(value2)
            result2[key2] =  value2 
    
    return result2


### return dataframe by features and target

def gettypesanddimensionsofdataframe(df):
    #Remove id, since sometimes mongodb adds this
    mytypes_filter = filter(lambda x:x[0] not in ('_id','type'),df.dtypes)
    mytypes_nodim = list(mytypes_filter)
    
    totestwith = df.limit(1).collect()[0].asDict()
    
    iterabletypes = [list,pyspark.ml.linalg.SparseVector,pyspark.ml.linalg.DenseVector]
    mytypes =[]
    #Figure out dimension
    for item in mytypes_nodim:
        if isTypeSimpleType(item[1]):
           cur = item + ("1",)
        elif item[1] == 'struct<origin:string,height:int,width:int,nChannels:int,mode:int,data:binary>':
            ## This is an image
            tovalidate = totestwith[item[0]]
            dimension = str([tovalidate.height,tovalidate.width,tovalidate.nChannels]).replace("[","").replace("]","").replace(" ","")
            cur = item + (dimension,)
        else:
            tovalidate = totestwith[item[0]]
            tovalidatetype = type(tovalidate)
            if(tovalidatetype in iterabletypes):
                mytuple = []
                checkfield = tovalidate
                while type(checkfield) in iterabletypes:
                    mytuple.append(len(checkfield))
                    checkfield = checkfield[0]
                mytuple = str(mytuple).replace("[","").replace("]","").replace(" ","")
                cur = item + (mytuple,)

            else:
                print(str(tovalidatetype)+" is not iterable cannot infer")
                cur = item + ("0",)
            
            
        mytypes.append(cur)

    #mytypes = list(map(lambda x:reformatPipedPre(x),mytypes))

    return mytypes

def isTypeSimpleType(type):
    if type in sparkSimpleTypes():
        return True
    else:
        return False

def sparkSimpleTypes():
    return ('float','string','double','int','short','long','byte','decimal','timestamp','date','boolean','null','data','bigint')


def buildFeatureVector(dataframe,features,target,targets):
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
    
    def convertSparseVectortoDenseVector(v):
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array
    from pyspark.sql import types as T
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import DenseVector
    
    toDenseVectorUdf = F.udf(convertSparseVectortoDenseVector, T.ArrayType(T.FloatType()))

    if 1 in feature_dict:
        feature_array = list(feature_dict[1].keys())
        vector_assembler = VectorAssembler(inputCols=feature_array, outputCol="features")
        train_df3 = vector_assembler.transform(train_df)
        
        train_df3 = train_df3.withColumn('features_array', toDenseVectorUdf('features'))
        train_df = train_df3
        selector.append(psf.col('features_array'))
    
    train_df3 = train_df
    for (key,value) in feature_dict.items():
        if key > 1:
            for (key2,value2) in value.items():
                colType = dict(train_df3.dtypes)[key2]
                if colType == 'vector':
                    selector.append(toDenseVectorUdf(key2).alias('target_'+str(key2)))
                else: 
                    selector.append(psf.col(key2).alias('feature_'+str(key2)))
                    
    target_dict = getfeaturedimensionbyproject(targets)
    
    for (key,value) in target_dict.items():
            for (key2,value2) in value.items():
                colType = dict(train_df3.dtypes)[key2]
                if colType == 'vector':
                    selector.append(toDenseVectorUdf(key2).alias('target_'+str(key2)))
                else: 
                    selector.append(psf.col(key2).alias('target_'+str(key2)))
    selector.append(col('type'))
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



def getDimensionsByProject(project):
    
    train_df = getTransformedData(project.pk,0)
    mytypes = gettypesanddimensionsofdataframe(train_df)
    inputdimensions = list(filter(lambda x:x[0].startswith('feature'),mytypes))#list(filter(lambda x:x[0] in featurenames,mytypes))
    outputdimensions = list(filter(lambda x:x[0].startswith('target'),mytypes))#list(filter(lambda x:x[0] in [project.target.fieldname],mytypes))
    return [inputdimensions,outputdimensions]


def getTransformedData(project_id,qualifier):
    project = get_object_or_404(Project, pk=project_id)
    #1 = mysite.dataoperation.TRAIN_DATA_QUALIFIER
    train_df = mysite.dataoperation.readfromcassandra(project_id,qualifier)
    train_df = mysite.dataoperation.transformdataframe(project,train_df,qualifier)
    train_df = applyfeaturetransition(project,train_df,project.features.all(),project.target,project.targets.all())
    train_df1 = mysite.dataoperation.buildFeatureVector(train_df,project.features.all(),project.target,project.targets.all())
    if qualifier > 0:
        train_df1 = train_df1.filter(train_df.type == qualifier)
    train_df1 = train_df1.drop('type')
    train_df3 = train_df1
    #duplicaterows(train_df3)
    return train_df3

def duplicaterows(dataframe):
    test = dataframe.groupby('target').count().collect()
    from functools import reduce
    maxvalue = reduce(lambda x,y: x if x['count'] > y['count'] else y,test)['count']
    from pyspark.sql.functions import explode
    for target in test:
        s = maxvalue/target['count']
        from math import floor
        from pyspark.sql.functions import array, lit
        if s >= 2:
            s = floor(s)
            toattach = dataframe.filter(dataframe.target==array(*[lit(x) for x in target['target']]))
            for i in range(0,s):
                
                dataframe = dataframe.union(toattach)
    test = dataframe.groupby('target').count().collect()
    return dataframe



##########################
##### DATATRANSFORM ######
##########################

def evaluateUDFOnDataFrame(project,dataframe):
    udfs = collectUDFSOnProject(project)
    udfs['type'] = col("type") # do not loose type, train test cv
    udflist = list(udfs.values())
    udflist = udflist + dataframe.columns
    return dataframe.select(udflist)
    
def collectUDFSOnProject(project):
    udfs = UDF.objects.filter(project=project.pk).all()
    from django.core import serializers
    data = serializers.serialize("json", udfs)
    import json
    udfdata = json.loads(data)

    functionarray = {}
    for udf in udfdata:
        import re
        pattern = re.compile("[^A-Za-z]")
        aliasname = pattern.sub("",udf['fields']['input'])
        functionname = aliasname+'_'+str(udf['pk'])
        udfpara = udf['fields']
        udfpara['functionname'] = functionname
        a = generateUDFonUDF(udfpara)
        functionarray[a[0]] = a[1]
    
    return functionarray



def generateUDFonUDF(udfdefinition):
    assert 'outputtype' in udfdefinition
    assert 'udfexecutiontext' in udfdefinition
    assert 'functionname' in udfdefinition
    assert 'input' in udfdefinition

    from pyspark.sql.functions import udf
    outputtype = udfdefinition['outputtype']
    left = ""
    right = ""
    for col in outputtype.split(','):
        left = left + 'pyspark.sql.types.'+col+'('
        right = ')'+right
    outputtypeeval = eval(left+right)
    
    udfcode = udfdefinition['udfexecutiontext']
    functionname = udfdefinition['functionname']
    exec('def '+functionname+'(output): \n\t'+udfcode.replace("\n","\n\t"))
    a_udf = eval('udf('+functionname+', outputtypeeval)')
    udfColInput = udfdefinition['input']
    if len(udfColInput.split(","))>1:
        udfColInput = eval("pyspark.sql.functions.array"+str(tuple(udfColInput.split(","))))

    b_udf = a_udf(udfColInput).alias(functionname)
    return [functionname,b_udf]

def transformdataframe(project,dataframe,type):
    from pyspark.sql import SQLContext
    from pyspark import StorageLevel

    spark = getsparksession(project.id,1)
    sqlContext = SQLContext(spark)
    if 'transform_temp_table' in sqlContext.tableNames():
        df2 = sqlContext.sql("select * from transform_temp_table")
    else:
        
        try:
            df2 = evaluateUDFOnDataFrame(project,dataframe)
            df2.registerTempTable("transform_temp_table")
            sqlContext.sql("CACHE TABLE transform_temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")

        except Exception as e:
            print(e)
            traceback.print_exc()
            df2 = dataframe
        
    return df2


def applyUdfTransformation(project):
        udfs = UDF.objects.filter(project=project.pk).all()
        from django.core import serializers
        data = serializers.serialize("json", udfs)
        import json
        udfdata = json.loads(data)

        functionarray = {}
        for udf in udfdata:
            import re
            pattern = re.compile("[^A-Za-z]")
            aliasname = pattern.sub("",udf['fields']['input'])
            functionname = aliasname+'_'+str(udf['pk'])
            udfpara = udf['fields']
            udfpara['functionname'] = functionname
            a = generateUDFonUDF(udfpara)
            functionarray[a[0]] = a[1]
        
        return functionarray