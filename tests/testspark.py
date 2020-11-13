from django.test import TestCase

from mysite import dataoperation
from django.shortcuts import render,get_object_or_404
from mysite.models import Project,Feature,NeuralNetwork


class AnimalTestCase(TestCase):
    def setUp(self):
        print("setup")

    def test_data_classi(self):
        project_id = 1
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
        from pyspark.ml.feature import OneHotEncoderEstimator
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
