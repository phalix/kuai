from django.test import TestCase
from django.test import Client

from mysite import dataoperation
from django.shortcuts import render, get_object_or_404
from mysite.models import Project, Feature, NeuralNetwork


class TestFeaturePipelines(TestCase):
    def setUp(self):
        print("setup")

    def testfeaturepipelineA(self):
        import mysite.featurepipelines as mf
        import mysite.dataoperation as md

        print(mf.getAllSparkFeatureModifiers())

        import pandas as pd
        inputx = [0,0,1,1]
        inputy = [0,1,0,1]
        outputand = [0,0,0,1]
        outputxor = [0,1,1,0]
        outputor = [0,1,1,1]
        typea = [0,0,0,0]

        pdf = pd.DataFrame(list(zip(inputx, inputy,outputand,outputxor,outputor,typea)), 
               columns =['inputx', 'inputy','outputand','outputxor','outputor','type']) 

        spark = md.getsparksession(1,1)
        df = spark.createDataFrame(pdf)


        

        result = mf.applySetOfFeatureModifiers("VectorAssembler,StandardScaler","inputy",df).collect()

        print("success")
        