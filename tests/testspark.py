from django.test import TestCase
from django.test import Client

from mysite import dataoperation
from django.shortcuts import render,get_object_or_404
from mysite.models import Project,Feature,NeuralNetwork


class KuaiTestCase(TestCase):
    def setUp(self):
        print("setup")

    def test_create_project(self):
        author = 'sebastian'
        projectname = 'testproject'

        c = Client()
        response = c.post('/createnewproject/', {'authorNameInput': author, 'projectNameInput': projectname})
        self.assertEqual(response.status_code>199 and response.status_code < 400,True)
        project = get_object_or_404(Project,pk=1)
        self.assertEqual(project.author,author)
        self.assertEqual(project.projectname,projectname)

        project_id = 1
        project_id_str = str(project_id)
        response = c.get('/dashboard/'+project_id_str+'/')
        self.assertEqual(response.status_code>199 and response.status_code < 400,True)


        response = c.post('/dataupload/'+project_id_str+'/', {
            'folderfile':"D:\\test\\*",
            'shuffledata':True,
            'trainshare':0.6,
            'testshare':0.2,
            'cvshare':0.2,
            'datatype':'img'
        })
        self.assertEqual(response.status_code>199 and response.status_code < 400,True)

        response = c.post('/datatransform/'+project_id_str+'/',{
            'selectstatement':'select(udfcategory("image.origin").alias("category"),udfcategory("image.origin").alias("cc22"),udfimage("image").alias("image"))',
            'udfclasses': """def category(value):
        return int(value.split('/')[5])
    
udfcategory = udf(category, IntegerType())

def imagetonp(image):
  import numpy as np
  #result = ImageSchema.toNDArray(value).tolist()
  result = np.ndarray(
        shape=(image.height, image.width, image.nChannels),
        dtype=np.uint8,
        buffer=image.data,
        strides=(image.width * image.nChannels, image.nChannels, 1)).tolist()
  return result 

udfimage = udf(imagetonp, ArrayType(ArrayType(ArrayType(IntegerType()))))"""
        })
        
        self.assertEqual(response.status_code>199 and response.status_code < 400,True)

        project = get_object_or_404(Project,pk=project_id)
        self.assertEqual(project.selectstatement,'select(udfcategory("image.origin").alias("category"),udfcategory("image.origin").alias("cc22"),udfimage("image").alias("image"))')
        
        response = c.post('/dataclassification/'+project_id_str+'/',{
            
        })

        import mysite.dataoperation as md
        self.assertEqual(md.getinputschema(project_id),{})
        
