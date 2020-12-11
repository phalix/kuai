from django.test import TestCase
from django.test import Client

from mysite import dataoperation
from django.shortcuts import render, get_object_or_404
from mysite.models import Project, Feature, NeuralNetwork
import json

class TestKuaiCase(TestCase):
    def setUp(self):
        print("setup")

    def test_create_project(self):
        author = 'sebastian'
        projectname = 'testproject'

        c = Client()
        response = c.post(
            '/createnewproject/', {'authorNameInput': author, 'projectNameInput': projectname})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        project = get_object_or_404(Project, pk=1)
        self.assertEqual(project.author, author)
        self.assertEqual(project.projectname, projectname)

        project_id = 1
        project_id_str = str(project_id)
        response = c.get('/dashboard/'+project_id_str+'/')
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/dataupload/'+project_id_str+'/', {
            'folderfile': "D:\\test\\*",
            'shuffledata': True,
            'trainshare': 0.6,
            'testshare': 0.2,
            'cvshare': 0.2,
            'datatype': 'img'
        })
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)


        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'ArrayType,ArrayType,ArrayType,IntegerType',
            'columns':'image.data',
            'udfcode1':'import numpy as np\n#print(np.reshape(output,(720,1280,3)).dtype)\nreturn np.reshape(output,(720,1280,3)).tolist()'
        })

        json.loads(response.content)['id']

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'FloatType',
            'columns':'image.height',
            'udfcode1':'return float(output)'
        })

        json.loads(response.content)['id']

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/datatransform/'+project_id_str+'/', {
            'selectstatement': 'select(udfcategory("image.origin").alias("category"),udfcategory("image.origin").alias("cc22"),udfimage("image").alias("image"))',
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

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        project = get_object_or_404(Project, pk=project_id)
        self.assertEqual(project.selectstatement,
                         'select(udfcategory("image.origin").alias("category"),udfcategory("image.origin").alias("cc22"),udfimage("image").alias("image"))')
        
        response = c.post('/setupdataclassifcation/'+project_id_str+'/', {
            'feature_imagedata_1': 'on', 
            'fttype_imagedata_1': 'ArrayType,ArrayType,ArrayType,IntegerType', 
            'fttransition_imagedata_1': '0', 
            'dimension_imagedata_1': '720,1280,3', 
            'ftreformat_imagedata_1': '                                        ', 
            'feature_imageheight_2': 'on', 
            'fttype_imageheight_2': 'IntegerType', 
            'fttransition_imageheight_2': '3', 
            'dimension_imageheight_2': '1', 
            'ftreformat_imageheight_2': '',
            'targetselection': 'imageheight_2'
            
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/dataclassification/'+project_id_str+'/', {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        import mysite.dataoperation as md
        print(md.getinputschema(project_id))
        self.assertEqual(md.getinputschema(project_id), {3: [[720, 1280, 3]]})
        print(md.getfeaturedimensionbyproject(project.features.all()))
        self.assertEqual(md.getfeaturedimensionbyproject(project.features.all()),{3: {'imagedata_1': [720, 1280, 3]}})


        response = c.post('/aiupload/'+project_id_str+'/', {
            'layer1':'Conv2D',
            'para1$filters%5':'1',
            'para1$kernel_size%5':'(720,1280)',
            'states[]1':'Input1_(720,1280,3,None)',
            'layer2':'Flatten',
            'states[]2':'1',

        })

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        
        response = c.post('/aioptandoutupload/'+project_id_str+'/', {
            'layer3':'Dense',
            'para3$units%5':'1',
            'para3$activation%5':'sigmoid',
            'states[]3':'2',
            'optimizerselect':'Adam',
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        
        response = c.post('/modelsummary/'+project_id_str+'/', {
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        print(response.content)


       ## DESSA Integration
        import mysite.experiments as me
        exp_id = me.getlatestexperiment(project_id)
        exp_id_str = str(exp_id)
        response = c.post('/uploadexpsetup/'+project_id_str+'/'+exp_id_str+'/', {
            'loss':'categorical_crossentropy',
            'metrics[]':' categorical_crossentropy,MSE,MAE',
            'noofepochs':'5',
            'batchsize':'4',
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post("/writetodessa/"+project_id_str+"/"+exp_id_str+"/", {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        
        #write to dessa and run
        response = c.post("/startExperimentsPerProject/"+project_id_str+"/"+exp_id_str+"/", {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        print("done")