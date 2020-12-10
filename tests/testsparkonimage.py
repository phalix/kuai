from django.test import TestCase
from django.test import Client

from mysite import dataoperation
from django.shortcuts import render, get_object_or_404
from mysite.models import Project, Feature, NeuralNetwork


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
            'feature_category': 'on', 'fttype_category': 'float', 'fttransition_category': '0', 'dimension_category': '1', 'ftreformat_category': '                                        ', 'feature_cc22': 'on', 'fttype_cc22': 'float', 'fttransition_cc22': '3', 'dimension_cc22': '1', 'ftreformat_cc22': '', 'feature_image': 'on', 'fttype_image': 'array<array<array<int>>>', 'fttransition_image': '0', 'dimension_image': '720,1280,3', 'ftreformat_image': '', 'fttype_cc22Pre': 'double', 'fttransition_cc22Pre': '0', 'dimension_cc22Pre': '', 'ftreformat_cc22Pre': '', 'targetselection': 'category'
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/dataclassification/'+project_id_str+'/', {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        import mysite.dataoperation as md
        print(md.getinputschema(project_id))
        self.assertEqual(md.getinputschema(project_id), {1: [1], 3: [[720, 1280, 3]]})
        print(md.getfeaturedimensionbyproject(project.features.all()))
        self.assertEqual(md.getfeaturedimensionbyproject(project.features.all()),{1: {'cc22': [1]}, 3: {'image': [720, 1280, 3]}})

        df = md.readfromcassandra(project_id,1)
        df2 = md.transformdataframe(project,df,1)
        
        project = get_object_or_404(Project, pk=1)

        a = md.buildFeatureVector(df2,project.features.all(),project.target)
        b = md.getXandYFromDataframe(a,project)
        a.show()
        print(b)
        ##TODO: to be complted!