from django.test import TestCase
from django.test import Client

from mysite import dataoperation
from django.shortcuts import render, get_object_or_404
from mysite.models import Project, Feature, NeuralNetwork


class TestKuaiCase(TestCase):
    def setUp(self):
        print("setup")

    def test_create_project(self):
        import mysite.dataoperation as md
        import json
        author = 'sebastian'
        projectname = 'testproject'

        c = Client()
        project_id = 111111
        response = c.post(
            '/createnewproject/', {'authorNameInput': author, 'projectNameInput': projectname,'pk':project_id})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        project = get_object_or_404(Project, pk=project_id)
        self.assertEqual(project.author, author)
        self.assertEqual(project.projectname, projectname)

        
        project_id_str = str(project_id)
        response = c.get('/dashboard/'+project_id_str+'/')
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        fp = open("examples/iris/iris.data")

        response = c.post('/dataupload/'+project_id_str+'/', {
            'filewithdata': fp,
            'folderfile': '',
            'shuffledata': True,
            'trainshare': 0.6,
            'testshare': 0.2,
            'cvshare': 0.2,
            'datatype': 'csv'
        })
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'FloatType',
            'columns':'a',
            'udfcode1':'return float(output)'
        })

        json.loads(response.content)['id']

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        
        self.assertEqual(float(json.loads(response.content)['result'])>=0, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'FloatType',
            'columns':'b',
            'udfcode1':'return float(output)'
        })

        json.loads(response.content)['id']
        self.assertEqual(float(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'FloatType',
            'columns':'c',
            'udfcode1':'return float(output)'
        })

        json.loads(response.content)['id']
        self.assertEqual(float(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'FloatType',
            'columns':'d',
            'udfcode1':'return float(output)'
        })

        json.loads(response.content)['id']
        self.assertEqual(float(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        
        project = get_object_or_404(Project, pk=project_id)
        
        response = c.post('/setupdataclassifcation/'+project_id_str+'/', {
            'feature_a_1': 'on', 
            'fttype_a_1': 'int', 
            'fttransition_a_1': '3', 
            'dimension_a_1': '1', 
            'ftreformat_a_1': 'VectorAssembler,StandardScaler', 
            'feature_b_2': 'on', 
            'fttype_b_2': 'int', 
            'fttransition_b_2': '3', 
            'dimension_b_2': '1', 
            'ftreformat_b_2': 'VectorAssembler,StandardScaler',
            'feature_c_3': 'on', 
            'fttype_c_3': 'int', 
            'fttransition_c_3': '0', 
            'dimension_c_3': '1', 
            'ftreformat_c_3': 'VectorAssembler,StandardScaler',
            'feature_d_4': 'on', 
            'fttype_d_4': 'int', 
            'fttransition_d_4': '0', 
            'dimension_d_4': '1', 
            'ftreformat_d_4': 'VectorAssembler,StandardScaler',
            'feature_klasse': 'on', 
            'fttype_klasse': 'int', 
            'fttransition_klasse': '0', 
            'dimension_klasse': '1', 
            'ftreformat_klasse': 'StringIndexer,OneHotEncoder',
            'targetselection': 'klasse'
            
        })
        

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/dataclassification/'+project_id_str+'/', {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        ###SETUP Neural Network


        response = c.post('/aiupload/'+project_id_str+'/', {
            'layer1':'Dense',
            'para1$units%5':'4',
            'para1$activation%5':'relu',
            'states[]1':'Input1_(1,None)'
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        
        response = c.post('/aioptandoutupload/'+project_id_str+'/', {
            'layer2':'Dense',
            'para2$units%5':'2',
            'para2$activation%5':'softmax',
            'states[]2':'1',
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
            'loss':'MAE',
            'metrics[]':'MSE',
            'noofepochs':'5',
            'batchsize':'4',
            'optimizerselect':'Adam',
            'optpara$learning_rate':'0.1',
            'experimenttypeselect':'PlainPythonExperiment',
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post("/writetodessa/"+project_id_str+"/"+exp_id_str+"/", {
            'writeDessa':'false'
        })
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        
        #write to dessa and run
        response = c.post("/startExperimentsPerProject/"+project_id_str+"/"+exp_id_str+"/", {
            'writeDessa':'false'
        })
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        print("done")