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

        import pandas as pd
        inputx = [0,0,1,1]
        inputy = [0,1,0,1]
        outputand = [0,0,0,1]
        outputxor = [0,1,1,0]
        outputor = [0,1,1,1]
        typea = [0,0,0,0]

        pdf = pd.DataFrame(list(zip(inputx, inputy,outputand,outputxor,outputor,typea)), 
               columns =['inputx', 'inputy','outputand','outputxor','outputor','type']) 

        spark = md.getsparksession(project_id,1)
        df = spark.createDataFrame(pdf)

        md.savetocassandra(project_id,df,md.TRAIN_DATA_QUALIFIER)


        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'IntegerType',
            'columns':'inputx',
            'udfcode1':'return output'
        })

        json.loads(response.content)['id']

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        
        self.assertEqual(int(json.loads(response.content)['result'])>=0, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'IntegerType',
            'columns':'inputy',
            'udfcode1':'return output'
        })

        json.loads(response.content)['id']
        self.assertEqual(int(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'IntegerType',
            'columns':'outputand',
            'udfcode1':'return output'
        })

        json.loads(response.content)['id']
        self.assertEqual(int(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'IntegerType',
            'columns':'outputor',
            'udfcode1':'return output'
        })

        json.loads(response.content)['id']
        self.assertEqual(int(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/createOrUpdateUDF/'+project_id_str+'/',{
            'id':0,
            'action':'ADD',
            'outputtype':'IntegerType',
            'columns':'outputxor',
            'udfcode1':'return output'
        })

        json.loads(response.content)['id']
        self.assertEqual(int(json.loads(response.content)['result'])>=0, True)
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)




        selectstmt = 'select("inputx","inputy","outputand")'
        udfclasses = ""
        response = c.post('/datatransform/'+project_id_str+'/', {
            'selectstatement': selectstmt,
            'udfclasses': udfclasses
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        



        project = get_object_or_404(Project, pk=project_id)
        self.assertEqual(project.selectstatement,
                         selectstmt)
        
        response = c.post('/setupdataclassifcation/'+project_id_str+'/', {
            'feature_inputx_1': 'on', 
            'fttype_inputx_1': 'int', 
            'fttransition_inputx_1': '0', 
            'dimension_inputx_1': '1', 
            'ftreformat_inputx_1': '                                        ', 
            'feature_inputy_2': 'on', 
            'fttype_inputy_2': 'int', 
            'fttransition_inputy_2': '0', 
            'dimension_inputy_2': '1', 
            'ftreformat_inputy_2': '                                        ',
            'feature_outputand_3': 'on', 
            'fttype_outputand_3': 'int', 
            'fttransition_outputand_3': '0', 
            'dimension_outputand_3': '1', 
            'ftreformat_outputand_3': '                                        ',
            'feature_outputor_4': 'on', 
            'fttype_outputor_4': 'int', 
            'fttransition_outputor_4': '0', 
            'dimension_outputor_4': '1', 
            'ftreformat_outputor_4': '                                        ',
            'feature_outputxor_5': 'on', 
            'fttype_outputxor_5': 'int', 
            'fttransition_outputxor_5': '0', 
            'dimension_outputxor_5': '1', 
            'ftreformat_outputxor_5': '                                        ',
            'targetselection': 'outputxor_5'
            
        })
        

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        response = c.post('/dataclassification/'+project_id_str+'/', {})
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        ###SETUP Neural Network


        response = c.post('/aiupload/'+project_id_str+'/', {
            'layer1':'Dense',
            'para1$units%5':'1',
            'para1$activation%5':'relu',
            'states[]1':'Input1_(1,None)'
        })
        
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        
        response = c.post('/aioptandoutupload/'+project_id_str+'/', {
            'layer2':'Dense',
            'para2$units%5':'1',
            'para2$activation%5':'sigmoid',
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