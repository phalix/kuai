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
            'folderfile': "examples/stock/wkn_846900_historic.csv",
            'shuffledata': True,
            'trainshare': 0.6,
            'testshare': 0.2,
            'cvshare': 0.2,
            'datatype': 'csv'
        })
        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)
        selectstmt = 'select("Erster","Stuecke")'
        udfclasses = ""
        response = c.post('/datatransform/'+project_id_str+'/', {
            'selectstatement': selectstmt,
            'udfclasses': udfclasses
        })

        self.assertEqual(response.status_code >
                         199 and response.status_code < 400, True)

        