from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404


from django import forms

from .models import Project

def index(request,project_id):
    template = loader.get_template('project/dashboard.html')
    project = get_object_or_404(Project, pk=project_id)
    context = {
        "project" : project,
        "project_id" : project_id,
        "menuactive": 1,
    }
    return HttpResponse(template.render(context, request))

def saveexecutionmodel(request,project_id):
    project = get_object_or_404(Project, pk=project_id)
    cm = request.POST['execution_text']
    project.execution_text = cm
    project.save()
    return HttpResponseRedirect('/dashboard/'+str(project_id))


    