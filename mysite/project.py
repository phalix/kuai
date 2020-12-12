from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404

from datetime import datetime

from django import forms

from .models import Project

def index(request):
    template = loader.get_template('projects.html')
    
    context = {
        "projects": Project.objects.all(),
        "menuactive":1,
    }
    return HttpResponse(template.render(context, request))

def index2(request,project_id):
    template = loader.get_template('projects.html')
    
    context = {
        "projects": Project.objects.all(),
        "menuactive":1,
        "project_id" : project_id,
    }
    return HttpResponse(template.render(context, request))

def new(request):
    template = loader.get_template('newproject.html')
    
    context = {
    
    }
    return HttpResponse(template.render(context, request))

def createnewproject(request):
    authorname = request.POST['authorNameInput']
    projectname = request.POST['projectNameInput']
    
    p = Project(author=authorname,projectname=projectname,datacreated=datetime.now())
    if 'pk' in request.POST:
        pk = request.POST['pk']
        p.pk = pk
    
    p.save()
    return HttpResponseRedirect('/dashboard/'+str(p.pk))
    
    