from django.http import HttpResponse, HttpResponseRedirect 
from django.template import loader
from django.urls import reverse
from django.shortcuts import render,get_object_or_404

from datetime import datetime

from django import forms

from .models import Project,Configuration

def index(request):
    template = loader.get_template('project/projects.html')
    
    context = {
        "projects": Project.objects.all(),
        "menuactive":1,
    }
    return HttpResponse(template.render(context, request))

def index2(request,project_id):
    template = loader.get_template('project/projects.html')
    
    context = {
        "projects": Project.objects.all(),
        "menuactive":1,
        "project_id" : project_id,
    }
    return HttpResponse(template.render(context, request))

def new(request):
    template = loader.get_template('project/newproject.html')
    
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

defaultValues = {
    "DessaServer":"http://localhost:5555",
    'spark.jars.packages': 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.0',
    'Author':'Sebastian',
    'maindir':".",
    'mongoconnection':'mongodb://localhost:27017',
    'delimeter':','
}

def getSystemStats(request):
    import psutil, json

    cpu = psutil.cpu_percent()
    memory = psutil.swap_memory().percent
    return HttpResponse(json.dumps({
        'cpu':cpu,
        'memory':memory,
        'gpu':0,
    }))


def getSetting(project_id,fieldname):
    project = get_object_or_404(Project,pk=project_id)
    checkSettingConsistency(project)
    curConfig = project.configuration.filter(fieldname=fieldname).all()
    result = []
    for config in curConfig:
        result.append(config.option)
    return result
    


def setupSettings(request,project_id):
    project = get_object_or_404(Project,pk=project_id)
    checkSettingConsistency(project)
    curConfig = project.configuration.all()
    for value in request.POST:
        if value.startswith("Option"):
            fieldname = value[7:]
            c = curConfig.filter(fieldname=fieldname).all()
            for c_2 in c:
                c_2.option = request.POST[value]
                c_2.save()
    try:
        import mysite.dataoperation as md
        md.getsparkcontext(project_id,1).stop()
    except:
        print("Could not stop Spark")
    return HttpResponseRedirect('/settings/'+str(project_id))


def editProjectSettings(request,project_id):
    template = loader.get_template('project/settings.html')
    project = get_object_or_404(Project,pk=project_id)
    checkSettingConsistency(project)
    
    curConfig = project.configuration.all()

    context = {
        "curConfig":curConfig,
        "project_id":project_id,
        "project":project,
    }
    return HttpResponse(template.render(context, request))

def checkSettingConsistency(project):
    curConfig = project.configuration.all()
    for key,value in defaultValues.items():
        if len(list(curConfig.filter(fieldname=key)))==0:
            c = Configuration(fieldname=key,option=value)
            c.save()
            project.configuration.add(c)
            project.save()
    
    