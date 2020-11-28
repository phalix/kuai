"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path

from . import dashboard, neuralnetwork, project, dataoperation, experiments


urlpatterns = [
    path('admin/', admin.site.urls),
    #general
    path('new/', project.new), 
    
    path('createnewproject/', project.createnewproject, name='createnewproject'),
    #by projectid
    path('dashboard/<int:project_id>/', dashboard.index), 
    path('saveexecutionmodel/<int:project_id>/', dashboard.saveexecutionmodel, name='saveexecutionmodel'),
    path('createOrUpdateUDF/<int:project_id>/', dataoperation.createOrUpdateUDF, name='createOrUpdateUDF'),
    path('open/<int:project_id>/', project.index2), 
    path('openproject/<int:project_id>/', dashboard.index,name="openproject"),
    path('datasetup/<int:project_id>/', dataoperation.index),
    path('transform/<int:project_id>/', dataoperation.setuptransformdata),
    path('dataupload/<int:project_id>/', dataoperation.uploaddata,name="uploaddata"), 
    path('datatransform/<int:project_id>/', dataoperation.transformdata,name="transformdata"), 
    path('dataselection/<int:project_id>/', dataoperation.index,name="dataselection"), 
    path('dataanalysis/<int:project_id>/', dataoperation.analysis,name="dataanalysis"),
    path('analysisbychart/<int:project_id>/', dataoperation.analysisbychart,name="dataanalysisbychart"), 
    path('setupdataclassifcation/<int:project_id>/',dataoperation.setupdataclassifcation,name="setupdataclassifcation"),
    path('dataclassification/<int:project_id>/', dataoperation.dataclassification,name="dataclassification"), 
    path('ai/<int:project_id>/', neuralnetwork.index), 
    path('modelsummary/<int:project_id>/', neuralnetwork.modelsummary), 
    path('aiupload/<int:project_id>/', neuralnetwork.aiupload,name="uploadnetworkconf"), 
    path('aioptandoutupload/<int:project_id>/', neuralnetwork.aioptandoutupload,name="uploadoptandout"), 
    path('optimizer/<int:project_id>/', neuralnetwork.optimizer),
    path('parameter/<int:project_id>/', experiments.parameter),
    path('experimentsetup/<int:project_id>/<int:experiment_id>/', experiments.experimentsetup),
    path('experimentsetup/<int:project_id>/', experiments.experimentsetuplastexperiment),
    path('setparameter/<int:project_id>/', experiments.setparameter,name="setparameter"),
    path('runexperiment/<int:project_id>/<int:experiment_id>/', experiments.run),
    path('runexperiment/<int:project_id>/', experiments.runlatestexperiment),
    path('uploadexpsetup/<int:project_id>/<int:experiment_id>/', experiments.uploadexpsetup,name="uploadexpsetup"), 


    #no projectid
    path('dashboard/', project.index), 
    path('open/', project.index), 
    path('datasetup/', project.index),
    path('dataselection/', project.index), 
    path('dataclassification/', project.index), 
    path('ai/', project.index), 
    path('aiupload/', project.index), 
    path('optimizer/', project.index),
    path('parameter/', project.index),
    path('experimentsetup/', project.index),
    path('setparameter/', project.index),
    path('runexperiment/',project.index),
    path('transform/',project.index),

    path('getExperimentsPerProject/<int:project_id>/<int:experiment_id>/', experiments.getexperimentsstasperproject, name='experiment_status'),
    path('startExperimentsPerProject/<int:project_id>/<int:experiment_id>/', experiments.startExperimentsPerProject, name='start_experiment'),
    path('stopExperimentsPerProject/<int:project_id>/<int:experiment_id>/', experiments.stopExperimentsPerProject, name='stop_experiment'),
    path('deleteExperimentsPerProject/<int:project_id>/<int:experiment_id>/', experiments.deleteExperimentData, name='delete_experiment'),

]
