# kuAI [ka̠i]
[![Build Status](https://travis-ci.com/phalix/kuai.svg?branch=master)](https://travis-ci.com/phalix/kuai)
## This software uses:
Django (https://www.djangoproject.com/)  
Keras/TensorFlow (https://www.tensorflow.org/)  
Spark (https://spark.apache.org/)  
MongoDB (https://www.mongodb.com/de)  
Sleek Dashboard (https://sleek.tafcoder.com/)  
Seaborn (https://seaborn.pydata.org/)  
Atlas Dessa (https://www.atlas.dessa.com/)  
Apache Arrow (https://arrow.apache.org/)  
and everything is programmed in Python.  

Integration with https://www.atlas.dessa.com/ or https://www.openml.org/

As well as an AutoML Framework and a diagram framework.

## What is Kuai
Kuai enables a quick protoyping approach for neural networks.  
It enables the classical workflow of  
* data preparation with spark (In Progress)
* AI modelling with a graphical UI for Keras/Tensorflow (In Progress)  
* Simulation and Test  (In Progress) 
* Deployment (to be done)

Everything is supported by easy to use user interfaces.

## setup
install java  
install python and pip > 3  
pip install pandas Django pyspark tensorflow keras notebook ipywidgets seaborn matplotlib

## runapp
run "python manage.py runserver"
## Warning
Before deploying you should change the SECRET_KEY in mysite/settings.py, e.g. with https://djecrety.ir/  

## TODO
* Make Jobs for Dessa Configurable!
* Make a configuration of Dessa Jobs
* rename from mysite to kuai
* get type from dataframe schema!
* write more tests
* create docker container for this
* create udf assistant
* store diagram definitions and recreate
* support multiple output
* enable asynchronous background processes for data transformation
* permanently store in database and restore
* support GANs and Auto Encoders through configurable Experiments

