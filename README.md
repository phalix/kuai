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

## What is Kuai
Kuai enables a quick protoyping approach for neural networks.  
It enables the classical workflow of  
* data preparation with spark (In Progress)
* AI modelling with a graphical UI for Keras/Tensorflow (In Progress)  
* Simulation and Test  (In Progress) 
* Deployment (to be done)

Everything is supported by easy to use user interfaces.

## Setup
install java  
install python and pip == 3.8.6  
pip install -r  requirements.txt  
python manage.py makemigrations  
python manage.py migrate  

## Run application
python manage.py runserver  

## Warning
Before deploying you should change the SECRET_KEY in mysite/settings.py, e.g. with https://djecrety.ir/  

## Docker
https://hub.docker.com/r/bittmans/kuai

## TODO
* rename from mysite to kuai
* support multiple output
* enable asynchronous background processes for data transformation
* support GANs and Auto Encoders through configurable Experiments

