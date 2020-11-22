# kuai
## This software uses:
Django (https://www.djangoproject.com/)  
Keras/TensorFlow (https://www.tensorflow.org/)  
Spark (https://spark.apache.org/)  
MongoDB (https://www.mongodb.com/de)  
Sleek Dashboard (https://sleek.tafcoder.com/)  
Seaborn (https://seaborn.pydata.org/)  
Atlas Dessa (https://www.atlas.dessa.com/)  
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

## todo
* rename from mysite to kuai
* generate dimensions automatically for basic types
* get type from dataframe schema!
* integrate experiment engine
* write more tests
* create docker container for this
