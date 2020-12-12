import foundations
#from pyspark.sql import SparkSession
from tensorflow import keras
import numpy as np
import pyarrow.parquet as pq

prjdir = 'hi du nase'#prjdir
logdir = prjdir+"/logs"
parquet_files = prjdir+"/data/1.parquet"
parquet_files2 = prjdir+"/data/2.parquet"
model = keras.models.load_model(prjdir+'/model')
print("____________________________________________")
print("_________________Model______________________")
print("____________________________________________")
model.summary()
print("____________________________________________")
print("_________________Model______________________")
print("____________________________________________")
print("____________________________________________")
print("_________________TRAIN______________________")
print("____________________________________________")

import DessaCallback as dc
import PyArrowDataExtraction as de




tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
foundations.set_tensorboard_logdir(logdir)

prjdir = 'hi du nase'#prjdir
depth = 3
prjdir = 'hi du nase'#prjdir
lrate = 1e-3

pds = pq.ParquetDataset(parquet_files)
table = pds.read()
xy = de.getXandYFromPyArrow(table)
model.fit(x=xy['x'],y=xy['y'],epochs=epochs,batch_size = batch_size,callbacks=[tensorboard_callback,dc.CustomCallback()])

print("____________________________________________")
print("_________________TRAIN______________________")
print("____________________________________________")

pds = pq.ParquetDataset(parquet_files2)
table = pds.read()
xy = getXandYFromPyArrow(table)
model.evaluate(x=xy['x'],y=xy['y'],callbacks=[tensorboard_callback,CustomCallback()])

# Log some hyper-parameters
foundations.log_param('depth', depth)
foundations.log_params({'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lrate})
#TODO: save trained model!
#foundations.save_artifact('README.txt', 'Project_README')