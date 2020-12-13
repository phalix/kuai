import foundations#foundations
#from pyspark.sql import SparkSession
from tensorflow import keras
import numpy as np
import pyarrow.parquet as pq

prjdir = ''#prjdir
logdir = prjdir+"/logs"
parquet_files = prjdir+"/data/1.parquet"
parquet_files2 = prjdir+"/data/2.parquet"
modeldir = prjdir+'/model'
model = keras.models.load_model(modeldir)
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




tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)#tensorboard
foundations.set_tensorboard_logdir(logdir)#foundations

epochs = 5 #epochs
depth = 3
batch_size = 1 #batchsize
lrate = 1e-3

pds = pq.ParquetDataset(parquet_files)
pds.split_row_groups = True
table = pds.read()
xy = de.getXandYFromPyArrow(table)
callbacks = []
callbacks.append(tensorboard_callback)#tensorboard
callbacks.append(dc.CustomDessaCallback())#foundations
model.fit(x=xy['x'],y=xy['y'],epochs=epochs,batch_size = batch_size,callbacks=callbacks)

print("____________________________________________")
print("_________________TRAIN______________________")
print("____________________________________________")
print("____________________________________________")
print("_________________TEST_______________________")
print("____________________________________________")
pds = pq.ParquetDataset(parquet_files2)
pds.split_row_groups = True
table = pds.read()
xy = de.getXandYFromPyArrow(table)
callbacksEvaluate = []
callbacksEvaluate.append(dc.CustomDessaCallback())#foundations
a = model.evaluate(x=xy['x'],y=xy['y'],callbacks=callbacksEvaluate)
print(a)
b = model.predict(x=xy['x'])
print(b)
'''# Log some hyper-parameters
foundations.log_param('depth', depth)
foundations.log_params({'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lrate})'''

model.save(modeldir)
#foundations.save_artifact('README.txt', 'Project_README')

print("____________________________________________")
print("_________________TEST_______________________")
print("____________________________________________")