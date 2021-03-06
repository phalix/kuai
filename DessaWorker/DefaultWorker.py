import foundations#foundations
#from pyspark.sql import SparkSession
import tensorflow as tf

try:  
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:  # Invalid device or cannot modify virtual devices once initialized.  
    pass
from tensorflow import keras
import numpy as np
import pyarrow.parquet as pq

prjdir = '.'#prjdir
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
print("_________________DATA_______________________")
print("____________________________________________")

import DessaCallback as dc
import PyArrowDataExtraction as de




tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)#tensorboard
csv_callback = keras.callbacks.CSVLogger("experiment_training.log", separator=",", append=False)#csvlogger
csv_callback_test = keras.callbacks.CSVLogger("experiment_testing.log", separator=",", append=False)#csvloggertesting
foundations.set_tensorboard_logdir(logdir)#foundations

pds = pq.ParquetDataset(parquet_files)
pds.split_row_groups = True
table = pds.read()
print(str(table.num_rows))
xy = de.getXandYFromPyArrow(table)


pds2 = pq.ParquetDataset(parquet_files2)
pds2.split_row_groups = True
table2 = pds2.read()
print(str(table2.num_rows))
if not table2.num_rows>0:
    table2 = table
xy2 = de.getXandYFromPyArrow(table2)


print("____________________________________________")
print("_________________DATA_______________________")
print("____________________________________________")
callbacks = []

callbacks.append(tensorboard_callback)#tensorboard
callbacks.append(dc.CustomDessaCallback("train"))#foundations
callbacks.append(csv_callback)

callbacksEvaluate = []
callbacksEvaluate.append(dc.CustomDessaCallback("test"))#foundations
callbacksEvaluate.append(csv_callback_test)

print("____________________________________________")
print("_________________TRAIN______________________")
print("____________________________________________")

#overwrite model if needed optimizer = keras.optimizer.Adam();model.compile(optimizer=optimizer,loss=loss,metrics=metrics)#change optimizer

epochs = 5 #epochs
depth = 3
batch_size = 1 #batchsize
lrate = 1e-3

model.fit(x=xy['x'],
    y=xy['y'],
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    validation_data=(xy2['x'], xy2['y']),
    validation_freq=10,
    )

print("____________________________________________")
print("_________________TRAIN______________________")
print("____________________________________________")
print("____________________________________________")
print("_________________TEST_______________________")
print("____________________________________________")


a = model.evaluate(x=xy['x'],y=xy['y'],callbacks=callbacksEvaluate)
print(a)
b = model.predict(x=xy['x'])
print(b)
model.save(modeldir)
#foundations.save_artifact('README.txt', 'Project_README')

print("____________________________________________")
print("_________________TEST_______________________")
print("____________________________________________")