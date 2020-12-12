import foundations
#from pyspark.sql import SparkSession
from tensorflow import keras
import numpy as np
import pyarrow.parquet as pq

logdir = "./logs"
parquet_files = "./data/1.parquet"
parquet_files2 = "./data/2.parquet"
model = keras.models.load_model('./model')
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
class CustomCallback(keras.callbacks.Callback):
    def loggingtofoundations(self,logs=None):
        if logs:
            for key,value in logs.items():
                foundations.log_metric(key, value)
            

    def on_train_begin(self, logs=None):
        self.loggingtofoundations(logs)
        
    def on_train_end(self, logs=None):
        self.loggingtofoundations(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.loggingtofoundations(logs)

    def on_epoch_end(self, epoch, logs=None):
        self.loggingtofoundations(logs)

    def on_test_begin(self, logs=None):
        self.loggingtofoundations(logs)

    def on_test_end(self, logs=None):
        self.loggingtofoundations(logs)

    def on_predict_begin(self, logs=None):
        self.loggingtofoundations(logs)

    def on_predict_end(self, logs=None):
        self.loggingtofoundations(logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.loggingtofoundations(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.loggingtofoundations(logs)

    def on_test_batch_begin(self, batch, logs=None):
        self.loggingtofoundations(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.loggingtofoundations(logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self.loggingtofoundations(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.loggingtofoundations(logs)

def getXandYFromPyArrow(table):
    ### This needs to be done since for multi dim arrays, the numpy shape is lost...
    ### pyarrow to pylist crashed
    ### numpy to_list does not work recursively
    def recResolution(rec):
        if isinstance(rec,np.ndarray):
            rec_2 = rec.tolist()
            return list(map(lambda x: recResolution(x) if isinstance(x,np.ndarray) else x,rec_2))
        else:
            return rec
    columns = table.column_names
    #columns = table.columns
    featurevalues = list(filter(lambda x:x.startswith("feature"),columns))
    x = []
    for feature in featurevalues:
        x_3 = table[feature]
        x_3 = np.asarray(recResolution(x_3.to_numpy()))
        x.append(x_3)
    if len(x) == 1:
        x = x[0]
    targetvalues = list(filter(lambda x:x.startswith("target"),columns))
    y = []
    for target in targetvalues:
        y_3 = table[target]
        y_3 = np.asarray(recResolution(y_3.to_numpy()))
        y.append(y_3)
    if len(y) == 1:
        y = y[0]
    return {"x":x,"y":y}




tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
foundations.set_tensorboard_logdir(logdir)

epochs = 5
depth = 3
batch_size = 1
lrate = 1e-3

pds = pq.ParquetDataset(parquet_files)
table = pds.read()
xy = getXandYFromPyArrow(table)
model.fit(x=xy['x'],y=xy['y'],epochs=epochs,batch_size = batch_size,callbacks=[tensorboard_callback,CustomCallback()])

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