from tensorflow import keras


class CustomDessaCallback(keras.callbacks.Callback):
    
    def __init__(self,prefix):
        keras.callbacks.Callback.__init__(self)
        self.prefix = prefix


    def loggingtofoundations(self,logs=None):
        import foundations
        if logs:
            for key,value in logs.items():
                foundations.log_metric(str(self.prefix)+key, value)
            

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