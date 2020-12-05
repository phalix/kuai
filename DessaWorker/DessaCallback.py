from tensorflow import keras

class CustomDessaCallback(keras.callbacks.Callback):
    import foundations
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