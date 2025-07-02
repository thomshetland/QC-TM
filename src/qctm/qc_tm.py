import numpy as np
import pickle
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from config_loader import load_config, get_training_config, get_preprocessing_config

class QC_TM:
    def __init__(self, vocab_size, config_path="config.yaml"):
        """Initialize QC_TM with config from YAML file."""
        self.training_config = get_training_config(config_path)
        self.preprocessing_config = get_preprocessing_config(config_path)
        # Set default values
        self.clauses = self.training_config.get('clauses', 10000)
        self.T = self.training_config.get('T', 10000)
        self.s = self.training_config.get('s', 1.0)
        self.epochs = self.training_config.get('epochs', 50)
        self.vocab_size = vocab_size
        self.maxlen = self.preprocessing_config.get('maxlen', 20)

    def train_test(self, X_train, Y_train, X_test, Y_test):
                # Train the model
        from time import time
        print("Training Tsetlin Machine...")

        tm = MultiClassConvolutionalTsetlinMachine2D(self.clauses, self.T, self.s, (1, self.maxlen, self.vocab_size * 5), (1, 1), grid=(16*13,1,1),
        block=(128,1,1))
        for i in range(self.epochs):
            start_training = time()
            tm.fit(X_train, Y_train, epochs=1, incremental=True)
            stop_training = time()

            start_testing = time()
            result_test = 100*(tm.predict(X_test) == Y_test).mean()
            stop_testing = time()

            result_train = 100*(tm.predict(X_train) == Y_train).mean()

            print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % 

                (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
  
        