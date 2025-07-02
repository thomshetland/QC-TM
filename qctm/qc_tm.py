import numpy as np
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from time import time
import pickle
import kagglehub
from datasets import load_dataset
from transformers import BertTokenizer
from collections import Counter
from scipy import sparse
from sklearn.utils import shuffle

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from sklearn.utils import shuffle

class QC_TM:
    def __init__(self):
        # Initialize any necessary attributes or configurations for QC_TM
        self.config = {}
        print("QC_TM initialized with default configuration.")
    
    def train_test(train_x, train_y, test_x, test_y):
        # Train the model
        from time import time
        print("Training Tsetlin Machine...")

        tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (1, maxlen, vocab_size * 3), (1, 1), grid=(16*13,1,1),
        block=(128,1,1))
        for i in range(epochs):
            start_training = time()
            tm.fit(X_train, Y_train, epochs=1, incremental=True)
            stop_training = time()

            start_testing = time()
            result_test = 100*(tm.predict(X_test) == Y_test).mean()
            stop_testing = time()

            result_train = 100*(tm.predict(X_train) == Y_train).mean()

            print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % 

                (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
            return result_test, result_train    