from collections import Counter
from config_loader import load_config, get_training_config, get_preprocessing_config
from scipy.sparse import lil_matrix
import numpy as np
from ..thresholds.cosine import CosineSimilarity

class PreProcessing:

    def __init__(self, config_path="config.yaml", tokenizer=None):
        self.preprocessing_config = get_preprocessing_config(config_path)
        self.num_words = self.preprocessing_config.get('num_words')
        self.maxlen = self.preprocessing_config.get('maxlen')
        self.cosine_threshold = self.preprocessing_config.get('cosine_threshold')
        self.tokenizer = tokenizer

        
    def create_vocabulary(self, train_x):

        word_counts = Counter()
        i = 0
        for text in train_x:
           print(f"Tokenizing document {i+1}/{len(train_x)}")
           i += 1
           tokens = self.tokenizer.tokenize(text)
           word_counts.update(tokens)

        vocab = ["[PAD]", "[UNK]"] + [word for word, _ in word_counts.most_common(self.num_words)]
        vocab_size = len(vocab)
        word_to_index = {word: index for index, word in enumerate(vocab)}
        return word_to_index, vocab_size
    
    def preprocess_data(self, train_x, train_y, test_x, test_y, vocab_size, word_to_index, feature_names_vocab, word_profile, tokenizer, CosineSimilarity = CosineSimilarity):
       
        X_train = lil_matrix((len(train_x), self.maxlen * vocab_size * 5), dtype=np.uint32)
        Y_train = train_y.astype(np.uint32)

        for i in range(len(train_x)): # for documet_idx, document in enumerate(train_x): Loops thorugh the documents
            print(f"Processing document {i + 1} / {len(train_x)}")
            doc_tokens = tokenizer.tokenize(train_x[i])
            if len(doc_tokens) < self.maxlen:
                doc_tokens += ["<PAD>"] * (self.maxlen - len(doc_tokens))
            elif len(doc_tokens) > self.maxlen:
                doc_tokens = doc_tokens[:self.maxlen]
            
            for j in range(len(doc_tokens)):  # for i, word_id in enumerate(document): Loops through each token in the document
                        
                if j < self.maxlen and doc_tokens[j] in word_to_index:
                    index = word_to_index[doc_tokens[j]]
                    slot_start = j * vocab_size * 5
                    X_train[i, slot_start + index + (2 * vocab_size)] = 1 # Set the slot for the token to 1
                    
                    for k in range(len(doc_tokens)):
                        if k != j and k < self.maxlen and doc_tokens[k] in word_to_index:
                            other_word = doc_tokens[k] # Gets the other word
                            other_index = word_to_index.get(other_word) # Gets the index of the other word
                            if other_index is None:
                                other_index = word_to_index["<UNK>"]  # Assign <UNK> index if not found
                            if k == j - 1:
                                X_train[i, slot_start + other_index + vocab_size] = 1
                            elif k == j + 1:
                                X_train[i, slot_start + other_index + (3 * vocab_size)] = 1
                            else:
                                #cosine_value = check_cosine(doc_tokens[j], other_word, word_to_index, word_profile)
                                cosine_value = CosineSimilarity.check_cosine(doc_tokens[j], other_word, feature_names_vocab, word_profile) # Checks the PMI between the words
                                if cosine_value >= self.cosine_threshold:  
                                    if k < j:
                                        X_train[i, slot_start + other_index] = 1
                                    
                                    else:
                                        X_train[i, slot_start + other_index + (4 * vocab_size)] = 1
                                        
                                
        X_train = X_train.tocsr()  # Convert to CSR format for efficient processing
                    
        X_test = lil_matrix((len(test_x), self.maxlen * vocab_size * 5), dtype=np.uint32)
        Y_test = test_y.astype(np.uint32)
        for i in range(len(test_x)):
            print(f"Processing document {i + 1} / {len(test_x)}")
            doc_tokens = tokenizer.tokenize(test_x[i])
            if len(doc_tokens) < self.maxlen:
                doc_tokens += ["<PAD>"] * (self.maxlen - len(doc_tokens))
            elif len(doc_tokens) > self.maxlen:
                doc_tokens = doc_tokens[:self.maxlen]
            
            for j in range(len(doc_tokens)):
                if j < self.maxlen and doc_tokens[j] in word_to_index:
                    index = word_to_index[doc_tokens[j]]
                    slot_start = j * vocab_size * 5
                    X_test[i, slot_start + index + (2 * vocab_size)] = 1  # Set the slot for the token to 1
                    
                    for k in range(len(doc_tokens)):
                        if k != j and k < self.maxlen and doc_tokens[k] in word_to_index:
                            other_word = doc_tokens[k] # Gets the other word
                            other_index = word_to_index.get(other_word) # Gets the index of the other word
                            if other_index is None:
                                other_index = word_to_index["<UNK>"]  # Assign <UNK> index if not found
                            if k == j - 1:
                                X_test[i, slot_start + other_index + vocab_size] = 1
                            elif k == j + 1:
                                X_test[i, slot_start + other_index + (3 * vocab_size)] = 1
                            else:
                                #cosine_value = check_cosine(doc_tokens[j], other_word, word_to_index, word_profile)
                                cosine_value = CosineSimilarity.check_cosine(doc_tokens[j], other_word, feature_names_vocab, word_profile) # Checks the PMI between the words
                                if cosine_value >= self.cosine_threshold:  
                                    if k < j:
                                        X_test[i, slot_start + other_index] = 1
                                    
                                    else:
                                        X_test[i, slot_start + other_index + (4 * vocab_size)] = 1
                                    

        X_test = X_test.tocsr()  # Convert to CSR format for efficient processing 

        return X_train, Y_train, X_test, Y_test

