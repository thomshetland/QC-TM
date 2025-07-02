from datasets import load_dataset
import numpy as np
from sklearn.utils import shuffle
class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def load_data(self):
        if self.dataset_name == "cr":
            train_ds = load_dataset("Setfit/cr", split="train")
            test_ds = load_dataset("Setfit/cr", split="test")

            train_x = np.array(train_ds["text"])
            train_y = np.array(train_ds["label"])

            test_x = np.array(test_ds["text"])
            test_y = np.array(test_ds["label"])

            train_x, train_y = shuffle(train_x, train_y, random_state=42)
            test_x, test_y = shuffle(test_x, test_y, random_state=42)
        
        elif self.dataset_name == "subj":
            train_ds = load_dataset("Setfit/subj", split="train")
            test_ds = load_dataset("Setfit/subj", split="test")

            train_x = np.array(train_ds["text"])
            train_y = np.array(train_ds["label"])

            test_x = np.array(test_ds["text"])
            test_y = np.array(test_ds["label"])
        elif self.dataset_name == "pc":
            train_ds = load_dataset("json", data_files={"train": "procon_Train.json", "test": "procon_Test.json"})

            train_x = [example["text"] for example in train_ds["train"]]
            train_y = [example["label"] for example in train_ds["train"]]

            test_x = [example["text"] for example in train_ds["test"]]
            test_y = [example["label"] for example in train_ds["test"]]

            label_map = {"negative": 0, "positive": 1}


            train_y = np.array([label_map[label] for label in train_y], dtype=np.uint32)
            test_y = np.array([label_map[label] for label in test_y], dtype=np.uint32)

        return train_x, train_y, test_x, test_y
    