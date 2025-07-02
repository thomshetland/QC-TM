from data.dataset import Dataset
from qctm.qc_tm import QC_TM
from transformers import BertTokenizer
import pickle

from .src.qctm.qc_tm import QC_TM
from .src.data.dataset import Dataset
from .src.preprocessing import PreProcessing
from .src.thresholds.cosine import CosineThreshold
from .src.thresholds.pmi import PMIThreshold

from config_loader import load_config
def main():
    config = load_config()
    Dataset(config["preprocessing"]["dataset_name"])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    word_profile = pickle.load(open("word_profile.p", "rb"))
    feature_name = pickle.load(open("feature_name.p", "rb"))

    train_x, train_y, test_x, test_y = Dataset(load_config("dataset_name")).load_data()
    preprocessor = PreProcessing(tokenizer=tokenizer)
    word_to_index, vocab_size = preprocessor.create_vocabulary(train_x)
    X_train, Y_train, X_test, Y_test = preprocessor.preprocess_data(
        train_x, train_y, test_x, test_y, vocab_size, word_to_index, feature_name, word_profile, tokenizer
    )
    qc_tm = QC_TM(vocab_size)
    qc_tm.train_test(X_train, Y_train, X_test, Y_test)
if __name__ == "__main__":
    main()