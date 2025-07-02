from data.dataset import Dataset
from qctm.qc_tm import QC_TM

if __name__ == "__main__":

    # Example usage
    dataset = Dataset("example_dataset", "/path/to/data")
    qc_tm = QC_TM()

    print(f"Dataset Name: {dataset.dataset_name}, Data Path: {dataset.data_path}")
    print("QC_TM instance created successfully.")