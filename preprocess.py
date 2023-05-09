from os import makedirs, path
from pickle import dump
import numpy as np

from args import get_parser

# Loads the data from the .txt files and saves them into pickled objects
def load_and_save(filename, dataset, output_folder):
    temp = np.genfromtxt(
        path.join(dataset, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(f"{filename}: {temp.shape}")
    with open(path.join(output_folder, filename+".pkl"), "wb") as file:
        dump(temp, file)

# Each dataset should in general have three files:
# 1. train.txt -> contains csv of features for training
# 2. test.txt -> contains csv of features for evaluation
# 3. labels.txt -> contains the labels of the test data
def load_data(dataset):
    dataset_folder = path.join("datasets", dataset)
    output_folder = path.join(dataset_folder,"processsed")
    # Make the directory if it does not exist
    makedirs(output_folder, exist_ok=True)
    # Load the 3 files and save them
    load_and_save("train.txt", dataset_folder, output_folder)
    load_and_save("test.txt", dataset_folder, output_folder)
    load_and_save("labels.txt", dataset_folder, output_folder)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)