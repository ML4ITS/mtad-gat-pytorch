import ast
import csv
import os
from pickle import dump

import numpy as np

import args


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset="SMD"):
    """Author: https://github.com/NetManAIOps/OmniAnomaly/"""
    if dataset == "SMD":
        output_folder = "ServerMachineDataset/processed"
        os.makedirs(output_folder, exist_ok=True)
        dataset_folder = "ServerMachineDataset"
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save("train", filename, filename.strip(".txt"), dataset_folder, output_folder)
                load_and_save("test", filename, filename.strip(".txt"), dataset_folder, output_folder)
                load_and_save("test_label", filename, filename.strip(".txt"), dataset_folder, output_folder)
    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/smap_and_msl"
        output_folder = f"{dataset_folder}/processed"
        os.makedirs(output_folder, exist_ok=True)

        with open(os.path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])

        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)

        print(dataset, "test_label", labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)


if __name__ == "__main__":
    ds = args.get_parsed_args().dataset.upper()
    load_data(ds)
