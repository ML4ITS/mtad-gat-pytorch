from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd

from args import get_parser


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)
    elif dataset =="SWAT":
        swat = pd.read_csv(path.join('datasets/data', 'SWaT_Dataset_Attack_v0.csv'))
        swat = swat.drop(' Timestamp', axis=1)
        
        if args.cut < 1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            swat = swat.iloc[:int(len(swat)*args.cut)]
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))
        

        swat = swat.iloc[::int(1/sample_rate)]#resampling
        labels = (swat['Normal/Attack'].values=='Attack')
        values = swat.drop('Normal/Attack', axis=1).values
        
        train_test_split=args.train_test_split

        
        if args.scaler == 'quantile':
            from sklearn.preprocessing  import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            from sklearn.preprocessing  import MinMaxScaler
            scaler = MinMaxScaler()
        
        values = scaler.fit_transform(values) 
        #spectral residual data cleaning
        if args.spectral_residual:
            for i in range(values.shape[1]):
                values[:,i] = spectral_residual_replace(values[:,i])

        train_values = values[:int(train_test_split*len(labels)),:]
        train_labels = labels[:int(train_test_split*len(labels))]

        if args.no_anomaly_train:
            print('removing anomalies from training data')
            train_values = train_values[train_labels==False]

        test_values = values[int(train_test_split*len(labels)):,:]
        test_labels = labels[int(train_test_split*len(labels)):]

        #dump train values into file
        makedirs('datasets/data/processed', exist_ok=True)
        path_pkl = path.join('datasets/data/processed', 'SWAT_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)



        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'SWAT_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)


        #dump test labels into file
        path_pkl = path.join('datasets/data/processed', 'SWAT_test_label.pkl')
        with open(path_pkl, 'wb') as file:
           dump(test_labels, file)

#Spectral residual implementation for simple univariate outlier detection https://arxiv.org/pdf/1906.03821.pdf
import numpy as np
from sklearn.preprocessing import StandardScaler
def spectral_residual_replace(x, tau=2, window_size=20):
    #compute fourier transform
    fft_result = np.fft.fft(x)

    #compute phase and log amplitude of fft
    log_amplitude = np.log(np.abs(fft_result)) 
    phase = np.angle(fft_result)
    
    #smooth the amplitude and compute the residual
    smoothed_log_amplitude = np.convolve(log_amplitude, np.ones(window_size)/window_size, mode = 'same')
    residual_log_amplitude = smoothed_log_amplitude-log_amplitude
    
    #compute the spectral residual
    im_unit = 1j
    sr = np.abs(np.fft.ifft(np.exp(residual_log_amplitude + im_unit*phase)))
    
    #standardize the spectral residual
    scaler = StandardScaler()
    sr = scaler.fit_transform(sr.reshape(-1,1)).reshape(-1)
    
    #identify outliers (sr is now a 0-1 normal distribution)
    outliers =  (sr > tau)
    
    #replace outliers
    x_replaced = x.copy()
    x_replaced[outliers] = np.mean(x) 

    return x_replaced




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
