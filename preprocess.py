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
    elif dataset =="SKAB":
        ## import ##
        skab_no_attack = pd.read_csv(path.join('datasets/data/SKAB', 'anomaly-free.csv'), delimiter=';')
        skab_no_attack = skab_no_attack.drop('datetime', axis=1)

        skab_attack = pd.read_csv(path.join('datasets/data/SKAB/attacks', '1.csv'),  delimiter=';')
        skab_attack = skab_attack.drop('datetime', axis=1)
        skab_attack = skab_attack.drop('changepoint', axis=1)
        
        ## cutting ##
        if args.cut<1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            skab_no_attack = skab_no_attack.iloc[:int(len(skab_no_attack)*args.cut)]
        
        ## resampling ##
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))
        skab_no_attack = skab_no_attack.iloc[::int(1/sample_rate)]#resampling
        skab_attack = skab_attack.iloc[::int(1/sample_rate)]#resampling
        
        train_values = skab_no_attack.values
        test_values = skab_attack.drop('anomaly', axis=1).values
        test_labels = (skab_attack['anomaly'].values==1)

        ## scaling ##
        if args.scaler == 'quantile':
            from sklearn.preprocessing  import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            from sklearn.preprocessing  import MinMaxScaler
            scaler = MinMaxScaler()
        
        train_values = scaler.fit_transform(train_values)
        test_values = scaler.transform(test_values)

        #dump train values into file
        makedirs('datasets/data/processed', exist_ok=True)
        path_pkl = path.join('datasets/data/processed', 'SKAB_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)

        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'SKAB_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)

        #dump test labels into file
        path_pkl = path.join('datasets/data/processed', 'SKAB_test_label.pkl')
        with open(path_pkl, 'wb') as file:
           dump(test_labels, file)
    elif dataset=='WADI':
        
        wadi = pd.read_csv(path.join('datasets/data', 'WADI_attackdataLABLE.csv'), delimiter=',', skiprows=1 )
        wadi = wadi.drop('Row ', axis=1)
        wadi = wadi.drop('Date ', axis=1)
        wadi = wadi.drop('Time', axis=1)
        wadi = wadi.drop('2_LS_001_AL', axis=1) #nan column
        wadi = wadi.drop('2_LS_002_AL', axis=1) #nan column
        wadi = wadi.drop('2_P_001_STATUS', axis=1) #nan column
        wadi = wadi.drop('2_P_002_STATUS', axis=1) #nan column
        wadi = wadi.dropna(axis=0)

        if args.cut < 1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            wadi = wadi.iloc[:int(len(wadi)*args.cut)]
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))

        wadi = wadi.iloc[::int(1/sample_rate)]#resampling
        labels = (wadi['Attack LABLE (1:No Attack, -1:Attack)'].values==-1)
        values = wadi.drop('Attack LABLE (1:No Attack, -1:Attack)', axis=1).values
        
        train_test_split=args.train_test_split

        if args.scaler == 'quantile':
            from sklearn.preprocessing  import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform')
        if args.scaler =='standard':
            from sklearn.preprocessing  import StandardScaler
            scaler = StandardScaler()
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
        path_pkl = path.join('datasets/data/processed', 'WADI_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)



        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'WADI_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)


        #dump test labels into file
        path_pkl = path.join('datasets/data/processed', 'WADI_test_label.pkl')
        with open(path_pkl, 'wb') as file:
           dump(test_labels, file)
    elif dataset =="ACT":
        X_1 = pd.read_csv(path.join('datasets/data/ACT/Train', 'X_train.txt'), delimiter=' ', header=None)
        X_2 = pd.read_csv(path.join('datasets/data/ACT/Test', 'X_test.txt'), delimiter=' ', header=None)
        values = pd.concat([X_1, X_2], axis=0, ignore_index=True)

        y_1 = pd.read_csv(path.join('datasets/data/ACT/Train', 'y_train.txt'), delimiter=' ', header=None)
        y_2 = pd.read_csv(path.join('datasets/data/ACT/Test', 'y_test.txt'), delimiter=' ', header=None)
        y = pd.concat([y_1, y_2], axis=0, ignore_index=True)
        labels = np.array([x in range(7,13) for x in y.values])

        if args.cut < 1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            values = values.iloc[:int(len(values)*args.cut)]
            labels = labels[:int(len(labels)*args.cut)]
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))

        values = values.iloc[::int(1/sample_rate)].values#resampling
        labels = labels[::int(1/sample_rate)]#resampling

        train_test_split=args.train_test_split

        if args.scaler == 'quantile':
            from sklearn.preprocessing  import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform')
        if args.scaler =='standard':
            from sklearn.preprocessing  import StandardScaler
            scaler = StandardScaler()
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
        path_pkl = path.join('datasets/data/processed', 'ACT_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)



        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'ACT_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)


        #dump test labels into file
        path_pkl = path.join('datasets/data/processed', 'ACT_test_label.pkl')
        with open(path_pkl, 'wb') as file:
           dump(test_labels, file)
    
    
    elif dataset=='METRO':
        
        metro = pd.read_csv(path.join('datasets/data', 'MetroPT3.csv'))
        metro.timestamp = pd.to_datetime(metro.timestamp)

        start_attack = ['2020-04-18 00:00:00', '2020-05-29 23:30:00', '2020-06-05 10:00:00', '2020-07-15 14:30:00']
        end_attack = ['2020-04-18 23:59:00', '2020-05-30 06:00:00', '2020-06-07 14:30:00', '2020-07-15 19:00:00']
        label = np.zeros(metro.shape[0])
        for i in range(4):
            label += ((metro.timestamp>=start_attack[i] ) & (metro.timestamp<=end_attack[i])).values
        label = label==1

        test_mask = (metro.timestamp >= '2020-04-17 00:00:00') & (metro.timestamp <= '2020-07-16 00:00:00')
        train_mask = np.logical_not(test_mask)
        metro_test = metro[test_mask]
        test_label = label[test_mask]
        metro_train = metro[train_mask]
        train_label = label[train_mask]

        if args.cut < 1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            metro_train = metro_train.iloc[:int(len(metro_train)*args.cut)]
            metro_test = metro_test.iloc[:int(len(metro_test)*args.cut)]
            train_label = train_label[:int(len(train_label)*args.cut)]
            test_label = test_label[:int(len(test_label)*args.cut)]
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))

        metro_train = metro_train.iloc[::int(1/sample_rate)]#resampling
        metro_test = metro_test.iloc[::int(1/sample_rate)]#resampling    
        train_label = train_label[::int(1/sample_rate)]#resampling
        test_label = test_label[::int(1/sample_rate)]#resampling

        train_values = metro_train.iloc[:,2:].values
        test_values = metro_test.iloc[:,2:].values

        if args.scaler == 'quantile':
            from sklearn.preprocessing  import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform')
        if args.scaler =='standard':
            from sklearn.preprocessing  import StandardScaler
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing  import MinMaxScaler
            scaler = MinMaxScaler()
        
        train_values = scaler.fit_transform(train_values)
        test_values = scaler.transform(test_values) 


        #dump train values into file
        makedirs('datasets/data/processed', exist_ok=True)
        path_pkl = path.join('datasets/data/processed', 'METRO_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)


        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'METRO_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)


        #dump test labels into file
        path_pkl = path.join('datasets/data/processed', 'METRO_test_label.pkl')
        with open(path_pkl, 'wb') as file:
           dump(test_label, file)

    elif dataset == 'IVECO':
        import os
        dfs = []
        for filename in os.listdir('datasets/data/IVECO'):
            if filename.endswith('.csv'):
                file_path = path.join('datasets/data/IVECO', filename)

            df = pd.read_csv(file_path)
            dfs.append(df)

        iveco_all = pd.concat(dfs)
        iveco_all.reset_index(drop = True, inplace = True)

        iveco_all = iveco_all.iloc[:,5:] #esclusione colonne non informative

        if args.cut < 1:
            print('Cutting the dataset at ' + str(args.cut) + ' length \n')
            iveco_all = iveco_all.iloc[:int(len(swat)*args.cut)]
        sample_rate = args.resample_rate
        if sample_rate<=0 or sample_rate>1:
            print('Incorrect resample rate, defaulting to 1\n')
            sample_rate = 1
        else:
            print('resampling to one observation every '+ str(int(1/sample_rate)))

        iveco_all = iveco_all.iloc[::int(1/sample_rate)]#resampling
        #labels = (swat['Normal/Attack'].values=='Attack')

        #na handling
        thresh = 0.75
        na = []
        for i in range(len(iveco_all.columns)):
            na.append(sum(iveco_all.iloc[:,i].isna()))
        where = (np.array(na) < iveco_all.shape[0]*thresh)

        iveco_all = iveco_all.iloc[:, where]

        values = iveco_all.values
        
        
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

        train_values = values[:int(train_test_split*len(values)),:]
        #train_labels = labels[:int(train_test_split*len(labels))]

        # if args.no_anomaly_train:
        #     print('removing anomalies from training data')
        #     train_values = train_values[train_labels==False]

        test_values = values[int(train_test_split*len(values)):,:]
        #test_labels = labels[int(train_test_split*len(labels)):]

        #dump train values into file
        makedirs('datasets/data/processed', exist_ok=True)
        path_pkl = path.join('datasets/data/processed', 'IVECO_train.pkl')
        with open(path_pkl, 'wb') as file:
            dump(train_values, file)


        #dump test values into file
        path_pkl = path.join('datasets/data/processed', 'IVECO_test.pkl')
        with open(path_pkl, 'wb') as file:
            dump(test_values, file)

        #dump test labels into file (NB ALL FALSE FOR UNSUPERVISED DATA)
        path_pkl = path.join('datasets/data/processed', 'IVECO_test_label.pkl')
        test_labels = np.full(len(test_values), False, dtype=bool) #ALL FALSE VALUES FOR UNSUPERVISED DATA
        test_labels[-1]=True #for calculating-auc-roc
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
