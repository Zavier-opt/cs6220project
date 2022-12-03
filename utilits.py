from torch.utils.data import DataLoader, Subset, random_split, Dataset
import pandas as pd
import numpy as np
import torch as th


class MyDataset(Dataset):
    def __init__(self, data, label, **hyperparameters):
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


def set_path(experiment, num_of_split, sub_id,isRandom):
    data_path, test_path = "", ""
    if experiment == "age":
        if not isRandom:
            data_path = "data/" + experiment + "/" + "train_age_" + str(sub_id) + ".csv"
            test_path = "data/test.csv"
        else:
            data_path = "data/" + experiment+"_random" + "/" + "train_age_" + str(sub_id) + ".csv"
            test_path = "data/test.csv"
    if experiment == "occupation":
        if not isRandom:
            data_path = "data/" + experiment + "/" + "occupation_train_" + str(sub_id) + ".csv"
            test_path = "data/test.csv"
        else:
            data_path = "data/" + experiment+"_random" + "/" + "occupation_r_train_" + str(sub_id) + ".csv"
            test_path = "data/test.csv"
    if experiment == "user":
        data_path = "data/"+experiment+"/train_user"+str(num_of_split)+"_"+str(sub_id)+".csv"
        test_path = "data/test.csv"
    if experiment == "movie":
        data_path = "data/"+experiment+"/train_movie"+str(num_of_split)+"_"+str(sub_id)+".csv"
        test_path = "data/test.csv"
    return data_path, test_path
