import sys
import os
import torch
from .preprocessing import get_pyg_dataset
from .utils import train_model_with_resampling, link_performance_evaluation_to_csv_row

model_dir = os.path.dirname(os.path.abspath(__file__))+"/trained_models"
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Datasets"

dataset = [nome for nome in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, nome))]


if "__pycache__" in dataset:
    dataset.remove("__pycache__")


print("Training these datasets: ", dataset)

args = {
    "model_name": "GGSAGE",
    "hidden_channels": 64,
    "out_channels": 64,
    "epochs": 200,
    "lr": 0.001,
    "weight_decay": 0.,
    "k": 5,
    "optimizer_class": torch.optim.Adam,
    "test_ratio": 0.1,
    "train_ratio": 0.85,
    "val_ratio": 0.05,
    "test_neg_ratio": None,
    "train_val_neg_ratio": None,
    "overall": False,
    "evaluation_batch_size": 64,
    "verbose": 3
}


for item in dataset:

    data = get_pyg_dataset(item, features_flag=True)
    args["data"]=data

    try:
        model, results, perfs_dict, training_times = train_model_with_resampling(**args)

        # Write in text file 
        with open(f"{model_dir}/results.txt", 'a') as text_file:
            text_file.write(f"DATASET {item}:\n{perfs_dict}\n\n")

        # Write in CSV file
        with open(f"{model_dir}/results.csv", mode='a', newline='') as csv_file:
            link_performance_evaluation_to_csv_row(perfs_dict, item, training_times, csv_file)

    except Exception as e:
        with open(f"{model_dir}/results.txt", 'a') as text_file:
            text_file.write(f"DATASET {item}:\nThere was an exception: {e}\n\n")
        with open(f"{model_dir}/results.csv", mode='a', newline='') as csv_file:
            link_performance_evaluation_to_csv_row(None, item, None, csv_file)