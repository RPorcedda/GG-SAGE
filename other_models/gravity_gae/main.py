# Credit to: https://github.com/deezer/gravity_graph_autoencoders/tree/master

import os

import scipy.sparse as sp
import torch
from torch import tensor as tt
from torch.nn import  Sequential, ReLU, BCEWithLogitsLoss
from torch_geometric import seed_everything
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

import methods


def set_model_and_train(
    model_name,
    dataset,
    task,
    num_runs,
    lrscheduler,
    lr,
    num_epochs,
    val_loss_fn,
    early_stopping,
    use_sparse_representation
    ):


    model = None
    train_data, val_data, test_data = None, None, None

    methods.data_suggested_parameters_sets["general"]["general"]["dataset"]=dataset
    train_data, val_data, test_data = methods.load_general(**methods.data_suggested_parameters_sets["general"][task], device = device)

    input_dimension = train_data.x.shape[1]
    methods.models_suggested_parameters_sets["general"]["general"][model_name]["input_dimension"]=input_dimension

    if model_name == "gravity_gae":
        model = methods.get_gravity_gae(**methods.models_suggested_parameters_sets["general"][task][model_name]).to(device)
    elif model_name == "gravity_vgae":
        model = methods.get_gravity_vgae(**methods.models_suggested_parameters_sets["general"][task][model_name]).to(device)
    elif model_name == "sourcetarget_gae":
        model = methods.get_sourcetarget_gae(**methods.models_suggested_parameters_sets["general"][task][model_name]).to(device)
    elif model_name == "sourcetarget_vgae":
        model = methods.get_sourcetarget_vgae(**methods.models_suggested_parameters_sets["general"][task][model_name]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
    # Array of test set performances
    perfs = []
    # Array of training times
    training_times =[]  
    # Compute train and validation set loss normalization
    tot_train_edges = train_data.edge_label.size(0)
    tot_pos_train_edges = int(train_data.edge_label.sum())
    tot_neg_edges_train = tot_train_edges - tot_pos_train_edges
    pos_weight = torch.tensor(tot_neg_edges_train / tot_pos_train_edges) # 5
    norm = tot_train_edges / (2 * tot_neg_edges_train) 

    train_loss = None
    if model_name == "gravity_gae"    or  model_name == "sourcetarget_gae" :
        train_loss = methods.StandardLossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))
    elif (model_name == "gravity_vgae" or model_name == "sourcetarget_vgae") and task == "general" :
        train_loss = methods.VGAELossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))
    elif (model_name == "gravity_vgae" or model_name == "sourcetarget_vgae") and (task == "biased" or task == "biased_rev" or task == "bidirectional" ):
        train_loss = methods.VGAELossWrapper(1/2, methods.recon_loss)


    for i in range(num_runs):
        training_times.append(methods.train(train_data, model,  train_loss , optimizer, device, num_epochs, lrscheduler = lrscheduler, early_stopping = early_stopping, val_data = val_data,  val_loss_fn =  val_loss_fn, patience = 50, retrain_data = train_data, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10))
        perfs.append(methods.evaluate_link_prediction(model, test_data, None, device = device))

    # Print test-set performances
    mean_std_dict = methods.summarize_link_prediction_evaluation(perfs)
    # markdown_table = methods.pretty_print_link_performance_evaluation(mean_std_dict, model_name)
    # print(markdown_table)
    # return markdown_table
    return model_name, mean_std_dict, training_times


####################################################################################
model_dir = os.path.dirname(os.path.abspath(__file__))+"/trained_models"
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/Datasets"
accepted_datasets = [nome for nome in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, nome))]
if "__pycache__" in accepted_datasets:
    accepted_datasets.remove("__pycache__")

possible_models = ["gravity_gae", "gravity_vgae", "sourcetarget_gae", "sourcetarget_vgae"]

# Please set the parameters below
seed_everything(12345)             # Seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # One of `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` or `torch.device("cpu")`
num_runs = 5                        # Number of initial configuration to average over
lrscheduler = None                  # Learning rate scheduler
task = "general"

for dataset in accepted_datasets:
    for model_name in possible_models:
        # model_name = "gravity_vgae"        # Please specify what model you'd like to use. Must be one of {"gravity_gae", "gravity_vgae", "sourcetarget_gae", "sourcetarget_vgae"}
        # dataset = "cora"                    # Only "cora" is implemented right now
        # task    = "general"           # One of {"general", "biased", "biased_rev", "bidirectional"}
        # lr                        = methods.setup_suggested_parameters_sets[dataset][task][model_name]["lr"]
        # num_epochs                = methods.setup_suggested_parameters_sets[dataset][task][model_name]["num_epochs"]
        # val_loss_fn               = methods.setup_suggested_parameters_sets[dataset][task][model_name]["val_loss_fn"]                      # The validation loss for early stopping. Default is the sum of AUC and AP over the validation set.
        # early_stopping            = methods.setup_suggested_parameters_sets[dataset][task][model_name]["early_stopping"]                   # True or False
        # use_sparse_representation = methods.models_suggested_parameters_sets[dataset][task][model_name]["use_sparse_representation"]        # True or False
        lr                        = methods.setup_suggested_parameters_sets["general"]["general"][model_name]["lr"]
        num_epochs                = methods.setup_suggested_parameters_sets["general"]["general"][model_name]["num_epochs"]
        val_loss_fn               = methods.setup_suggested_parameters_sets["general"]["general"][model_name]["val_loss_fn"]                      # The validation loss for early stopping. Default is the sum of AUC and AP over the validation set.
        early_stopping            = methods.setup_suggested_parameters_sets["general"]["general"][model_name]["early_stopping"]                   # True or False
        use_sparse_representation = methods.models_suggested_parameters_sets["general"]["general"][model_name]["use_sparse_representation"]  

        try:
            # markdown_results = set_model_and_train(model_name, dataset, task, num_runs, lrscheduler, lr, num_epochs, val_loss_fn, early_stopping, use_sparse_representation)
            model_name, mean_std_dict, training_times = set_model_and_train(model_name, dataset, task, num_runs, lrscheduler, lr, num_epochs, val_loss_fn, early_stopping, use_sparse_representation)

            markdown_table = methods.pretty_print_link_performance_evaluation(mean_std_dict, model_name)
            print(markdown_table)

            # Write in text file 
            with open(f"{model_dir}/results.txt", 'a') as text_file:
                text_file.write(f"DATASET {dataset}:\n{markdown_table}\n\n")

            # Write in CSV file
            with open(f"{model_dir}/results.csv", mode='a', newline='') as csv_file:
                methods.link_performance_evaluation_to_csv_row(mean_std_dict, model_name, dataset, training_times, csv_file)

        except Exception as e:
            with open(f"{model_dir}/results.txt", 'a') as text_file:
                text_file.write(f"DATASET {dataset}:\nThere was an exception: {e}\n\n")
            with open(f"{model_dir}/results.csv", mode='a', newline='') as csv_file:
                methods.link_performance_evaluation_to_csv_row(None, model_name, dataset, None, csv_file)










































### Expected Results

## cora, general

#  gravity_gae
# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.882 +- 0.004 |
# | F1    | 0.728 +- 0.01  |
# | hitsk | 0.741 +- 0.006 |
# | AP    | 0.913 +- 0.003 |

# gravity_vgae
# |       | Gravity-VGAE   |
# |:------|:---------------|
# | AUC   | 0.901 +- 0.004 |
# | F1    | 0.742 +- 0.01  |
# | hitsk | 0.75 +- 0.02   |
# | AP    | 0.912 +- 0.007 |


# sourcetarget_gae
# |       | SourceTarget-GAE   |
# |:------|:-------------------|
# | AUC   | 0.872 +- 0.006     |
# | F1    | 0.48 +- 0.08       |
# | hitsk | 0.62 +- 0.02       |
# | AP    | 0.891 +- 0.005     |


# sourcetarget_vgae
# |       | SourceTarget-VGAE   |
# |:------|:--------------------|
# | AUC   | 0.88 +- 0.01        |
# | F1    | 0.45 +- 0.02        |
# | hitsk | 0.64 +- 0.02        |
# | AP    | 0.899 +- 0.007      |


## cora, biased

# gravity_gae
# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.82 +- 0.004  |
# | F1    | 0.725 +- 0.005 |
# | hitsk | 0.45 +- 0.01   |
# | AP    | 0.826 +- 0.002 |


# gravity_vgae
# |       | gravity_vgae     |
# |:------|:-----------------|
# | AUC   | 0.825 +- 0.002   |
# | F1    | 0.758 +- 0.002   |
# | hitsk | 0.447 +- 0.003   |
# | AP    | 0.8316 +- 0.0009 |

# sourcetarget_gae
# |       | SourceTarget-GAE   |
# |:------|:-------------------|
# | AUC   | 0.627 +- 0.006     |
# | F1    | 0.36 +- 0.07       |
# | hitsk | 0.2 +- 0.04        |
# | AP    | 0.65 +- 0.02       |


# sourcetarget_vgae
# |       | sourcetarget_vgae   |
# |:------|:--------------------|
# | AUC   | 0.633 +- 0.005      |
# | F1    | 0.54 +- 0.01        |
# | hitsk | 0.26 +- 0.01        |
# | AP    | 0.682 +- 0.009      |


## cora, biased rev

# gravity_gae
# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.82 +- 0.004  |
# | F1    | 0.725 +- 0.005 |
# | hitsk | 0.45 +- 0.01   |
# | AP    | 0.826 +- 0.002 |


# gravity_vgae
# |       | gravity_vgae     |
# |:------|:-----------------|
# | AUC   | 0.8359 +- 0.0009 |
# | F1    | 0.771 +- 0.004   |
# | hitsk | 0.438 +- 0.003   |
# | AP    | 0.8368 +- 0.0009 |

# sourcetarget_gae
# |       | SourceTarget-GAE   |
# |:------|:-------------------|
# | AUC   | 0.894 +- 0.005     |
# | F1    | 0.815 +- 0.009     |
# | hitsk | 0.6 +- 0.01        |
# | AP    | 0.885 +- 0.007     |

# sourcetarget_vgae
# |       | sourcetarget_vgae   |
# |:------|:--------------------|
# | AUC   | 0.895 +- 0.003      |
# | F1    | 0.819 +- 0.007      |
# | hitsk | 0.59 +- 0.01        |
# | AP    | 0.884 +- 0.004      |



## cora, bidirectional

# gravity_gae
# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.792 +- 0.006 |
# | F1    | 0.711 +- 0.007 |
# | hitsk | 0.62 +- 0.02   |
# | AP    | 0.798 +- 0.003 |


# gravity_vgae
# |       | gravity_vgae   |
# |:------|:---------------|
# | AUC   | 0.8 +- 0.01    |
# | F1    | 0.75 +- 0.01   |
# | hitsk | 0.63 +- 0.05   |
# | AP    | 0.774 +- 0.008 |

# sourcetarget_gae
# |       | SourceTarget-GAE   |
# |:------|:-------------------|
# | AUC   | 0.77 +- 0.02       |
# | F1    | 0.72 +- 0.01       |
# | hitsk | 0.63 +- 0.07       |
# | AP    | 0.76 +- 0.03       |

# sourcetarget_vgae
# |       | sourcetarget_vgae   |
# |:------|:--------------------|
# | AUC   | 0.819 +- 0.007      |
# | F1    | 0.74 +- 0.01        |
# | hitsk | 0.71 +- 0.02        |
# | AP    | 0.81 +- 0.02        |