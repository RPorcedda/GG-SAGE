import torch
from typing import Optional, Tuple, Union, Type, Literal
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from .models import GGSAGE
import time
import csv

def extract_test_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    test_ratio: Optional[float] = 0.1,
    neg_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor]:
    """
    Extract edges for test set and prepare a set to be later splitted in training and validation.
    Params:
        edge_index: reindexed edge_index
        num_nodes: number of nodes in graph
        test_ratio: ratio of positive edges in test set
        neg_ratio: ratio of negative edges to sample from total negative edges.
            Default is None, meaning that the number of negative edges sampled will be
            approximately equal to the number of positive edges in test set
    Return:
        train_val_edge_index: positive edge indices left for training and validation splitting
        test_edge_index: positive test edge indices
        train_val_edge_index_neg: negative edge indices left for training and validation splitting
        test_edge_index_neg: negative test edge indices
    """

    # Number of positive edges (total, train_val set, test set)
    num_edges = edge_index.size(1)
    num_train_val = int(num_edges * (1-test_ratio))
    num_test = int(num_edges * test_ratio)

    all_indices = torch.randperm(num_edges) # Shuffle edges
    # Here we are not actually shuffling the edge_index tensor (in which each entry is a pair
    # of node indices), but we are creating a random permutation of a tensor containing
    # integer values from 0 to num_edges

    # Slice positive edges (train and val sets, test set)
    train_val_edge_indices = all_indices[:num_train_val]
    test_edge_indices = all_indices[num_train_val:num_train_val + num_test]
    
    # Obtain the actual edge indices for train_val set and test set
    train_val_edge_index = edge_index[:, train_val_edge_indices]
    test_edge_index = edge_index[:, test_edge_indices]
    
    # Negative edges
    num_all_edges_neg = num_nodes**2 - num_nodes - num_edges # Total of negative edges
    all_edges_neg = negative_sampling(edge_index=edge_index,
                                        num_nodes=num_nodes,
                                        num_neg_samples=num_all_edges_neg)
    if neg_ratio is None:
        test_edge_index_neg = all_edges_neg[:,:num_test]
        train_val_edge_index_neg = all_edges_neg[:,num_test:]
    else:
        neg_num_test = int(num_all_edges_neg*test_ratio*neg_ratio)
    
        test_edge_index_neg = all_edges_neg[:,:neg_num_test]
        train_val_edge_index_neg = all_edges_neg[:,neg_num_test:]

    return train_val_edge_index, test_edge_index, train_val_edge_index_neg, test_edge_index_neg



def train_val_split(
    train_val_edge_index: torch.Tensor,
    train_val_edge_index_neg: torch.Tensor,
    neg_ratio: Optional[float] = None, # Prima si chiamava "m"
    val_ratio: Optional[float] = 0.05,
    test_ratio: Optional[float] = 0.1
    ) -> Tuple[torch.Tensor]:
    """
    Split training and validation set from train_val indices obtained in extract_test_edges().
    Params:
        train_val_edge_index: indices of positive train_val edges
        train_val_edge_index_neg: indices of negative train_val edges
        neg_ratio: ratio of negative edges to sample from total negative edges
        val_ratio: Optional[float] = 0.05,
        test_ratio: ratio of edges in test set
    Return:
        train_edge_index: positive training edge indices
        val_edge_index: positive validation edge indices
        train_edge_index_neg: negative training edge indices
        val_edge_index_neg: negative training edge indices
    """

    train_ratio = 1 - test_ratio - val_ratio
    new_train_ratio = train_ratio/(train_ratio+val_ratio) # Which ratio of train_val set
    # is for training
    new_val_ratio = val_ratio/(train_ratio+val_ratio) # Which ratio of train_val set
    # is for validation
    num_edges = train_val_edge_index.size(1)

    # From now on, it is basically the same procedure adopted in extract_test_edges()
    all_indices = torch.randperm(num_edges)
    num_train = int(num_edges*new_train_ratio)
    num_val = int(num_edges*new_val_ratio)
    if num_val==0:
        num_val=1
        num_train-=1
    train_edge_indices = all_indices[:num_train]
    val_edge_indices = all_indices[num_train:num_train + num_val]
    train_edge_index = train_val_edge_index[:,train_edge_indices]
    val_edge_index = train_val_edge_index[:,val_edge_indices]

    num_edges_neg = train_val_edge_index_neg.size(1)
    all_indices_neg = torch.randperm(num_edges_neg)

    if neg_ratio is None:
        num_train_neg = num_train
        num_val_neg = num_val
    else:
        num_train_neg = int(num_edges_neg*new_train_ratio*neg_ratio)
        num_val_neg = int(num_edges_neg*new_val_ratio*neg_ratio)

    train_edge_indices_neg = all_indices_neg[:num_train_neg]
    val_edge_indices_neg = all_indices_neg[num_train_neg:num_train_neg + num_val_neg]
    train_edge_index_neg = train_val_edge_index_neg[:,train_edge_indices_neg]
    val_edge_index_neg = train_val_edge_index_neg[:,val_edge_indices_neg]

    return train_edge_index, val_edge_index, train_edge_index_neg, val_edge_index_neg
    


def train_and_evaluate(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mode: Optional[bool] = True,
    test_mode: Optional[bool] = False
    ) -> Tuple[Union[float, np.ndarray]]:
    """
    Performs one epoch of training with backpropagation or just evaluation
    Params:
        model: model object,
        data: directed graph PyG object
        optimizer: optimizer for training
        train_mode: whether you are training or evaluating the model
    Return:
        loss: loss computed with model training loss function
        total_recon_error: reconstruction error
            (returned only if model includes anomaly detection)
        auc: Area Under Curve of ROC
        ap: Average Precision
    """

    if train_mode:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train_mode):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        pos_edge_index, neg_edge_index = data.edge_index, data.edge_index_neg

        pos_results = model.predict_links(z, pos_edge_index)
        neg_results = model.predict_links(z, neg_edge_index)

        if model.name in ["GGSAGE"]:
            losses = model.loss_function(pos_results, neg_results)
            pos_pred = pos_results
            neg_pred = neg_results
            loss = losses

        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([
                torch.ones(pos_pred.size(0)),
                torch.zeros(neg_pred.size(0))
            ], dim=0)
        detached_labels = labels.cpu().numpy()
        detached_preds = preds.cpu().detach().numpy()

        auc = roc_auc_score(detached_labels, detached_preds)
        ap = average_precision_score(detached_labels, detached_preds)

        if train_mode:
            loss.backward()
            optimizer.step()

    if model.name in ["GGSAGE"]:
        return loss.item(), auc, ap



def evaluate_in_batches(
    model: torch.nn.Module,
    data: Data,
    batch_size: Optional[int]=64
    ) -> Tuple[Union[float, np.ndarray]]:
    """
    Performs one epoch of training with backpropagation or just evaluation
    Params:
        model: model object
        data: directed graph PyG object
        batch_size: 
    Return:
        loss: loss computed with model training loss function
        total_recon_error: reconstruction error
            (returned only if model includes anomaly detection)
        auc: Area Under Curve of ROC
        ap: Average Precision
    """
    #device = data.x.device
    # Get node embeddings
    z = model(data.x, data.edge_index)
    # Get positive and negative edges
    pos_edge_index = data.edge_index
    num_pos_samples = data.edge_index.size(1)
    num_neg_samples = int((data.num_nodes**2 - num_pos_samples))
    neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                        num_nodes=data.num_nodes,
                                        num_neg_samples=num_neg_samples)
    pos_preds = []
    neg_preds = []
    # Create data loaders for batching
    pos_dataset = TensorDataset(pos_edge_index.t())
    neg_dataset = TensorDataset(neg_edge_index.t())
    pos_loader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=False)
    neg_loader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        # Positive Edges Batch Evaluation
        for pos_batch in tqdm(pos_loader, desc="Positive Edges Batch Evaluation"):
            pos_indices = pos_batch[0].t()
            pos_results = model.predict_links(z, pos_indices)
            if model.name=="GGSAGE":
                pos_pred = pos_results
            pos_preds.append(pos_pred)
        # Negative Edges Batch Evaluation
        for neg_batch in tqdm(neg_loader, desc="Negative Edges Batch Evaluation"):
            neg_indices = neg_batch[0].t()
            neg_results= model.predict_links(z, neg_indices)
            if model.name=="GGSAGE":
                neg_pred = neg_results
            neg_preds.append(neg_pred)

    pos_preds = torch.cat(pos_preds, dim=0)
    neg_preds = torch.cat(neg_preds, dim=0)
    
    all_probs = torch.cat([pos_preds, neg_preds], dim=0)
    all_labels = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))], dim=0)

    auc_score = roc_auc_score(all_labels.cpu().numpy(), all_probs.cpu().numpy())
    ap_score = average_precision_score(all_labels.cpu().numpy(), all_probs.cpu().numpy())
    conf_mat = confusion_matrix(all_labels.cpu().numpy(), all_probs.cpu().numpy() > 0.5)

    return auc_score, ap_score, conf_mat



def train_model_with_resampling(
    data: Data,
    model_name: Literal["GGSAGE"],
    hidden_channels: Optional[int] = 64,
    out_channels: Optional[int] = 64,
    epochs: Optional[int] = 100,
    lr: Optional[float] = 0.001,
    weight_decay: Optional[float] = None,
    k: Optional[int] = 5,
    optimizer_class: Optional[torch.optim.Optimizer] = torch.optim.Adam,
    test_ratio: Optional[float] = 0.1,
    train_ratio: Optional[float] = 0.85,
    val_ratio: Optional[float] = 0.05,
    test_neg_ratio: Optional[float] = None,
    train_val_neg_ratio: Optional[float] = None,
    overall: Optional[bool] = False,
    evaluation_batch_size: Optional[float] = 64,
    verbose: Optional[int] = 3,
    early_stopping: Optional[bool] = True,
    patience: Optional[int] = 50,
    retrain: Optional[bool] = True,
) -> Tuple[Union[torch.nn.Module, dict]]:
    """
    Performs training with resampling and Early Stopping
    """
    print()
    print(f"Training {model_name} with resampling")
    print()

    results = {
        "train_loss": [],
        "train_auc": [],
        "train_ap": [],
        "val_loss": [],
        "val_auc": [],
        "val_ap": [],
    }

    final_test_auc = []
    final_test_ap = []
    training_times = [] 

    # First, extract fixed test set
    (
        train_val_edge_index,
        test_edge_index,
        train_val_edge_index_neg,
        test_edge_index_neg,
    ) = extract_test_edges(
        edge_index=data.edge_index, num_nodes=data.num_nodes, test_ratio=test_ratio, neg_ratio=test_neg_ratio
    )

    for fold in range(k):
        # Then, in each fold, split train and val set
        (
            train_edge_index,
            val_edge_index,
            train_edge_index_neg,
            val_edge_index_neg,
        ) = train_val_split(
            train_val_edge_index=train_val_edge_index,
            train_val_edge_index_neg=train_val_edge_index_neg,
            neg_ratio=train_val_neg_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Create train, val, and test data objects
        train_data = Data(x=data.x, edge_index=train_edge_index, edge_index_neg=train_edge_index_neg, y=data.y)

        val_data = Data(x=data.x, edge_index=val_edge_index, edge_index_neg=val_edge_index_neg, y=data.y)

        test_data = Data(x=data.x, edge_index=test_edge_index, edge_index_neg=test_edge_index_neg, y=data.y)

        if model_name == "GGSAGE":
            model = GGSAGE(
                in_channels=data.num_node_features, hidden_channels=hidden_channels, out_channels=out_channels
            )

        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Early Stopping variables
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        # Save initial states for retraining
        initial_model_state = model.state_dict()
        initial_optimizer_state = optimizer.state_dict()

        for epoch in range(epochs):
            train_results = train_and_evaluate(
                model=model,
                data=train_data,
                optimizer=optimizer,
                train_mode=True,
            )
            val_results = train_and_evaluate(
                model=model,
                data=val_data,
                optimizer=optimizer,
                train_mode=False,
            )


            train_loss, train_auc, train_ap = train_results
            val_loss, val_auc, val_ap = val_results

            results["train_loss"].append(train_loss)
            results["train_auc"].append(train_auc)
            results["train_ap"].append(train_ap)
            results["val_loss"].append(val_loss)
            results["val_auc"].append(val_auc)
            results["val_ap"].append(val_ap)

            if verbose == 3:
                if epoch%10==0:
                    print(
                        f"Epoch {epoch + 1}: Fold {fold + 1}, "
                        f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}"
                    )   

            # Early Stopping logic
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}"
                    )
                    break

        # Retrain model on the best epoch count
        if early_stopping and retrain:
            print(f"Retraining for {best_epoch} epochs...")
            model.load_state_dict(initial_model_state)
            optimizer.load_state_dict(initial_optimizer_state)
            start = time.time()
            for _ in range(best_epoch):
                train_and_evaluate(
                    model=model,
                    data=train_data,
                    optimizer=optimizer,
                    train_mode=True,
                )
            end = time.time()
            training_times.append(end-start)

        # Test set evaluation
        test_results = train_and_evaluate(
            model=model,
            data=test_data,
            optimizer=optimizer,
            train_mode=False,
            test_mode=True,
        )
        
        _, test_auc, test_ap = test_results

        final_test_auc.append(test_auc)
        final_test_ap.append(test_ap)

        if verbose >= 2:
            print(f"Fold {fold + 1} Test AUC: {test_auc:.4f}")
            print(f"Fold {fold + 1} Test AP: {test_ap:.4f}")

    perfs_dict ={
        "AUC": {
            "mean": np.mean(final_test_auc),
            "std": np.std(final_test_auc),
        },
        "AP":{
            "mean": np.mean(final_test_ap),
            "std": np.std(final_test_ap),
        }
    } 

    if verbose >= 1:
        print(f"Average Test AUC: {perfs_dict["AUC"]["mean"]:.4f} +- {perfs_dict["AUC"]["std"]:.4f}")
        print(f"Average Test AP: {perfs_dict["AP"]["mean"]:.4f} +- {perfs_dict["AP"]["std"]:.4f}")
        print(f"Training Time: {np.mean(training_times):.4f} +- {np.std(training_times):.4f}")

    if overall:
        with torch.no_grad():
            torch.cuda.empty_cache()
            overall_auc, overall_ap, overall_conf_mat = evaluate_in_batches(model, data)
        print("AUC on the entire graph:", overall_auc)
        print("AP on the entire graph:", overall_ap)
        print("Confusion matrix:\n", overall_conf_mat)

    return model, results, perfs_dict, training_times



def link_performance_evaluation_to_csv_row(mean_std_dict, dataset, training_times, csv_file):
    # Define the header for the CSV file
    header = ['Dataset', 'AUC_mean', 'AUC_std', 'AP_mean', 'AP_std', 'Train_time_mean', 'Train_time_std']

    writer = csv.writer(csv_file)
    # If the file is empty, write the header
    if csv_file.tell() == 0:
        writer.writerow(header)
    if mean_std_dict is not None:
        # Write dataset and results into the CSV
        writer.writerow([dataset,
                            mean_std_dict['AUC']['mean'],
                            mean_std_dict['AUC']['std'],
                            mean_std_dict['AP']['mean'],
                            mean_std_dict['AP']['std'],
                            np.mean(training_times),
                            np.std(training_times)
                            ])
    else:
        # If an exception occurs, record None
        writer.writerow([dataset,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None
                            ])