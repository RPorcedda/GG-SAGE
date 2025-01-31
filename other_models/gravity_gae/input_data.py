import networkx as nx
import scipy.sparse as sp
import os
import torch
from typing import Optional, Tuple, Union, Type, Literal
import sklearn
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/Datasets"
accepted_datasets = [nome for nome in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, nome))]
if "__pycache__" in accepted_datasets:
    accepted_datasets.remove("__pycache__")


def reindex_edges(
    edge_index: torch.Tensor
    ) -> Tuple[Union[torch.Tensor,dict]]:
    """
    Remap edge indices.
    Params:
        edge_index: edge_index from pyG Data object
    Return:
        reindexed_edges: reindexed edge_index
        new_indices_map: mapping dictionary
    """
    all_nodes = edge_index.flatten() # Flatten the edge_index to 1D list of node indices
    # Get the unique nodes and create a mapping to new indices
    unique_connected_nodes = torch.unique(all_nodes, sorted=True).tolist()
    new_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_connected_nodes)}
    # Reindex edges
    for i, node_idx in enumerate(all_nodes):
        all_nodes[i] = new_indices_map[node_idx.item()]
    reindexed_edges = all_nodes.view(edge_index.size())
    
    return reindexed_edges, new_indices_map


def encode_string_nodeLabels(
    df: pd.DataFrame,
    node_lb:sklearn.preprocessing.LabelEncoder
    ) -> pd.DataFrame:
    """
    Applies Label Encoding on string attributes.
    Params:
        df:  DataFrame containing the edgelist
        node_lb: fitted LabelEncoder
    Return:
        df:  encoded node features DataFrame
    """
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    df[df.columns[1]] = df[df.columns[1]].astype(str)

    df[df.columns[0]] = node_lb.transform(df[df.columns[0]])
    df[df.columns[1]] = node_lb.transform(df[df.columns[1]])
    df[df.columns[0]] = df[df.columns[0]].astype(int)
    df[df.columns[1]] = df[df.columns[1]].astype(int)
    return df


def encode_string_columns(
    df: pd.DataFrame,
    nodeLabelEncoder: Optional[bool]=False,
    index: Optional[str]=None
    ):# -> Union[pd.DataFrame, Tuple[Union[[pd.DataFrame, sklearn.preprocessing.LabelEncoder]]]]:
    """
    OneHotEncoding for categorical features and remap node labels.
    Params:
        df:  DataFrame containing the node features
        nodeLabelEncoder: if True, labels are remapped and
        index: label column name
    Return:
        df:  remapped DataFrame of node features
        node_lb: fitted LabelEncoder
    """
    if nodeLabelEncoder:
        df[index] = df[index].astype(str)
        node_lb = LabelEncoder()
        node_lb.fit(df[index])
        df[index] = node_lb.transform(df[index])
    for i, col in enumerate(df.columns):
        if index is not None and col==index: # Leave node labels
            continue
        if df[col].dtype=="object":
            df[col] = df[col].astype(str)
        if pd.api.types.is_string_dtype(df[col]):
            lb = LabelBinarizer()
            lb_results = lb.fit_transform(df[col])

            lb_results_df = pd.DataFrame(lb_results,
                                        columns=[f"{col}_{cls}" for cls in lb.classes_], index=df.index)

            df = pd.concat([df, lb_results_df], axis=1)
            df.drop(columns=[col], inplace=True)

    if not nodeLabelEncoder:
        df["COPIED_INDEX"]=df.index

    if nodeLabelEncoder:
        return df, node_lb
    else:
        return df


def find_node_label_column(
    node_features_df: pd.DataFrame,
    edgelist_df: pd.DataFrame
    ) -> str:
    """
    This function checks if any column in the node features DataFrame contains labels that match exactly 
    with the node labels present in the edgelist. If such a column is found, it returns the name 
    of the column; otherwise, it returns None.
    
    Parameters:
    node_features_df: The DataFrame containing node features.
    edgelist_df: The DataFrame containing edges with node labels/indices.
    
    Returns:
    column: The name of the column that matches the node labels in the edgelist, or None if no such column exists.
    """
    # Get the set of unique node labels from the edgelist (assuming columns 'source' and 'target' in edgelist)
    edgelist_labels = set(edgelist_df['source']).union(set(edgelist_df['target']))
    # Iterate through each column in the node features DataFrame
    for column in node_features_df.columns:
        # Get the values of the column and check if they are unique
        node_feature_labels = node_features_df[column]
        if node_feature_labels.is_unique:
            print(f"Column {column} has unique values")
            # Convert the unique values of the column to a set
            node_feature_label_set = set(node_feature_labels)
            # Check if the unique values of the column match the labels in the edgelist
            if (
                edgelist_labels.issubset(node_feature_label_set) or
                node_feature_label_set.issubset(edgelist_labels) or
                len(edgelist_labels.intersection(node_feature_label_set))/len(edgelist_labels)>0.7 or
                len(edgelist_labels.intersection(node_feature_label_set))/len(node_feature_label_set)>0.7
            ):
                return column  # Return the column name if there's a match

    return None  # Return None if no matching column is found



def clean_edgelist(
    edgelist: pd.DataFrame,
    node_data: pd.DataFrame,
    index: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Remove rows from the edge list where either the source or target node is not present in node_data.
    
    Params:
        edgelist: DataFrame containing the edge list with columns ["source", "target"]
        node_data: DataFrame containing node features, indexed by node IDs
        
    Returns:
        cleaned_edgelist: DataFrame containing only edges where both nodes exist in node_data
    """
    # Get the set of valid nodes from node_data
    if index is not None:
        valid_nodes = set(node_data[index])
    else:
        valid_nodes = set(node_data.index)
    
    # Filter edgelist to keep only rows where both source and target are in the valid nodes
    cleaned_edgelist = edgelist[edgelist['source'].isin(valid_nodes) & edgelist['target'].isin(valid_nodes)] #.reset_index(drop=True)
    
    return cleaned_edgelist


def clean_node_data(
    node_data: pd.DataFrame,
    edgelist: pd.DataFrame,
    index: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Remove rows from the node_data DataFrame where node label is not present 
    in the edgelist as either a source or target.
    
    Params:
        node_data: DataFrame containing node features
        edgelist: DataFrame containing the edge list with columns ["source", "target"]
        
    Returns:
        cleaned_node_data: DataFrame containing only rows where the node ID exists in the edgelist
    """
    nodes_in_edgelist = set(edgelist['source']).union(set(edgelist['target']))

    if index is not None:
        cleaned_node_data = node_data[node_data[index].isin(nodes_in_edgelist)] #.reset_index(drop=True)
    else:
        cleaned_node_data = node_data[node_data.index.isin(nodes_in_edgelist)] #.reset_index(drop=True)

    return cleaned_node_data


def standardize_non_binary_columns(node_data):
    """
    Apply StandardScaler on non-binary features
    
    Params:
        node_data: DataFrame containing node features
        
    Returns:
        standardized_node_data: DataFrame containing standardized columns
    """
    scaler = scaler = StandardScaler()
    standardized_node_data = node_data.copy()
    
    # Iterate through the columns
    for column in node_data.columns:
        # Check if the column is binary (contains only 0s and 1s)
        if node_data[column].nunique() > 2:  # Column has more than 2 unique values, so it's not binary
            # Normalize the column using Min-Max scaling
            standardized_node_data[column] = scaler.fit_transform(node_data[[column]])
            # Transform nans into zeros
            # standardized_node_data[column][np.isnan(standardized_node_data[column])]=0 
            standardized_node_data.loc[np.isnan(standardized_node_data[column]), column] = 0  
    
    return standardized_node_data



def load_data(
    dataset: str = "cora",
    reindex: Optional[bool]=True
    ):
    """
    Preprocess dataset to obtain a pyG Data object with features and edge indices
    Params:
        dataset: name of input dataset
        reindex: if True, nodes and edges are reindexed
    Return:
        directed_data: directed graph object
    """
    #accepted_datasets = ["cora"]
    if dataset not in accepted_datasets:
        raise Exception(f"{dataset} is not present in the accepted datasets")

    print()
    print(f"Preprocessing {dataset} dataset")

    directory = data_dir+f'/{dataset}'

    # Read edgelist
    edgelist = pd.read_csv(os.path.join(directory,
                                        f"{dataset}.edges"),
                                        sep='\t',
                                        header=None,
                                        names=["target", "source"])
    
    # Read features and encode string columns
    node_data = pd.read_csv(os.path.join(directory,
                                        f"{dataset}.node_feats"),
                                        sep='\t',
                                        header=None)

    label_col = find_node_label_column(node_data, edgelist)
    edgelist = clean_edgelist(edgelist, node_data, index=label_col)
    node_data = clean_node_data(node_data, edgelist, index=label_col)
    
    if label_col is not None:
        node_data, nodeLabelEncoder = encode_string_columns(node_data, nodeLabelEncoder=True, index=label_col)
        edgelist = encode_string_nodeLabels(edgelist, nodeLabelEncoder)
    else:
        node_data = encode_string_columns(node_data)

    # Convert the edge list DataFrame to a PyTorch tensor
    edge_index = torch.tensor(edgelist.values,
                                dtype=torch.long).t().contiguous()

    if reindex:
        edge_index, new_indices_map = reindex_edges(edge_index)
        if label_col is not None:
            node_data[label_col] = node_data[label_col].map(new_indices_map)
            node_data = node_data.set_index(label_col).sort_index()
            node_data.drop(columns=[label_col], inplace=True)
        else:
            node_data["COPIED_INDEX"] = node_data["COPIED_INDEX"].map(new_indices_map)
            node_data = node_data.set_index(node_data["COPIED_INDEX"]).sort_index()
            node_data.drop(columns=["COPIED_INDEX"], inplace=True)

    # Standardize non-binary columns
    node_data = standardize_non_binary_columns(node_data)

    px = pd.DataFrame(edge_index.T.numpy(), columns=["source","target"])
    adj = nx.adjacency_matrix(nx.from_pandas_edgelist(px, create_using=nx.DiGraph()))
    features = torch.tensor(node_data.values, dtype = torch.float32)
    # features = torch.eye(adj.shape[0])

    print(f"{dataset} dataset preprocessed")
    print(f'Number of nodes: {adj.shape[0]}')
    print(f'Number of edges: {len(px['source'])}')
    print(f'Number of features: {features.shape[1]}')
    print()

    return adj, features