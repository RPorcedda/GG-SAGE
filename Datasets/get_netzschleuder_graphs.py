# Credit to https://github.com/RPorcedda/EasyNetz

from typing import Optional
import graph_tool.all as gt
import time
import pickle
import pandas as pd
import os
import numpy as np

args={
    "directed": True,
    "also_undirected": False,
    "bipartite": False,
    "also_unipartite": False,
    "weighted": True,
    "also_unweighted": True,
    "node_props": True,
    "also_no_node_props": False,
    "min_num_nodes": 1,
    "max_num_nodes": 100000, #np.inf,
    "min_num_edges": 0,
    "max_num_edges": np.inf,
    "min_density": 0,
    "max_density": 0.5, #1,
    "take_max": True,
    "take_min": False,
    "ns_info_saved": True,
    "access_token": None
}

props_to_remove = ["name", "label", "nodeLabel"]


def check_properties(
    vp: list,
    value: dict,
    tags: list,
    directed: Optional[bool]=True,
    also_undirected: Optional[bool]=False,
    bipartite: Optional[bool]=False,
    also_unipartite: Optional[bool]=False,
    weighted: Optional[bool]=True,
    also_unweighted: Optional[bool]=True,
    node_props: Optional[bool]=True,
    also_no_node_props: Optional[bool]=False,
    min_num_nodes: Optional[int]=0,
    max_num_nodes: Optional[int]=np.inf,
    min_num_edges: Optional[int]=0,
    max_num_edges: Optional[int]=np.inf,
    min_density: Optional[float]=0,
    max_density: Optional[float]=1
    ) -> bool:
    # Clean properties (eliminate names, labels and class properties)
    vp = [prop for prop in vp if prop[0] not in props_to_remove and not prop[0].startswith("_")]

    v = value["num_vertices"]
    e = value["num_edges"]

    minmax_nodes_check = (min_num_nodes <= v and max_num_nodes >= v)
    minmax_edges_check = (min_num_edges <= e and max_num_edges >= e)

    if v==1:
        density=0
    else:
        density = e/(v*(v-1)) # density in undirected graphs
    if not value.get('is_directed', False):
        density *= 2 # correction for directed graphs

    density_check = (min_density <= density and max_density >= density)

    if directed:
        directed_check = also_undirected or value.get('is_directed', False)
    else:
        directed_check = not value.get('is_directed', False)

    if bipartite:
        bipartite_check = also_unipartite or value.get('is_bipartite', False)
    else:
        bipartite_check = not value.get('is_bipartite', False)

    if weighted:
        weighted_check = also_unweighted or "Weighted" in tags
    else:
        weighted_check = not "Weighted" in tags

    if node_props:
        node_props_check = also_no_node_props or vp!=[]
    else:
        node_props_check = vp==[]
    
    overall_check = (density_check and
                        minmax_nodes_check and
                        minmax_edges_check and
                        directed_check and
                        bipartite_check and
                        weighted_check and
                        node_props_check)

    return overall_check



def filter_graph_names(
    directed: Optional[bool]=True,
    also_undirected: Optional[bool]=False,
    bipartite: Optional[bool]=False,
    also_unipartite: Optional[bool]=False,
    weighted: Optional[bool]=True,
    also_unweighted: Optional[bool]=True,
    node_props: Optional[bool]=True,
    also_no_node_props: Optional[bool]=False,
    min_num_nodes: Optional[int]=0,
    max_num_nodes: Optional[int]=np.inf,
    min_num_edges: Optional[int]=0,
    max_num_edges: Optional[int]=np.inf,
    min_density: Optional[float]=0,
    max_density: Optional[float]=1,
    take_max: Optional[bool]=True,
    take_min: Optional[bool]=False,
    ns_info_saved: Optional[bool]=True,
    access_token: Optional[str]=None
    ) -> list:

    if ns_info_saved:
        print("\nOpening Netzschleuder Info file...\n")
        with open('Datasets/ns_info.pkl', 'rb') as f:
            ns_info = pickle.load(f)
    else:
        print("\nConnecting to Netzschleuder...\n")
        ns_info = gt.collection.ns_info

    filtered_keys = []

    print("Start filtering graphs based on specified properties\n")
    for key, value in ns_info.items():
        # Leave out restricted datasets
        if value['restricted'] and access_token is None:
            print(f"ACCESS TO {key} IS RESTRICTED. Access Token must be provided")
            continue

        an = value['analyses']
        tags = value['tags']

        if 'vertex_properties' not in an.keys(): # More graphs contained in analyses
            max_limit = 0
            min_limit = np.inf
            max_key=None
            min_key=None
            keys_to_check = []
            items_list = list(an.items())
            total_items = len(an.items())

            for idx, (key2, value2) in enumerate(items_list):
                if take_max or take_min:
                    if take_max:
                        if value2['num_vertices'] > max_limit:
                            max_limit = value2['num_vertices']
                            max_key = key2

                    if take_min:
                        if value2['num_vertices'] < min_limit:
                            min_limit = value2['num_vertices']
                            min_key = key2

                else:
                    # If neither take_max nor take_min, process each item directly
                    vp = value2['vertex_properties']
                    if check_properties(vp=vp,
                                        value=value2,
                                        tags=tags,
                                        directed=directed,
                                        also_undirected=also_undirected,
                                        bipartite=bipartite,
                                        also_unipartite=also_unipartite,
                                        weighted=weighted,
                                        also_unweighted=also_unweighted,
                                        node_props=node_props,
                                        also_no_node_props=also_no_node_props,
                                        min_num_nodes=min_num_nodes,
                                        max_num_nodes=max_num_nodes,
                                        min_num_edges=min_num_edges,
                                        max_num_edges=max_num_edges,
                                        min_density=min_density,
                                        max_density=max_density):
                        print(f"Graph selected: {key}/{key2}")
                        filtered_keys.append(f"{key}/{key2}")

            # After iterating, append the max or min key if applicable
            if take_max and max_key is not None:
                keys_to_check.append(max_key)

            if take_min and min_key is not None:
                keys_to_check.append(min_key)

            # Process the keys_to_check if take_max or take_min was True
            for key2 in keys_to_check:
                value2 = an[key2]
                vp = value2['vertex_properties']
                if check_properties(vp=vp,
                                    value=value2,
                                    tags=tags,
                                    directed=directed,
                                    also_undirected=also_undirected,
                                    bipartite=bipartite,
                                    also_unipartite=also_unipartite,
                                    weighted=weighted,
                                    also_unweighted=also_unweighted,
                                    node_props=node_props,
                                    also_no_node_props=also_no_node_props,
                                    min_num_nodes=min_num_nodes,
                                    max_num_nodes=max_num_nodes,
                                    min_num_edges=min_num_edges,
                                    max_num_edges=max_num_edges,
                                    min_density=min_density,
                                    max_density=max_density):
                    print(f"Graph selected: {key}/{key2}")
                    filtered_keys.append(f"{key}/{key2}")
                
        else:
            vp = an['vertex_properties']
            if check_properties(vp=vp,
                                value=an,
                                tags=tags,
                                directed=directed,
                                also_undirected=also_undirected,
                                bipartite=bipartite,
                                also_unipartite=also_unipartite,
                                weighted=weighted,
                                also_unweighted=also_unweighted,
                                node_props=node_props,
                                also_no_node_props=also_no_node_props,
                                min_num_nodes=min_num_nodes,
                                max_num_nodes=max_num_nodes,
                                min_num_edges=min_num_edges,
                                max_num_edges=max_num_edges,
                                min_density=min_density,
                                max_density=max_density):
                print(f"Graph selected: {key}")
                filtered_keys.append(key)

    print(f"\nNumber of graphs found is {len(filtered_keys)}\n")
    return filtered_keys



def get_save_graphs(graph_names: list):
    #graph_names = filter_graph_names(**args)

    for name in graph_names:
        retry=0
        success=False
        print(f"Collecting and saving {name}...\n")
        while not success and retry<3:
            try:
                name_to_save = name.split("/")[0]

                g = gt.collection.ns[name]

                # Get node properties df
                node_feats_df = pd.DataFrame()
                for key, value in g.vp.items():
                    if key not in props_to_remove and not key.startswith("_"):
                        if value.value_type() in ["int16_t", "int32_t", "int64_t", "double", "long double", "bool"]:
                            node_feats_df[key]=value.get_array()
                        elif value.value_type()=="string":
                            node_feats_df[key]=list(value)
                        else:
                            for i, column in enumerate(value.get_2d_array()):
                                node_feats_df[f"{key}_{i}"]= column
                # Get edges df
                edges_df = pd.DataFrame(g.get_edges())

                # Get node properties df
                edge_feats_df = pd.DataFrame()
                for key, value in g.ep.items():
                    if value.value_type() in ["int16_t", "int32_t", "int64_t", "double", "long double", "bool"]:
                        edge_feats_df[key]=value.get_array()
                    elif value.value_type()=="string":
                        edge_feats_df[key]=list(value)
                    else:
                        for i, column in enumerate(value.get_2d_array()):
                            edge_feats_df[f"{key}_{i}"]= column


                outdir = f"./Datasets/netzschleuder_{name_to_save}"
                outdir_name = f"{outdir}/netzschleuder_{name_to_save}"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                if not node_feats_df.empty:
                    node_feats_df.to_csv(f"{outdir_name}.node_feats", sep='\t', index=False, header=False)

                if not edge_feats_df.empty:
                    edge_feats_df.to_csv(f"{outdir_name}.edge_feats", sep='\t', index=False, header=False)

                edges_df.to_csv(f"{outdir_name}.edges", sep='\t', index=False, header=False)

                success=True
            except Exception as e:
                print(e)
                retry+=1
                print("Connection to Netzschleuder failed. Retrying in 3 seconds...\n")
                time.sleep(3)

        if not success:
            print(f"{name} not collected. Going on...\n")

    print("Collection completed!")



if __name__=='__main__':

    filtered_graph_names = filter_graph_names(**args)

    correctInput = False

    while not correctInput:
        print("Do you want to proceed with collection and savings of these graphs? ",
        "(YES/NO):    ")

        x = input()

        if x not in ["YES","NO"]:
            print("Only possible answers are 'YES' or 'NO'\n")
        else:
            correctInput=True

    if correctInput and x=="YES":
        get_save_graphs(filtered_graph_names)
