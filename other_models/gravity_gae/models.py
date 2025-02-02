# Credit to: https://github.com/deezer/gravity_graph_autoencoders/tree/master

from torch.nn import Sequential, ReLU
from GNN import LayerWrapper, LinkPropertyPredictorGravity, LinkPropertyPredictorSourceTarget, GNN_FB, ParallelLayerWrapper
from Convolution import Conv
from custom_losses import losses_sum_closure, auc_loss, ap_loss
from VGAE import VGAE_Reparametrization


def get_gravity_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation, CLAMP, l , train_l):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = LinkPropertyPredictorGravity(l = l, train_l=train_l, CLAMP = CLAMP)
    return Sequential(encoder, decoder)




def get_gravity_vgae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation, CLAMP, l , train_l):

    
    unwrapped_layers_kwargs = [
                        {"layer": Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },
                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        }]



    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]),
                                    ParallelLayerWrapper([LayerWrapper(**unwrapped_layers_kwargs[1]), LayerWrapper(**unwrapped_layers_kwargs[2])]),
                                    VGAE_Reparametrization()])
    decoder = LinkPropertyPredictorGravity(l = l, train_l = train_l, CLAMP = CLAMP)
    return Sequential(encoder, decoder)



def get_sourcetarget_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = LinkPropertyPredictorSourceTarget()
    return Sequential(encoder, decoder)


def get_sourcetarget_vgae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation):

    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension ), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },
                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        }]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]),
                                        ParallelLayerWrapper([LayerWrapper(**unwrapped_layers_kwargs[1]), LayerWrapper(**unwrapped_layers_kwargs[2])]),
                                        VGAE_Reparametrization()])

    decoder = LinkPropertyPredictorSourceTarget()
    return Sequential(encoder, decoder)


models_suggested_parameters_sets = {"general":{
                                        "general":{
                                                    "gravity_gae": {"input_dimension":None , "hidden_dimension": 64, "output_dimension":64, "use_sparse_representation": True, "CLAMP" :None, "l": 1. , "train_l":True},

                                                    "gravity_vgae": {"input_dimension":None , "hidden_dimension": 64, "output_dimension":64, "use_sparse_representation": True, "CLAMP" :None, "l": 1. ,  "train_l":True},

                                                    "sourcetarget_gae": {"input_dimension":None , "hidden_dimension": 64, "output_dimension":64, "use_sparse_representation": True},

                                                    "sourcetarget_vgae": {"input_dimension":None , "hidden_dimension": 64, "output_dimension":64, "use_sparse_representation": True}

                                                    }
                                            }
}
                                        


setup_suggested_parameters_sets = {"general":{
                                        "general":{
                                                    "gravity_gae": {"num_epochs":200, "lr":0.001, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                                    "gravity_vgae": {"num_epochs":200, "lr":0.001, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                                    "sourcetarget_gae": {"num_epochs":200, "lr":0.001, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                                    "sourcetarget_vgae": {"num_epochs":200, "lr":0.001, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                                    }
                                        }
}