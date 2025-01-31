import numpy as np
from math import log10, floor
from pandas import DataFrame
import csv

# Model utilities
def reset_parameters(module):
    for layer in module.children():
            # print(f"layer 1= {layer}")
            if hasattr(layer, 'reset_parameters'):
                print(f"resetting {layer}")
                layer.reset_parameters()
            elif len(list(layer.children())) > 0:
                reset_parameters(layer)


def print_model_parameters_names(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)




def summarize_link_prediction_evaluation(performances):
    mean_std_dict = {}
    for metric in ['AUC', 'F1', 'hitsk', 'AP']:
        vals = []
        for run in performances:
            vals.append(run[metric])

        mean_std_dict[metric] = {"mean":np.nanmean( list(filter(None, vals)) ), "std": np.nanstd( list(filter(None, vals)) )}

    return mean_std_dict



def round_to_first_significative_digit(x):
    digit = -int(floor(log10(abs(x))))
    return digit, round(x, digit)

def pretty_print_link_performance_evaluation(mean_std_dict, model_name):
    performances_strings = {}

    for (metric,mean_std) in mean_std_dict.items():
        if np.isnan(mean_std["mean"]):
            performances_strings[metric] = str(None)
        elif mean_std["std"] == 0:
            digit, mean_rounded = round_to_first_significative_digit(mean_std["mean"]) 
            performances_strings[metric] = str(mean_rounded) + " +- " + str(mean_std["std"])
        else:
            digit, std_rounded = round_to_first_significative_digit(mean_std["std"])
            mean_rounded = round(mean_std["mean"], digit)
            performances_strings[metric] = str(mean_rounded) + " +- " + str(std_rounded)

    
    df = DataFrame(performances_strings.values(), columns = [model_name], index = performances_strings.keys()  )
    return df.to_markdown(index=True)

def link_performance_evaluation_to_csv_row(mean_std_dict, model_name, dataset, training_times, csv_file):
    # Define the header for the CSV file
    header = ['Dataset', 'Model', 'AUC_mean', 'AUC_std', 'AP_mean', 'AP_std', 'Train_time_mean', 'Train_time_std']

    writer = csv.writer(csv_file)
    # If the file is empty, write the header
    if csv_file.tell() == 0:
        writer.writerow(header)
    if mean_std_dict is not None:
        # Write dataset and results into the CSV
        writer.writerow([dataset, model_name,
                            mean_std_dict['AUC']['mean'],
                            mean_std_dict['AUC']['std'],
                            mean_std_dict['AP']['mean'],
                            mean_std_dict['AP']['std'],
                            np.mean(training_times),
                            np.std(training_times)
                            ])
    else:
        # If an exception occurs, record None
        writer.writerow([dataset, model_name,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None
                            ])