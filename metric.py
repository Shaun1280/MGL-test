import numpy as np
from rectorch.metrics import Metrics

def ranking_measure_testset(pred_scores, ground_truth, k, test_item):
    # Select the scores for the test item
    pred_scores = pred_scores[:, test_item]
    ground_truth = ground_truth[:, test_item]

    # Initialize lists to store the metrics
    metrics_list = {'ndcg': [], 'recall': []}

    # Calculate metrics and append to the respective lists
    for metric_name in metrics_list.keys():
        metric_function = getattr(Metrics, f'{metric_name}_at_k')
        metrics_list[metric_name] = metric_function(pred_scores, ground_truth, k).tolist()

    # Calculate the mean of each metric and return
    return {metric: np.mean(values) for metric, values in metrics_list.items()}

def ranking_measure_degree_testset(pred_scores, ground_truth, k, item_degrees, separate_rate, test_item):
    # Sort the item degrees
    sorted_item_degrees = sorted(item_degrees.items(), key=lambda x: x[1])
    item_list_sorted = [item for item, _ in sorted_item_degrees]

    # Calculate the lengths
    total_length = len(item_list_sorted)
    head_length = int(total_length * separate_rate)
    tail_length = int(total_length * separate_rate)

    # Get the items for head, tail, and body
    head_item = list(set(item_list_sorted[-head_length:]).intersection(set(test_item)))
    tail_item = list(set(item_list_sorted[:tail_length]).intersection(set(test_item)))

    # Initialize dictionary to store the metrics
    metrics_dict = {'head': head_item, 'tail': tail_item}
    metrics = {'head': {}, 'tail': {}, 'body': {}}

    # Calculate metrics for each part and store in the dictionary
    for part, items in metrics_dict.items():
        for metric in ['ndcg', 'recall']:
            metric_function = getattr(Metrics, f'{metric}_at_k')
            metric_values = np.nan_to_num(metric_function(pred_scores[:, items], ground_truth[:, items], k)).tolist()
            metrics[part][metric] = np.mean(metric_values)

    return metrics

