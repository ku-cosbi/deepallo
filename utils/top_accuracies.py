import numpy as np


THRESHOLD = 0.5

def topN(y_pred, y_true):
    paired = list(zip(y_pred, y_true))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

    topPCT = []

    n_true = sum(y_true)
    n = len(paired_sorted)
    partitions = [n * 0.01, n * 0.03, n * 0.05, n * 0.1, n * 0.2]
    for i in range(len(partitions)):
        topPCT.append(sum([1 for i in range(int(partitions[i])) if paired_sorted[i][0] > THRESHOLD and paired_sorted[i][1] == 1]) / n_true * 100)

    return topPCT

def topk(Y_Pred, Y_True):
    top1 = 0
    top2 = 0
    top3 = 0

    for i in range(len(Y_True)):
        sorted_indices = sorted(range(len(Y_Pred[i])), key=lambda k: Y_Pred[i][k], reverse=True)
        sorted_probs = sorted(Y_Pred[i], reverse=True)

        if (len(Y_True)) > 0 and sorted_probs[0] > THRESHOLD and Y_True[i][sorted_indices[0]] == 1:
            top1 += 1
        if (len(Y_True)) > 1 and sorted_probs[1] > THRESHOLD and Y_True[i][sorted_indices[1]] == 1:
            top2 += 1
        if (len(Y_True)) > 1 and sorted_probs[2] > THRESHOLD and Y_True[i][sorted_indices[2]] == 1:
            top3 += 1

    top1_percent = top1 / len(Y_True) * 100
    top2_percent = (top1 + top2) / len(Y_True) * 100
    top3_percent = (top1 + top2 + top3) / len(Y_True) * 100

    return top1_percent, top2_percent, top3_percent

def calculate_top_n_accuracies_homogenious(y_true, y_pred, n_values=[1, 2, 3]):
    """
    Calculate top-N accuracies for predictions of pocket presence.

    Parameters:
    - y_true: List of lists containing true binary labels for each protein's pockets.
    - y_pred: List of lists containing predicted binary labels for each protein's pockets.
    - n_values: List of N values to calculate top-N accuracy for.

    Returns:
    - Dictionary of top-N accuracies.
    """
    # Convert to NumPy arrays for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    top_n_accuracies = {f'top_{n}': 0 for n in n_values}
    total_proteins = len(y_true)

    for true, pred in zip(y_true, y_pred):
        # Sort indices by predicted value in descending order
        sorted_indices = np.argsort(pred)[::-1]

        # Calculate top-N accuracies
        for n in n_values:
            top_n_indices = sorted_indices[:n]
            # Check if any true positive is in the top-N predictions
            if any(true[idx] == 1 for idx in top_n_indices):
                top_n_accuracies[f'top_{n}'] += 1

    # Calculate the accuracy percentage
    for n in n_values:
        top_n_accuracies[f'top_{n}'] /= total_proteins

    return top_n_accuracies

def calculate_top_n_accuracies(y_true, y_pred, n_values=[1, 2, 3]):
    """
    Calculate top-N accuracies for predictions of pocket presence.

    Parameters:
    - y_true: List of lists containing true binary labels for each protein's pockets.
    - y_pred: List of lists containing predicted probabilities for each protein's pockets.
    - n_values: List of N values to calculate top-N accuracy for.

    Returns:
    - Dictionary of top-N accuracies.
    """
    top_n_accuracies = {f'top_{n}': 0 for n in n_values}
    total_proteins = len(y_true)

    for true, pred in zip(y_true, y_pred):
        true = np.array(true)
        pred = np.array(pred)

        # Sort indices by predicted probability in descending order
        sorted_indices = np.argsort(pred)[::-1]

        # Calculate top-N accuracies
        for n in n_values:
            top_n_indices = sorted_indices[:n]
            # Check if any true positive is in the top-N predictions
            if any(true[idx] == 1 for idx in top_n_indices):
                top_n_accuracies[f'top_{n}'] += 1

    # Calculate the accuracy percentage
    for n in n_values:
        top_n_accuracies[f'top_{n}'] /= total_proteins

    return top_n_accuracies
