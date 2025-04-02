import os
import torch
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config_folder import client_config_file

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def client_train_step(
    model: torch.nn.Module,
    data_loader,
    feature_name: str,
    label_name: str,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int=1,
    device: torch.device="cpu",
    verbose: bool=True,
    ):
    """
    Perform model training using torch DataLoader.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train() #put model in training mode
        train_loss, train_acc = 0, 0
        try:
            for i, batch in enumerate(data_loader):
                X, y = batch[feature_name].to(device), batch[label_name].to(device)
                y_logits = model(X)
                loss = loss_fn(
                    y_logits, 
                    y
                )
                train_loss += loss #accumulate loss per batch
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if len(y_logits)>1: #if batch has more than 1 sample no need to add another dimension
                    y_pred = F.softmax(
                        y_logits.squeeze(), 
                        dim=0).argmax(dim=1) #use softmax for multi class classification and argmax gives the index of the predicted class
                elif len(y_logits)==1: #if batch size is 1 then add 1 more dimension
                    y_pred = F.softmax(
                        y_logits.squeeze(), 
                        dim=0).unsqueeze(dim=0).argmax(dim=1)
                    
                acc = accuracy_fn(
                    y_true=y,
                    y_pred=y_pred
                )
                train_acc += acc #accumulate accuracy per batch
                
                if verbose and i%50==0:
                    print(f"Local Training Batch: {i} | Train Loss: {loss.item():.3f} | Train Acc: {acc:.3f}")
            train_loss = train_loss/len(data_loader) #average training loss
            train_acc = train_acc/len(data_loader) #average training acc
            print(f"Local Training Loss: {train_loss:.3f} | Local Training Accuracy {train_acc:.3f}")
        except Exception as e:
            print(f"Error training model {e}")

def client_test_step(
    model: torch.nn.Module,
    data_loader,
    feature_name: str,
    label_name: str,
    loss_fn: torch.nn.Module,
    device: torch.device="cpu",
    verbose: bool=True,
    ):
    """
    Performs a testing loop step on model going over data_loader.
    """
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch in data_loader:
            X, y = batch[feature_name].to(device), batch[label_name].to(device)
            test_logits = model(X)
            
            test_loss += loss_fn(
                test_logits,
                y
            )

            if len(test_logits)>1: #if batch has more than 1 sample no need to add another dimension
                test_pred = F.softmax(
                    test_logits.squeeze(), 
                    dim=0).argmax(dim=1) #use softmax for multi class classification and argmax gives the index of the predicted class
            elif len(test_logits)==1: #if batch size is 1 then add 1 more dimension
                test_pred = F.softmax(
                    test_logits.squeeze(), 
                    dim=0).unsqueeze(dim=0).argmax(dim=1)
                
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred
            )
        # adjust metrics and print out
        #total loss and total accuracy divided by the total number of batches
        test_loss = test_loss/len(data_loader)
        test_acc = test_acc/len(data_loader)

    if verbose:
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}")
    
    return test_loss, test_acc


def client_test_metrics(
    model: torch.nn.Module,
    data_loader,
    feature_name: str,
    label_name: str,
    loss_fn: torch.nn.Module,
    device: torch.device="cpu"
    ):
    """
    Evaluate the model and return F1-scores by class.
    Returns:
    f1_scores: f1 scores by class in a pandas dataframe
    """
    import pandas as pd
    from collections import defaultdict
    from sklearn.metrics import classification_report, confusion_matrix
    
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        f1_scores = list()
        precision = list()
        recall = list()
        for batch in data_loader:
            X, y = batch[feature_name].to(device), batch[label_name].to(device)
            y_logits = model(X)
            
            #test loss
            test_loss += loss_fn(
                y_logits, 
                y
            ).item()

            #test accuracy
            if len(y_logits)>1: #if batch has more than 1 sample no need to add another dimension
                y_preds = F.softmax(
                    y_logits.squeeze(), 
                    dim=0).argmax(dim=1) #use softmax for multi class classification and argmax gives the index of the predicted class
            elif len(y_logits)==1: #if batch size is 1 then add 1 more dimension
                y_preds = F.softmax(
                    y_logits.squeeze(), 
                    dim=0).unsqueeze(dim=0).argmax(dim=1)
            
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=y_preds
            )

            batch_metrics_df = batch_metrics(
                y=y.cpu(),
                y_preds=y_preds.cpu()
            )
            f1_scores.append(batch_metrics_df[['class', 'f1_scores']])
            precision.append(batch_metrics_df[['class', 'precision']])
            recall.append(batch_metrics_df[['class', 'recall']])

        #get mean metrics across all batches
        f1_scores_df = get_metrics_df(f1_scores, metric='f1_scores')
        precision_df = get_metrics_df(precision, metric='precision')
        recall_df = get_metrics_df(recall, metric='recall')

        test_loss = test_loss/len(data_loader)
        test_acc = test_acc/len(data_loader)

    return f1_scores_df, precision_df, recall_df, test_loss, test_acc

def get_metrics_df(score_df_list, metric='f1_scores'):
    """
    Given a list of dataframes for a given metric this function returns a 
    default dict with each key representing the class and value for each key
    as mean of given metric across all batches.
    score_df_list: list of dataframes for given metric
    metric: 'precision', 'recall', or 'f1_scores'
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    metrics_dict = defaultdict(list)
    for df in score_df_list:
        for row in df.iterrows():
            metrics_dict[int(row[1]['class'])].append(row[1][metric])

    _metrics_dict = defaultdict()
    for k, v in metrics_dict.items():
        _metrics_dict[k] = np.mean(np.array(v)) #replace this with a geometric mean of f1 scores and do math proof
    
    metrics_df = pd.DataFrame(index=range(len(_metrics_dict)))
    metrics_df['class'] = [int(i) for i in _metrics_dict.keys()]
    metrics_df[metric] = _metrics_dict.values()
    return metrics_df


def batch_metrics(y, y_preds):
    """
    Compute batch metrics:
    y: true labels
    y_preds: predicted labels
    Metrics: precision, recall, f1_scores
    """
    import pandas as pd
    from sklearn.metrics import classification_report

    metrics = classification_report(
        y.cpu(), 
        y_preds.cpu(), 
        output_dict=True,
        zero_division=0
    )
    classes = y.unique().cpu().numpy()

    batch_metrics_df = pd.DataFrame()
    batch_metrics_df['class'] = classes

    f1_scores = list()
    precision = list()
    recall = list()
    for c in classes:
        f1_scores.append(metrics[str(c)]['f1-score'])
        precision.append(metrics[str(c)]['precision'])
        recall.append(metrics[str(c)]['recall'])
    batch_metrics_df['f1_scores'] = f1_scores
    batch_metrics_df['precision'] = precision
    batch_metrics_df['recall'] = recall
    return batch_metrics_df