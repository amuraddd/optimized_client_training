import re
import json
import pandas as pd
import numpy as np
from collections import defaultdict

def get_final_validation_accuracy(df, client_type='otp'):
# client_type='otp'
# df = otp_metrics
    strategies = df['strategy'].unique().tolist()
    experiment_types = df['experiment_type'].unique().tolist()

    if client_type=='otp' or client_type=='ntp':
        column_name = 'accuracy_on_validation_dataset_after_local_training'
    if client_type=='otp_clone':
        column_name = 'accuracy_on_validation_dataset_after_aggregation_and_finetuning'
    if client_type=='otp' or client_type=='otp_clone':
        df['client_id'] = 0
    final_val_performance = pd.DataFrame()
    for strategy in strategies:
        df_copy = df[df['strategy']==strategy]
        experiment_types = df_copy['experiment_type'].unique().tolist()
        # print(strategy, experiment_types)
        for exp in experiment_types:
            temp_df = df_copy[df_copy['experiment_type']==exp]
            
            # print(f"Client Type: {client_type} | Strategy: {strategy} | Experiment Type: {exp}")
            # try:
            if client_type=='ntp':
                clients = temp_df['client_id'].unique().tolist()
                final_val_performance_per_client = pd.DataFrame()
                try:
                    for client in clients:
                        client_temp_df = temp_df[temp_df['client_id']==client]
                        client_temp_df = client_temp_df.iloc[[len(client_temp_df)-1]][['client_id', 'server_round', 'strategy', 'experiment_type', column_name]]
                        final_val_performance_per_client = pd.concat([final_val_performance_per_client, client_temp_df], ignore_index=True)
                        # print(temp_df)
                    final_val_performance = pd.concat([final_val_performance, final_val_performance_per_client], ignore_index=True)
                except Exception as e:
                    print(e)
                    continue
            if client_type=='otp' or client_type=='otp_clone':
                try:
                    temp_df = temp_df.iloc[[len(temp_df)-1]][['client_id', 'server_round', 'strategy', 'experiment_type', column_name]]
                    final_val_performance = pd.concat([final_val_performance, temp_df], ignore_index=True)
                except Exception as e:
                    print(e)
                    continue
    return final_val_performance

def get_final_validation_summary_df(dfs, clients, metric="accuracy"):
    all_final_dfs = list()
    # for df, client in zip(dfs, clients):
    #     all_final_dfs.append(get_final_validation_accuracy(df, client_type=client))
    dfs[2] = dfs[2][['strategy', 'client_id','experiment_type', f'{metric}_on_validation_dataset_after_aggregation']].\
        groupby(['strategy', 'client_id','experiment_type'], as_index=False).mean()
    summary_df = dfs[2][['strategy','experiment_type', f'{metric}_on_validation_dataset_after_aggregation']].groupby(['strategy','experiment_type']).mean().join(
                    dfs[1][['strategy','experiment_type', f'{metric}_on_validation_dataset_after_aggregation']].groupby(['strategy','experiment_type']).mean(),
                    lsuffix='_naive',
                    rsuffix='_optimal').join(
                        dfs[0][['strategy','experiment_type', f'{metric}_on_validation_dataset_after_aggregation_and_finetuning']].groupby(['strategy','experiment_type']).max(),
                        # lsuffix='_before_finetuning',
                        # rsuffix='_after_finetuning'
                    )
    for strategy in ['FedAvg', 'FedAvgM', 'FedCDA', 'FedProx', 'FedMedian']:
        try:
            summary_df.replace(summary_df.loc[pd.IndexSlice[:, f'{strategy}_None'], :][f'{metric}_on_validation_dataset_after_aggregation_and_finetuning'].iloc[0], np.nan, inplace=True)
        except:
            continue
    summary_df[f'optimal_validation_{metric}_improvement'] = summary_df[f'{metric}_on_validation_dataset_after_aggregation_and_finetuning']-\
                                                                summary_df[f'{metric}_on_validation_dataset_after_aggregation_naive']
    summary_df = summary_df[[f'{metric}_on_validation_dataset_after_aggregation_naive',
                            f'{metric}_on_validation_dataset_after_aggregation_optimal',
                            f'{metric}_on_validation_dataset_after_aggregation_and_finetuning',
                            f'optimal_validation_{metric}_improvement']]
    return summary_df

def get_best_performance(ntp_metrics, otp_metrics, otp_clone_metrics, metric="accuracy"):
    best_ntp_metrics = pd.DataFrame()
    # if metric=="accuracy":
    for client in ntp_metrics['client_id'].unique():
        for exp in ntp_metrics['experiment_type'].unique():
            exp_df = ntp_metrics[(ntp_metrics['experiment_type']==exp)&(ntp_metrics['client_id']==client)]#.tail()
            best_ntp_performance_df = exp_df[exp_df[f'{metric}_on_validation_dataset_after_aggregation']==\
                                        exp_df[f'{metric}_on_validation_dataset_after_aggregation'].max()]
            best_ntp_metrics = pd.concat([best_ntp_metrics, best_ntp_performance_df], ignore_index=True)
    # else:
    #     for exp in ntp_metrics['experiment_type'].unique():
    #         exp_df = ntp_metrics[(ntp_metrics['experiment_type']==exp)&(ntp_metrics['client_id']==client)]#.tail()
    #         best_ntp_performance_df = exp_df[exp_df[f'{metric}_on_validation_dataset_after_aggregation']==\
    #                                     exp_df[f'{metric}_on_validation_dataset_after_aggregation'].max()]
    #         best_ntp_metrics = pd.concat([best_ntp_metrics, best_ntp_performance_df], ignore_index=True)
    
    best_otp_metrics = pd.DataFrame()
    for exp in otp_metrics['experiment_type'].unique():
        exp_df = otp_metrics[otp_metrics['experiment_type']==exp]#.tail()
        best_otp_performance_df = exp_df[exp_df[f'{metric}_on_validation_dataset_after_aggregation']==\
                                    exp_df[f'{metric}_on_validation_dataset_after_aggregation'].max()]
        best_otp_metrics = pd.concat([best_otp_metrics, best_otp_performance_df], ignore_index=True)

    best_opt_clone_metrics = pd.DataFrame()
    for exp in otp_clone_metrics['experiment_type'].unique():
        exp_df = otp_clone_metrics[otp_clone_metrics['experiment_type']==exp]
        best_otp_clone_performance_df = exp_df[exp_df[f'{metric}_on_validation_dataset_after_aggregation_and_finetuning']==\
                                    exp_df[f'{metric}_on_validation_dataset_after_aggregation_and_finetuning'].max()]
        best_opt_clone_metrics = pd.concat([best_opt_clone_metrics, best_otp_clone_performance_df], ignore_index=True)
    return best_ntp_metrics, best_otp_metrics, best_opt_clone_metrics

def get_mean_naive_action(df):
    # df = ntp_metrics
    all_actions = list()
    weights_by_class = defaultdict(list)
    for client in df['client_id'].unique():
        temp_df = df[df['client_id']==client]
        for strategy in temp_df['strategy'].unique():
            temp_df = temp_df[temp_df['strategy']==strategy]
            for i, row in enumerate(temp_df.iterrows()):
                classes = list(json.loads(row[1]['f1_scores_on_local_dataset_after_aggregation'])['class'].values())
                action = [float(j) for j in re.findall(r"[-+]?(?:\d*\.*\d+)", row[1]['action'])]
                for c, a in zip(classes, action):
                    # for a in action:
                    weights_by_class[c].append(a)
    mean_values = {key: sum(value) / len(value) for key, value in weights_by_class.items()}
    temp_df = pd.DataFrame()
    temp_df['classes'] = mean_values.keys()
    temp_df['action'] = mean_values.values()
    temp_df = temp_df.sort_values(by=['classes'])
    return temp_df

def get_mean_optimal_action(df):
    # df = ntp_metrics
    df = df[(df['action_type']!='naive')]
    temp_df = df[~df['experiment_type'].str.contains(r'_take_random_action', regex=True)]
    weights_by_class = defaultdict(list)
    for i, row in enumerate(temp_df.iterrows()):
        classes = list(json.loads(row[1]['f1_scores_on_local_dataset_after_aggregation'])['class'].values())
        action = [float(j) for j in re.findall(r"[-+]?(?:\d*\.*\d+)", row[1]['action'])]
        for c, a in zip(classes, action):
            weights_by_class[c].append(a)
    mean_values = {key: sum(value) / len(value) for key, value in weights_by_class.items()}
    temp_df = pd.DataFrame()
    temp_df['classes'] = mean_values.keys()
    temp_df['action'] = mean_values.values()
    temp_df = temp_df.sort_values(by=['classes'])
    return temp_df