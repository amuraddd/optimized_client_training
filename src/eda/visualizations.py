import re
import json
import torch
import pandas as pd
import numpy as np
from torch import tensor
from matplotlib import rcParams
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plt.rcParams['font.family'] = 'Times New Roman'

def plot_mean_reward_action(agent_data, ntp_metrics, rolling_window=10, figsize=[15, 7]):
    rcParams['figure.figsize'] = figsize
    client_ids = ntp_metrics['client_id'].unique().tolist()
    experiment_types = ntp_metrics['experiment_type'].unique().tolist()#[:2]
    client_ids = np.sort([int(i) for i in client_ids])
    client_ids = [i for i in client_ids]


    fig, axs = plt.subplots(2,2)
    for exp in experiment_types:
        for i in client_ids:
            df = ntp_metrics[(ntp_metrics['client_id']==i)&(ntp_metrics['experiment_type']==exp)]
            rounds = df['server_round']
            ntp_reward = pd.Series([eval(row[1]['reward']).item() for row in df.iterrows()]).rolling(rolling_window).mean()
            axs[0, 0].plot(rounds, ntp_reward, label=f'naive_reward_client_{i}_{exp}')
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].set_xlabel("Server Round")
        axs[0, 0].legend(loc='upper left', fontsize=8)

        temp_agent_data = agent_data[agent_data['experiment_type']==exp]
        rounds = range(len(temp_agent_data))
        rewards = pd.Series([eval(row[1]['reward']).item() for row in temp_agent_data.iterrows()]).rolling(rolling_window).mean()
        axs[0, 1].plot(rounds, rewards, label=f'optimal_reward_{exp}')
        axs[0, 1].set_ylabel("Reward")
        axs[0, 1].set_xlabel("Server Round")
        axs[0, 1].legend(loc='upper left', fontsize=8)
    
    # axs[0, 1].show()
    for exp in experiment_types:
        for i in client_ids:
            df = ntp_metrics[(ntp_metrics['client_id']==i)&(ntp_metrics['experiment_type']==exp)]
            rounds = df['server_round']
            ntp_action = pd.Series([np.mean(np.array(
                [float(i) for i in re.findall(r"[-+]?(?:\d*\.*\d+)", row[1]['action'])]
            )) for row in df.iterrows()]).rolling(rolling_window).mean()
            axs[1, 0].plot(rounds, ntp_action, label=f'naive_action_client_{i}_{exp}')
            axs[1, 0].set_ylabel("Mean Action Across All Classes")
            axs[1, 0].set_xlabel("Server Round")
            axs[1, 0].legend(loc='upper left', fontsize=8)
        temp_agent_data = agent_data[agent_data['experiment_type']==exp]
        actions = pd.Series([np.mean(eval(row[1]['action'])) for row in temp_agent_data.iterrows()]).rolling(rolling_window).mean()
        axs[1, 1].plot(range(len(actions)), actions, label=f'optimal actions {exp}')
        axs[1, 1].set_ylabel("Mean Action Across All Classes")
        axs[1, 1].set_xlabel("Server Round")
        axs[1, 1].legend(loc='upper left', fontsize=8)

def plot_actor_critic_loss(agent_training_data, rolling_window=10, figsize=[15, 5]):
    rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(1, 2)
    experiment_types = agent_training_data['experiment_type'].unique().tolist()#[:2]
    for exp in experiment_types:
        temp_agent_training_data = agent_training_data[agent_training_data['experiment_type']==exp]
        rounds = range(len(temp_agent_training_data))
        loss = pd.Series([row[1]['actor_loss'] for row in temp_agent_training_data.iterrows()]).rolling(rolling_window).mean()
        critic_loss = pd.Series([row[1]['critic_loss'] for row in temp_agent_training_data.iterrows()]).rolling(rolling_window).mean()

        axs[0].plot(rounds, loss, label=f'Actor Loss | {exp}')
        axs[0].set_ylabel("Actor Loss")
        axs[0].set_xlabel("Server Round")
        axs[0].legend(loc="upper right", fontsize=8)


        axs[1].plot(rounds, critic_loss, label=f'Critic Loss | {exp}')
        axs[1].set_ylabel("Critic Loss")
        axs[1].set_xlabel("Server Round")
        axs[1].legend(loc="upper right", fontsize=8)

def plot_otp_loss_accuracy(otp_metrics, rolling_window=10, figsize=[15, 5]):
    rcParams['figure.figsize'] = figsize
    rounds = otp_metrics['server_round']
    fig, axs = plt.subplots(1, 2)
    estimated_loss = otp_metrics['estimated_loss'].rolling(rolling_window).mean()
    otp_loss_on_local_dataset_after_aggregation = otp_metrics['loss_on_local_dataset_after_aggregation'].rolling(rolling_window).mean()
    otp_loss_on_local_dataset_after_local_training = otp_metrics['loss_on_local_dataset_after_local_training'].rolling(rolling_window).mean()
    otp_loss_on_neutral_dataset_after_local_training = otp_metrics['loss_on_neutral_dataset_after_local_training'].rolling(rolling_window).mean()

    axs[0].plot(rounds, estimated_loss, label='estimated_loss')
    axs[0].plot(rounds, otp_loss_on_local_dataset_after_aggregation, label='loss_on_local_dataset_after_aggregation')
    axs[0].plot(rounds, otp_loss_on_local_dataset_after_local_training, label='loss_on_local_dataset_after_local_training')
    axs[0].plot(rounds, otp_loss_on_neutral_dataset_after_local_training, label='loss_on_neutral_dataset_after_local_training')
    axs[0].set_xlabel("Server Round")
    axs[0].legend(loc="upper right", fontsize=8)

    otp_accuracy_on_local_dataset_after_aggregation = otp_metrics['accuracy_on_local_dataset_after_aggregation'].rolling(rolling_window).mean()
    otp_accuracy_on_local_dataset_after_local_training = otp_metrics['accuracy_on_local_dataset_after_local_training'].rolling(rolling_window).mean()
    accuracy_on_neutral_dataset_after_local_training = otp_metrics['accuracy_on_neutral_dataset_after_local_training'].rolling(rolling_window).mean()
    axs[1].plot(rounds, otp_accuracy_on_local_dataset_after_aggregation, label='accuracy_on_local_dataset_after_aggregation')
    axs[1].plot(rounds, otp_accuracy_on_local_dataset_after_local_training, label='accuracy_on_local_dataset_after_local_training')
    axs[1].plot(rounds, accuracy_on_neutral_dataset_after_local_training, label='accuracy_on_neutral_dataset_after_local_training')
    axs[1].set_xlabel("Server Round")
    axs[1].legend(loc="upper left", fontsize=8)

def plot_mean_metric(ntp_metrics, otp_metrics, metrics=['precision', 'recall', 'f1_scores'], rolling_window=10, figsize=[15, 7]):
    rcParams['figure.figsize'] = figsize
    client_ids = ntp_metrics['client_id'].unique().tolist()
    experiment_types = ntp_metrics['experiment_type'].unique().tolist()#[:2]
    client_ids = np.sort([int(i) for i in client_ids])
    client_ids = [i for i in client_ids]

    fig, axs = plt.subplots(1, len(metrics))
    for exp in experiment_types:
        for idx, metric in enumerate(metrics):
            all_clients_mean_metric = pd.DataFrame()
            for i in client_ids:
                df = ntp_metrics[(ntp_metrics['client_id']==i)&(ntp_metrics['experiment_type']==exp)]
                mean_metric_list = list()
                mean_metric = pd.DataFrame(index=range(len(df)))
                for row in df.iterrows():
                    mean_metric_list.append(pd.DataFrame(eval(row[1][f'{metric}_on_validation_dataset_after_local_training']))[metric].mean())
                mean_metric['client_id'] = i
                mean_metric[f'mean_{metric}'] = mean_metric_list
                all_clients_mean_metric = pd.concat([all_clients_mean_metric, mean_metric])

            otp_mean_metric_by_exp = otp_metrics[otp_metrics['experiment_type']==exp]
            otp_mean_metric = pd.DataFrame(index=range(len(otp_mean_metric_by_exp)))
            otp_mean_metric_list = list()
            for row in otp_mean_metric_by_exp.iterrows():
                otp_mean_metric_list.append(pd.DataFrame(eval(row[1][f'{metric}_on_validation_dataset_after_local_training']))[metric].mean())
            otp_mean_metric['client_id'] = i
            otp_mean_metric[f'mean_{metric}'] = otp_mean_metric_list

            for i in client_ids:
                axs[idx].plot(all_clients_mean_metric[all_clients_mean_metric['client_id']==i][[f'mean_{metric}']].rolling(rolling_window).mean(), label=f'client_{i}_mean_{metric}_{exp}')
                # axs[idx].legend(loc="upper left", fontsize=10)
            axs[idx].plot(otp_mean_metric[f'mean_{metric}'].rolling(rolling_window).mean(), label=f'otp_client_mean_{metric}_{exp}')
            # axs[idx].legend(loc="upper left", fontsize=10)

def plot_mean_metric_with_finetuning(ntp_metrics, otp_metrics, metrics=['precision', 'recall', 'f1_scores'], rolling_window=10, figsize=[15, 7]):
    rcParams['figure.figsize'] = figsize
    client_ids = ntp_metrics['client_id'].unique().tolist()
    experiment_types = ntp_metrics['experiment_type'].unique().tolist()#[:2]
    client_ids = np.sort([int(i) for i in client_ids])
    client_ids = [i for i in client_ids]

    fig, axs = plt.subplots(1, len(metrics))
    for exp in experiment_types:
        for idx, metric in enumerate(metrics):
            all_clients_mean_metric = pd.DataFrame()
            for i in client_ids:
                df = ntp_metrics[(ntp_metrics['client_id']==i)&(ntp_metrics['experiment_type']==exp)]
                mean_metric_list = list()
                mean_metric = pd.DataFrame(index=range(len(df)))
                for row in df.iterrows():
                    mean_metric_list.append(pd.DataFrame(eval(row[1][f'{metric}_on_validation_dataset_after_local_training']))[metric].mean())
                mean_metric['client_id'] = i
                mean_metric[f'mean_{metric}'] = mean_metric_list
                all_clients_mean_metric = pd.concat([all_clients_mean_metric, mean_metric])

            otp_mean_metric_by_exp = otp_metrics[otp_metrics['experiment_type']==exp]
            otp_mean_metric = pd.DataFrame(index=range(len(otp_mean_metric_by_exp)))
            otp_mean_metric_list = list()
            for row in otp_mean_metric_by_exp.iterrows():
                otp_mean_metric_list.append(pd.DataFrame(eval(row[1][f'{metric}_on_validation_dataset_after_aggregation_and_finetuning']))[metric].mean())
            otp_mean_metric['client_id'] = i
            otp_mean_metric[f'mean_{metric}'] = otp_mean_metric_list

            for i in client_ids:
                axs[idx].plot(all_clients_mean_metric[all_clients_mean_metric['client_id']==i][[f'mean_{metric}']].rolling(rolling_window).mean(), label=f'client_{i}_mean_{metric}_{exp}')
                axs[idx].legend(loc="upper left", fontsize=10)
            axs[idx].plot(otp_mean_metric[f'mean_{metric}'].rolling(rolling_window).mean(), label=f'otp_client_mean_{metric}_{exp}')
            axs[idx].legend(loc="lower right", fontsize=10)

def plot_action_distribution(agent_data, mean_ntp_actions, num_classes, width=1500, height=650, rows=1, cols=2, font_size=15, margin=dict(l=30, r=30, t=30, b=30), 
                             left_x=0.05, left_y=0.9, right_x=0.5, right_y=0.6):
  fig = make_subplots(
    rows=rows, 
    cols=cols, 
    subplot_titles=('Optimal Actions',  'Naive Actions'),
    specs=[[{"type": "polar"} for _ in range(2)] for _ in range(1)]
  )
  # mean_ntp_actions = mean_ntp_actions[mean_ntp_actions['experiment_type']==exp]
  all_classes = list()
  for i, row in enumerate(mean_ntp_actions.iterrows()):
      classes = row[1]['classes']
      classes.append(classes[0])
      num_classes = len(classes)
      action = list(row[1]['action'])
      action.append(action[0])
      client_id = row[1]['client_id']
      if client_id==1:
          continue
      # print(classes[0], classes, action)
      fig.add_trace(
        go.Scatterpolar(
          r=action,
          theta=[str(i) for i in classes],
          name=f"Client_{client_id}",
          showlegend=False,
          marker=dict(size=7, symbol=[2]),
          line=dict(shape='linear', dash='solid'),
          opacity=1
        ),
        row=1, 
        col=2
      )
      # del action, classes


  # agent_data = agent_data[agent_data['experiment_type']==exp]
  del action, classes
  for i, row in enumerate(agent_data.iterrows()):
    if (row[1]['server_round']%15==0 and row[1]['server_round']<=100) or (row[1]['server_round']==100):
      classes = row[1]['classes']
      classes.append(classes[0])
      num_classes = len(classes)
      action = eval(row[1]['action'])
      action.append(action[0])

      sorted_actions = dict()
      # print(action, classes)
      for a, c in zip(action, classes):
          sorted_actions[int(c)] = a
      # print(sorted_actions)
      sorted_actions = {key:value for key, value in sorted(sorted_actions.items(), key=lambda sorted_actions: sorted_actions[0])}
      classes = [str(i) for i in list(sorted_actions.keys())]
      classes.append(classes[0])
      action = list(sorted_actions.values())
      action.append(action[0])

      fig.add_trace(
        go.Scatterpolar(
          r=action,
          theta=classes,
          name=f"a({row[1]['server_round']})",
          showlegend=True,
          marker=dict(size=7, symbol=[2]),
          line=dict(shape='linear', dash='solid'),
          opacity=np.min([0.3+np.mean(eval(row[1]['action'])),1], axis=0),
          legendwidth=1
        ),
        row=1, 
        col=1
      )
      del action, classes

  fig.update_layout(
    autosize=True,
      width=width,
      height=height,
    margin=margin,
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    font=dict(size=14, family="Times New Roman"),
    title_font_family="Times New Roman",
    legend=dict(yanchor="auto", y=0.5, xanchor="left", x=-5),
    paper_bgcolor = "#ffffff",
  )
  
  fig.update_annotations(
    x=left_x,
    y=left_y,
    selector={'text':'Optimal Actions'},    
    font_size=font_size
  )
  fig.update_annotations(
    x=right_x,
    y=right_y,
    selector={'text':'Naive Actions'},    
    font_size=font_size
  )
  # fig.update_layout(title=0.2)
  # fig.update_traces(textposition='top center')

  fig.show()

def plot_accuracy_after_aggregation_local_training_and_finetuning(ntp_metrics, otp_metrics, otp_clone_metrics, figsize=[12, 5]):
    
    rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(1, 2)
    client_ids = ntp_metrics['client_id'].unique().tolist()
    experiment_types = ntp_metrics['experiment_type'].unique().tolist()
    client_ids = np.sort([int(i) for i in client_ids])
    client_ids = [i for i in client_ids]
    for exp in experiment_types[:]:
        print(exp)
        temp_otp_metrics = otp_metrics[(otp_metrics['experiment_type']==exp)]
        temp_otp_clone_metrics = otp_clone_metrics[(otp_clone_metrics['experiment_type']==exp)]
        for client in client_ids:
            temp_ntp_metrics = ntp_metrics[(ntp_metrics['experiment_type']==exp)&(ntp_metrics['client_id']==client)]
            
            
            rounds = len(temp_ntp_metrics)
            acc = [float(i) for i in temp_ntp_metrics['accuracy_on_local_dataset_after_aggregation']]
            axs[0].plot(range(rounds), acc, label=f"Client_{client}_accuracy_on_local_dataset_after_aggregation_{exp}")

            acc = [float(i) for i in temp_ntp_metrics[temp_ntp_metrics['client_id']==client]['accuracy_on_validation_dataset_after_local_training']]
            axs[1].plot(range(rounds), acc, label=f"Client_{client}_accuracy_on_validation_dataset_after_local_training_{exp}")
        
        rounds = len(temp_otp_metrics)
        axs[0].plot(range(rounds), temp_otp_metrics['accuracy_on_local_dataset_after_aggregation'][:rounds], label = f'otp_accuracy_on_local_dataset_after_aggregation_{exp}')
        axs[1].plot(range(rounds), temp_otp_metrics['accuracy_on_validation_dataset_after_local_training'][:rounds], label = f'otp_accuracy_on_validation_dataset_after_local_training_{exp}')

        rounds = len(temp_otp_clone_metrics)
        axs[0].plot(range(rounds), temp_otp_clone_metrics['accuracy_on_validation_dataset_after_aggregation_and_finetuning'][:rounds], label = f'otp_accuracy_on_validation_dataset_after_aggregation_and_finetuning_{exp}')
        axs[1].plot(range(rounds), temp_otp_clone_metrics['accuracy_on_validation_dataset_after_aggregation_and_finetuning'][:rounds], label = f'otp_accuracy_on_validation_dataset_after_aggregation_and_finetuning_{exp}')
        
        axs[0].set_xlabel('Server Round/Finetuning Epoch')
        axs[0].set_ylabel('Accuracy after aggregation and finetuning')
        # axs[0].legend(loc='lower right', fontsize=8)

        axs[1].set_xlabel('Server Round')
        axs[1].set_ylabel('Accuracy after local training')
        # axs[1].legend(loc='lower right', fontsize=8)

def plot_accuracy_after_local_training(ntp_metrics, otp_metrics, otp_clone_metrics, figsize=[12, 5]):
    rcParams['figure.figsize'] = figsize
    client_ids = ntp_metrics['client_id'].unique().tolist()
    experiment_types = ntp_metrics['experiment_type'].unique().tolist()
    client_ids = np.sort([int(i) for i in client_ids])
    client_ids = [i for i in client_ids]
    for exp in experiment_types:
      temp_ntp_metrics = ntp_metrics[ntp_metrics['experiment_type']==exp]
      temp_otp_metrics = otp_metrics[otp_metrics['experiment_type']==exp]
      temp_otp_clone_metrics = otp_clone_metrics[otp_clone_metrics['experiment_type']==exp]
      for client in client_ids:
          acc = [float(i) for i in temp_ntp_metrics[temp_ntp_metrics['client_id']==client]['accuracy_on_local_dataset_after_local_training']]
          plt.plot(acc, label=f"Client_{client}_accuracy_on_local_dataset_after_local_training_{exp}")
      plt.plot(temp_otp_metrics['accuracy_on_local_dataset_after_aggregation'], label = f'accuracy_on_local_dataset_after_local_training_{exp}')
      plt.plot(temp_otp_clone_metrics['accuracy_on_local_dataset_after_aggregation_and_finetuning'], label = f'accuracy_on_local_dataset_after_aggregation_and_finetuning_{exp}')
      plt.xlabel('Server Round/Finetuning Epoch')
      plt.ylabel('Accuracy')
      plt.legend()

def plot_mean_action_distribution(otp, ntp, width=500, height=300, rows=1, cols=1, font_size=15):
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        # subplot_titles=('Optimal vs. Naive Actions'),
        specs=[[{"type": "polar"} for _ in range(1)] for _ in range(1)]
    )

    r = ntp['action'].tolist()
    r.append(r[0])
    theta=[str(i) for i in np.sort(ntp['classes'].tolist())]
    theta.append(theta[0])
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            name=f'Naive Action',
            marker=dict(size=12, color="#475762")
        ),
        row=1, 
        col=1
    )
    
    r = otp['action'].tolist()
    r.append(r[0])
    theta=[str(i) for i in np.sort(otp['classes'].tolist())]
    theta.append(theta[0])
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            name=f'Optimized Action',
            marker=dict(size=12, color="mediumseagreen")
        ),
        row=1, 
        col=1
    )

    fig.update_layout(
        title="Naive vs. Optimized Data Partition Selection",
        title_x=0.06,
        font_size=18,
        autosize=True,
        width=width,
        height=height,
        margin=dict(l=5, r=5, t=50, b=25),
        polar=dict(
            bgcolor="white",
            angularaxis = dict(
                linewidth = 3,
                showline=True,
                linecolor='black',
                gridcolor = "grey"
            ),
            radialaxis = dict(
                visible=True,
                range=[0, 1],
                side = "counterclockwise",
                showline = False,
                linewidth = 3,
                gridcolor = "white",
                gridwidth = 2,
            )
        ),
        showlegend=True,
        font=dict(size=17, 
            family="Times New Roman"),
        title_font_family="Times New Roman",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=-0.13,
        ),
        paper_bgcolor = "#ffffff",
    )
    fig.update_annotations(
        font_size=font_size
    )
    # fig.update_layout(title=0.2)
    fig.write_image("figures/data_selection_naive_optimal.pdf")
    fig.show()

def plot_incremental_performance(summary_df, strategy="FedAvg", font_size=12):
    rcParams['figure.figsize'] = 5.3, 4
    exp = strategy
    # summary_df = summary_df.iloc[:,[1,2]]
    temp_df = summary_df.loc[pd.IndexSlice[exp, :]]
    xticks = ['Naive \nAction', 'Normalized \nAction', 'Weighted Metric \nAction'] # 'Random \nAction'
    x = np.arange(len(xticks))
    y1 = temp_df[temp_df.columns[0]].values[:3]
    # summary_df[summary_df.columns[2]].plot(kind='bar')
    y2 = temp_df[temp_df.columns[1]].values[:3]
    y3 = temp_df[temp_df.columns[2]].values[:3]

    fig, ax = plt.subplots(layout='constrained')
    rects = ax.bar(x, y1.round(1), bottom=[0]*3, width=0.2, label='Naive Clients', color=["#B3C8CF"])
    ax.bar_label(rects, padding=3, fontsize=10)
    rects = ax.bar(x+0.25, y2.round(2), width=0.23, label='Optimized Client', color=["#293d9b"]) # bottom=y2.tolist(), #293d9b
    # ax.bar_label(rects, padding=3, fontsize=8)
    which = [0]
    for index, rect in enumerate(rects):
        if index in which:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.3, height,
                    ha='center', va='bottom', fontsize=10)
    rects = ax.bar(x+0.25, y3.round(2), bottom=y2.round(2), width=0.23, label='Incremental Accuracy', color=["#219B9D"]) # bottom=y2.tolist(), #12a79d
    ax.bar_label(rects, padding=3, fontsize=10)
    ax.set_xticks(x+0.12, xticks, fontsize=10)
    ax.legend(loc="upper left", ncols=1, fontsize=9)
    ax.set_ylim(0, 70)
    plt.ylabel('Validation Accuracy', fontsize=10)
    # plt.xlabel('Training Dataset Selection', fontsize=8)
    # plt.title(f'Validation Accuracy', fontsize=8)
    plt.show()
    fig.savefig('figures/motivation.pdf')

def plot_incremental_performance_barh(summary_df, strategy="FedAvg", font_size=12):
    rcParams['figure.figsize'] = 5.4, 4
    exp = strategy
    # summary_df = summary_df.iloc[:,[1,2]]
    temp_df = summary_df.loc[pd.IndexSlice[exp, :]]
    yticks = ['Naive \nAction', 'Normalized \nAction', 'Weighted Metric \nAction'] # 'Random \nAction'
    y = np.arange(len(yticks))
    y1 = temp_df[temp_df.columns[0]].values[:3]
    # summary_df[summary_df.columns[2]].plot(kind='bar')
    y2 = temp_df[temp_df.columns[1]].values[:3]
    y3 = temp_df[temp_df.columns[2]].values[:3]

    fig, ax = plt.subplots(layout='constrained')
    rects = ax.barh(y, y1.round(1), height=0.25, left=[0]*3, label='Naive Clients', color=["#475762"])
    ax.bar_label(rects, padding=3, fontsize=14)
    rects = ax.barh(y+0.25, y2.round(2), height=0.25, label='Optimized Client', color=["#C2E8CE"]) # bottom=y2.tolist(), #293d9b
    # ax.bar_label(rects, padding=3, fontsize=8)
    which = [0]
    for index, rect in enumerate(rects):
        if index in which:
            height = rect.get_width()
            ax.text(height+5, 0.325, height,
                    ha='center', va='bottom', fontsize=14)
    rects = ax.barh(y+0.25, y3.round(2), left=y2.round(2), height=0.25, label='Incremental Accuracy', color=["#00AD7C"]) # bottom=y2.tolist(), #12a79d
    # print(rects)
    ax.bar_label(rects, padding=3, fontsize=14)
    ax.set_yticks(y+0.12, yticks, fontsize=18)
    ax.tick_params(axis='y', direction='in', pad=-125)
    # ax.legend(loc="upper right", ncols=2, fontsize=15)
    ax.set_xlim(0, 80)
    ax.invert_yaxis()
    plt.xlabel('Validation Accuracy', fontsize=18)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('right')
    ax.yaxis.set_tick_params(labelright=True, labelleft=False)
    
    # plt.ylabel('Training Dataset Selection', fontsize=8)
    # plt.title(f'Validation Accuracy', fontsize=8)
    plt.show()
    fig.savefig('figures/motivation.pdf')