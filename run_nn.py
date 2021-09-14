
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



import nni

from models import Model1

PARAMS = {}

def set_params(default=None):
    global PARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--nni", type=int)
    parser.add_argument("--device_num", type=int)
    args = parser.parse_args()
    if args.nni:
        PARAMS = nni.get_next_parameter()
    elif default:
        PARAMS = default
    else:
        PARAMS = {"learning_rate": 0.001,
                  "batch_size": 64,
                  "dropout": 0.1,
                  "optimizer": "adam",
                  "activation": "relu",
                  "regularization": 0,
                  "layer_1": 300,
                  "layer_2": 100,
                  "train_frac": 0.8,
                  "test_frac": 0.2,
                  "epochs": 100
                  }

    PARAMS['nni'] = args.nni
    PARAMS['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PARAMS['batch_size'] = int(PARAMS['batch_size'])
    PARAMS['layer_1'] = int(PARAMS['layer_1'])
    PARAMS['layer_2'] = int(PARAMS['layer_2'])
    PARAMS['epochs'] = int(PARAMS['epochs'])

def plot_graph(values, x_label, y_label, label="", save_path=None, title=''):
    plt.plot(values, "-", label=label, color="b")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', which="both")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()
def plot_graph2(train_loss, train_auc, val_loss, val_auc, name):
    fig, axs = plt.subplots(1, 2, figsize=(8,3))
    axs[0].plot(train_loss)
    axs[0].grid(color="w")
    axs[0].set_facecolor('xkcd:light gray')
    axs[0].set_title("Train Loss")
    axs[0].set_yticks(np.arange(0.1, 0.35, 0.05))
    # axs[1, 0].plot(train_auc)
    # axs[1, 0].grid(color="w")
    # axs[1, 0].set_facecolor('xkcd:light gray')
    # axs[1, 0].set_title("Train F1")
    # axs[1, 0].sharex(axs[0, 0])
    # axs[1, 0].set_yticks(np.arange(0.8, 1, 0.05))
    axs[1].grid(color="w")
    axs[1].set_facecolor('xkcd:light gray')
    axs[1].plot(val_loss)
    axs[1].set_title("Test Loss")
    axs[1].set_yticks(np.arange(0.1, 0.35, 0.05))
    # axs[1, 1].plot(val_auc)
    # axs[1, 1].grid(color="w")
    # axs[1, 1].set_facecolor('xkcd:light gray')
    # axs[1, 1].set_title("Test F1")
    # axs[1, 1].set_yticks(np.arange(0.8, 1, 0.05))
    fig.tight_layout()
    plt.savefig(name + ".png")

def get_data():
    df_data = pd.read_csv("dataset_phishing.csv")

    df_data['target'] = pd.get_dummies(df_data['status'])['legitimate'].astype('int')
    df_data.drop('status',axis = 1, inplace=True)

    return df_data

def plot_corr(df_data):

    likely_cat = {}
    for var in df_data.iloc[:,1:].columns:
        likely_cat[var] = 1.*df_data[var].nunique()/df_data[var].count() < 0.002

    num_cols = []
    cat_cols = []
    for col in likely_cat.keys():
        if (likely_cat[col] == False):
            num_cols.append(col)
        else:
            cat_cols.append(col)



    corr = df_data[num_cols].corr()

    fig = plt.figure(figsize=(12,12),dpi=80)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0,
                square=True, linewidths=.5)
    plt.title('Correlation of Numerical(Continous) Features', fontsize=15,font="Serif")
    plt.show()


    df_distr =df_data.groupby('target')[num_cols].mean().reset_index().T
    df_distr.rename(columns={0:'0_Label',1:"1_Label"}, inplace=True)

    #plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='w'
    ax = df_distr[1:-3][['0_Label','1_Label']].plot(kind='bar', title ="Distribution of Average values across Target", figsize=(12, 8), legend=True, fontsize=12)
    ax.set_xlabel("Numerical Features", fontsize=14)
    ax.set_ylabel("Average Values", fontsize=14)
    #ax.set_ylim(0,500000)
    plt.show()



    sns.catplot("page_rank", hue="target", data=df_data, kind="count",
                palette={1:"green", 0:"blue"} ,height=5.0, aspect=11.7/8.27 )
    plt.show()


def create_external_test(df_data):
    X = df_data.iloc[:, :-1]
    y = df_data['target']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=PARAMS['test_frac'])
    train_x.to_csv('train_x_data.csv')
    train_y.to_csv('train_y_data.csv')
    test_x.to_csv('external_test_x_data.csv')
    test_y.to_csv('external_test_y_data.csv')

def get_train_data():
    X = pd.read_csv('train_x_data.csv', index_col=0)
    y = pd.read_csv('train_y_data.csv', index_col=0)
    df_data = pd.concat([X, y], axis=1)
    return df_data

def split(df_data):
    X= df_data.iloc[: , 1:-1]
    y= df_data['target']

    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42, train_size=PARAMS['train_frac'])


    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_x.values)
    train_x_norm = scaler.transform(train_x.values)
    test_x_norm = scaler.transform(test_x.values)


    return train_x_norm,test_x_norm,train_y,test_y
def prepare_data(train_x,test_x,train_y,test_y):




    x_tensor =  torch.from_numpy(train_x).float().to(PARAMS['device'])
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float().to(PARAMS['device'])
    xtest_tensor =  torch.from_numpy(test_x).float().to(PARAMS['device'])
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float().to(PARAMS['device'])


    y_tensor = y_tensor.unsqueeze(1)
    train_ds = TensorDataset(x_tensor, y_tensor)
    train_dl = DataLoader(train_ds, batch_size=PARAMS['batch_size'])


    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    test_loader = DataLoader(test_ds, batch_size=PARAMS['batch_size'])

    return train_dl, test_loader

def train_model(model, train_dl, test_loader, test_y, train_y):
    

    model.to(PARAMS['device'])

    loss_func = nn.BCELoss()
    learning_rate = PARAMS['learning_rate']
    regularization = PARAMS['regularization']
    if PARAMS['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization)

    epochs = PARAMS['epochs']


    model.train()
    train_loss_list = []
    train_f1_list = []
    test_loss_list = []
    test_f1_list = []
    for epoch in range(epochs):
        loss_values = []
        y_pred_list = []
        print("Epoch", epoch)
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = y_pred.tolist()
            y_pred_list = y_pred_list + y_pred

            loss_values.append(loss.item())



        train_loss_list.append(sum(loss_values) / len(loss_values))
        test_f1, test_loss = eval_model(model, test_loader, test_y, loss_func)
        test_f1_list.append(test_f1)
        test_loss_list.append(test_loss)

        y_true_train = train_y.values.ravel()
        y_pred_for_each_threshold = [round_by_threshold(y_pred_list, t) for t in np.arange(0, 1.005, 0.01)]
        f1_scores = [f1_score(y_true_train, y_pred_list) for y_pred_list in y_pred_for_each_threshold]
        print("best threshold is ", f1_scores.index(max(f1_scores)))
        train_f1 = max(f1_scores)
        train_f1_list.append(train_f1)
    print('Last iteration loss value: '+str(loss.item()))
    eval_model(model, test_loader, test_y, loss_func, final=True)


    plot_graph2(train_loss_list, train_f1_list, test_loss_list, test_f1_list, 'results2')


def eval_model(model, test_loader, test_y, loss_func, final=False):

    y_pred_list = []
    test_loss = []
    model.eval()
    loss_values = []
    with torch.no_grad():
        for xb_test, yb_test in test_loader:
            y_pred = model(xb_test)
            loss = loss_func(y_pred, yb_test)
            loss_values.append(loss.item())

            y_pred = y_pred.tolist()
            y_pred_list = y_pred_list + y_pred
        test_loss = sum(loss_values) / len(loss_values)




    y_true_test = test_y.values.ravel()
    y_pred_for_each_threshold = [round_by_threshold(y_pred_list, t) for t in np.arange(0, 1.005, 0.01)]
    f1_scores = [f1_score(y_true_test, y_pred_list) for y_pred_list in y_pred_for_each_threshold]
    print("best threshold is ", f1_scores.index(max(f1_scores)))
    test_f1 = max(f1_scores)
    print("F1 Score of the Model :\t"+str(test_f1))
    if PARAMS['nni'] and final:
        nni.report_final_result(test_f1)
    return test_f1, test_loss

    
def round_by_threshold(values, threshold):
    return [1 if value >= threshold else 0 for value in values]

def params_from_nni_results(file, n_rows=10):
    result_df = pd.read_csv(file, header=0)
    result_df.sort_values(by=['reward'], inplace=True, ascending=False)
    del result_df["trialJobId"]
    del result_df["intermediate"]
    del result_df["reward"]
    first_n_rows = result_df[0:n_rows]
    params_list = [{} for i in range(n_rows)]
    for i in range(n_rows):
        for j in first_n_rows.columns:
            params_list[i][j] = int(first_n_rows.iloc[i][j]) if type(first_n_rows.iloc[i][j]) is np.int64 else first_n_rows.iloc[i][j]
    return params_list

def run_best_paprams():
    for params in params_from_nni_results('results_nni - Copy.csv', 1):
        set_params(params)
        print(PARAMS)
        df_all_data = get_data()
        # plot_corr(df_data)
        create_external_test(df_all_data)
        df_data = df_all_data
        train_x, test_x, train_y, test_y = split(df_data)
        train_dl, test_loader = prepare_data(train_x, test_x, train_y, test_y)
        model = Model1(train_x.shape[1], PARAMS)
        train_model(model, train_dl, test_loader, test_y, train_y)
        print("finish run")
    print("FINISH")

if __name__ == '__main__':
    run_best_paprams()
    # set_params()
    # print(PARAMS)
    # df_all_data = get_data()
    # # plot_corr(df_data)
    # create_external_test(df_all_data)
    # df_data = get_train_data()
    # train_x, test_x, train_y, test_y = split(df_data)
    # train_dl, test_loader = prepare_data(train_x, test_x, train_y, test_y)
    # model = Model1(train_x.shape[1], PARAMS)
    # train_model(model, train_dl)
    # eval_model(model, test_loader, test_y)
    # print("FINISH")