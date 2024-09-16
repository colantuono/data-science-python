import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import * 
from sklearn.metrics import *


## configurações

np.set_printoptions()
np.set_printoptions(precision=4, suppress=True)

pd.options.display.float_format = '{:.4f}'.format

from matplotlib import rcParams

plt_params = {
    'figure.figsize':(6,4),
    'figure.dpi':75,
    'font.size':16,
    'font.family':'serif',
    'xtick.labelsize':14,
    'ytick.labelsize':14,
    'axes.labelsize':14,
    'legend.fontsize':10,
    'lines.linewidth':12,
    'grid.color':'gray',
    'savefig.bbox':'tight',
    'savefig.dpi':1000,
    'savefig.transparent':False,
}

rcParams.update(plt_params)

import warnings
warnings.filterwarnings('ignore')

seed = 123


def model_eval(model, X_test, y_test):
    '''Retorna a curva ROC, classificação e Probabilidade'''
    y_score = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_score[:,1], average=None)
    precision, recall, threshold_PRCurve = precision_recall_curve(y_test, y_score[:, 1])
    auc_pr = auc(recall, precision)
    
    y_hat = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    
    print(f'Accuracia: {accuracy:.2%}')
    print(f'AUC-ROC Score médio: {roc_auc:.4f}')
    print(f'AUC-PR Score médio: {auc_pr:.4f}')
    return y_hat, y_score, roc_auc, auc_pr

def plot_curva_roc(fper, tper, auc, nome_modelo=''):
    '''função para gráfico da cruva ROC'''
    if auc > .85:
        cor = [0, .7, .4]
    elif auc < .85 and auc > .75:
        cor = 'orange'
    else: 
        cor = 'red'
    
    plt.plot(fper, tper, color=cor, linewidth=1, label=f'AUC-ROC = {auc:.4f}')
    plt.plot([0,1], [0,1], color='gray', linewidth=1, linestyle='--', label = f'AUC-ROC = 0.5')
    plt.xlabel(f'False Positive Rate')
    plt.ylabel(f'True Positive Rate')
    plt.title(f'Curva ROC - {nome_modelo}')
    plt.legend();
    return plt
    
    
def plot_curva_pr(recall, precision, auc_pr, nome_modelo=''):
    '''função para gráfico da cruva ROC'''
    AP = np.sum(np.diff(recall[:-1])*precision[1:-1])
    if auc_pr > .85:
        cor = [0, .7, .4]
    elif auc_pr < .85 and auc_pr > .75:
        cor = 'orange'
    else: 
        cor = 'red'
    
    plt.plot(recall[1:-1], precision[1:-1], color=cor, linewidth=1, label=r'AUC-PR = %0.4f' % auc_pr)
    plt.plot([0,1], [AP, AP], color='gray', linewidth=1, linestyle='--', label = f'AUC-PR = {AP:.4f}')
    plt.xlabel(f'Precision')
    plt.ylabel(f'Recall')
    plt.title(f'Curva PR - {nome_modelo}')
    plt.ylim([0,1])
    plt.legend();
    return plt

def ks(dataset):
    '''calcula o KS score e o p-valor'''
    nb_stat_ks, nb_ks_val = ks_2samp(
        dataset[dataset['target']==0],
        dataset[dataset['target']==1]
    )
    
    print(f'KS: {nb_stat_ks[1]:.4f}')
    print(f'p-valor: {nb_ks_val[1]:.4f}')
    
    return nb_stat_ks, nb_ks_val
    
def plot_separacao_classes(dataset, colunas=[''], titulos=[''], modelo=''):
    '''plota o gráfico de Sepração de Classes'''
    x_div = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    x_axis = np.arange(0.1, 1.1, .1)
    
    class0_lst = []
    class1_lst = []
    
    plt.figure(figsize=(len(colunas)*8, 5))
    for i, coluna in enumerate(colunas):
        plt.subplot(1, len(colunas), i+1)
        class1 = np.histogram(dataset[dataset['target'] == 1][coluna].values,
                bins = np.arange(0.0, 1.1, .1))[0]
        class1 = class1 / np.sum(class1)
        class1_lst.append(class1)
    
        class0 = np.histogram(dataset[dataset['target'] == 0][coluna].values,
                bins = np.arange(0.0, 1.1, .1))[0]
        class0 = class0 / np.sum(class0)
        class0_lst.append(class0)
        
        bar1 = plt.bar (x_axis+0.02, class1, width=.035, label='Classe 1',
                        color='r', edgecolor='k', alpha=.6)
        bar0 = plt.bar (x_axis-0.02, class0, width=.035, label='Classe 0',
                        color='b', edgecolor='k', alpha=.6)
        plt.xticks(x_axis, x_div)
        plt.ylim([0,1])

        plt.legend()
        plt.xlabel('Faixa de Probabilidade')
        plt.title(f'Separação de Classes - {titulos[i]}')
    return plt, class0_lst, class1_lst
    
    
def pracc_metrics(y_test, y_prob, threshold_lst=np.arange(0.1, 1.1, .1)):
    metrics_dic = {
        'percentil':[],
        'tp':[], 'tn':[], 'fp':[], 'fn':[],
        'precision':[], 'recall':[],
        'accuracy':[],
    }
    
    for thresh_cut in threshold_lst:
        y_pred = (y_prob[:,1] >= thresh_cut).astype(int)
        metrics_dic['tp'].append( ((y_pred==y_test) & (y_test==1)).sum() )
        metrics_dic['tn'].append( ((y_pred==y_test) & (y_test==0)).sum() )
        metrics_dic['fp'].append( ((y_pred!=y_test) & (y_test==0)).sum() )
        metrics_dic['fn'].append( ((y_pred!=y_test) & (y_test==1)).sum() )
        metrics_dic['percentil'].append( np.round(thresh_cut, 2) )
        metrics_dic['precision'].append( np.round(precision_score(y_test, y_pred, average='macro'), 4) )
        metrics_dic['recall'].append( np.round(recall_score(y_test, y_pred, average='macro'), 4) )
        metrics_dic['accuracy'].append( np.round(accuracy_score(y_test, y_pred), 4) )

    return pd.DataFrame(metrics_dic)
    
    