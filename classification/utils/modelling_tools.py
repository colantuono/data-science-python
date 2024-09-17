import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import * 
from sklearn.metrics import *
from sklearn.metrics import *
from sklearn.model_selection import *


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
    stat_ks, ks_val = ks_2samp(
        dataset[dataset['target']==0],
        dataset[dataset['target']==1]
    )
    
    print(f'KS: {stat_ks[1]:.4f}')
    print(f'p-valor: {ks_val[1]:.4f}')
    
    return stat_ks, ks_val

def kstable(dataset, target=None, prob=None, print=False):
    dataset['target0'] = 1 - dataset[target]
    dataset['bucket'] = pd.qcut(dataset[prob], 10)
    grouped = dataset.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events'] = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by='min_prob', ascending=False).reset_index(drop=True)
    kstable['event_rate'] = (kstable['events'] / dataset[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable['nonevents'] / dataset['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_event_rate'] = (kstable['events'] / dataset[target].sum()).cumsum()
    kstable['cum_nonevent_rate'] = (kstable['nonevents'] / dataset['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_event_rate'] - kstable['cum_nonevent_rate'], 3) * 100 
    
    if print:
        print(kstable)
        
    return kstable
    
 
    
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
    
def plot_learning_curve(estimator, X, y, title='', cv=5,
                        n_jobs=1, scorer=None, ylim=None, seed=123,
                        train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    
    if scorer is None:
        scorer=make_scorer(accuracy_score)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scorer, random_state=seed,
        n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1, ddof=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1, ddof=1)
    
    train_scores_last4 = train_scores[-4:,:] 
    test_scores_last4 = test_scores[-4:,:] 

    train_avg = train_scores_last4.mean()
    train_std = train_scores_last4.std(ddof=1)
    train_n = len(train_scores_last4[-1])
    test_avg = test_scores_last4.mean()
    test_std = test_scores_last4.std(ddof=1)
    test_n = len(test_scores_last4[-1])
    
    txt_lst = [
        f'{scorer}: Score',
        f"Train Score: {train_avg:.3f} ({train_std:.3f})"        
        f"Test Score: {test_avg:.3f} ({test_std:.3f})"        
               ]
    
    for line in txt_lst:
        print(line)
    print('-'*20)
    
    plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=.2, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=.2, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', linewidth=1, color='r',
    label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', linewidth=1, color='g',
    label='Cross-Validation Score')
    
    plt.legend(loc='best');
    return plt, txt_lst, [train_avg, train_std, train_n], [test_avg, test_std, test_n]
    
def t_test_S(x0, x_hat, S, n, alpha=.5):
    t_obs = (x_hat - x0)*np.sqrt(n)/S
    
    print(f'Probabilidade da metrica calculada para base de teste pertencer à validação de treino:')
    p_value = 1-t.cdf(np.abs(t_obs), n-1)
    print(f'P-Valor = {p_value:.4f} ({t_obs:.3f})')
    
    
    if p_value > alpha:
        print(f'O valor {x0:.4f} tem probabilidade de {p_value:.2%} de pertencer a validação por flutuação estatística, e com IC={1-alpha}, não rejeito H0')
    else:
        print(f'O valor {x0:.4f} tem probabilidade de {p_value:.2%} de pertencer a validação por flutuação estatística, e com IC={1-alpha}, rejeito H0')

    return p_value

def permutation_test(arr_model1, arr_model2, alpha=.5):
    arr_model1 = np.array(arr_model1)
    arr_model2 = np.array(arr_model2)

    avg_arr1 = arr_model1.mean()
    avg_arr2 = arr_model2.mean()

    mean_diff = avg_arr1 - avg_arr2 

    full_arr = np.concatenate([arr_model1, arr_model2])

    mean_lst = []
    
    for i in range(10000):
        avg1 = np.random.choice(full_arr, size=len(arr_model1), replace=True).mean()
        avg2 = np.random.choice(full_arr, size=len(arr_model2), replace=True).mean()

        mean_lst.append(avg1-avg2)
        
    if mean_diff > 0:
        p_val = np.sum(np.array(mean_lst) >= mean_diff)/i
    else:
        p_val = np.sum(np.array(mean_lst) <= mean_diff)/i
    
    print(f'Diferença entre as médias: {avg1:.4f}-{avg2:.4f}={mean_diff:.4f}')
    
    if p_val > alpha:
        print(f'Os modelos parecem produzir o mesmo resultado com IC={1-alpha} (não rejeito H0)')    
    else:
        print(f'Os modelos parecem produzir resultados diferentes com IC={1-alpha} (rejeito H0)')      
    
    return p_val, mean_lst, mean_diff

def plot_pts(data, p_val, mean_diff, title=''):
    ni, xi = np.histogram(data, bins=17)
    
    fi = ni/np.sum(ni)
    
    delta_xi = xi[2] - xi[1]
    xi = xi + np.abs(delta_xi)/2
    plt.plot([mean_diff, mean_diff], [0, np.max(ni)*0.85])
    
    if p_val <= .05:
        alinhamento='center'
    elif mean_diff < 0:
        alinhamento='right'
    else:
        alinhamento='left'
        
    plt.text(mean_diff, np.max(ni)*.9, f'p_val={p_val:.3f}')
    plt.bar(xi[:-1], ni, label='Data', edgecolor='k', color='g', width=np.abs(xi[2]-xi[1]*.9))
    plt.title(f'Permutation Test - {title}')
    plt.ylabel('Counts')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-3,3));
    return plt
        
def psi(score_init, score_new, num_bins=10, modo='fixo'):
    eps = 1e-4
    
    ## prepara os bins
    min_val = min(min(score_init), min(score_new))
    max_val = max(max(score_init), max(score_new))
    
    if modo == 'fixo':
        bins = [min_val + (max_val - min_val)*(i) / num_bins for i in range(num_bins+1) ]
    elif modo == 'quantil':
        bins = pd.qcut(score_init, q=num_bins, retbins=True)[1] ## cria quantis baseado na população inicial
    else:
        raise ValueError(f'Modo {modo} não reconhecido, deve ser {"fixo"} ou {"quantil"}')
    
    bins[0] = min_val - eps # corrige o limite inferior
    bins[-1] = max_val + eps # corrige o limite superior
    
    ## bucketiza a população inicial e conta a amostra dentro de cada bucket
    bins_init = pd.cut(score_init, bins=bins, labels=range(1, num_bins+1))
    df_init = pd.DataFrame({'Inicial': score_init, 'bins': bins_init})
    grp_init = df_init.groupby('bins').count()
    grp_init['Percentual Inicial'] = grp_init['Inicial'] / sum(grp_init['Inicial'])
  
    ## bucketiza a nova população e conta a amostra dentro de cada bucket
    bins_new = pd.cut(score_new, bins=bins, labels=range(1, num_bins+1))
    df_new = pd.DataFrame({'Nova': score_new, 'bins': bins_new})
    grp_new = df_new.groupby('bins').count()
    grp_new['Percentual Nova'] = grp_new['Nova'] / sum(grp_new['Nova'])

    ## compara os bins 
    psi_df = grp_init.join(grp_new, on='bins', how='inner')
    
    # Adiciona um valor pequeno quando a porcentagem for 0
    psi_df['Percentual Inicial'] = psi_df['Percentual Inicial'].apply(lambda x: eps if x == 0 else x)
    psi_df['Percentual Nova'] = psi_df['Percentual Nova'].apply(lambda x: eps if x == 0 else x)
    
    ## calcula o PSI
    psi_df['PSI'] = psi_df['Percentual Inicial'] - psi_df['Percentual Nova'] * np.log(psi_df['Percentual Inicial'] / psi_df['Percentual Nova'])
    
    return psi_df['PSI'].values

    
     