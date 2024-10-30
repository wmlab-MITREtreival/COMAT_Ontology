import pickle
#from sklearn.metrics import classification_report,multilabel_confusion_matrix
from sklearn import metrics
import pandas as pd
import torch

def evaluation_save(y_true, y_pred, name, technique_name):
    m_columns=["true_negative", "false_positive", "false_negative", "true_positive"]
    result = metrics.classification_report(torch.FloatTensor(y_true), torch.FloatTensor(y_pred),  output_dict=True, target_names=technique_name)
    result_df = pd.DataFrame(result).transpose()
    result_df.to_csv("pred_result/"+name+".csv")

    result_matrix_df=pd.DataFrame(metrics.multilabel_confusion_matrix(y_true,y_pred).reshape(-1, 4), columns=m_columns)
    result_matrix_df.to_csv("pred_result/m_"+name+".csv")

def generate_ttplst(technique_name, ttpid):
    ttplst=[0]*len(technique_name)
    for i, ttp in enumerate(technique_name):
        if ttp.lower() in ttpid:
            ttplst[i]=1
    return ttplst

def evaluation(Technique_name):
    with open('data_source/'+Technique_name+'.txt', 'rb') as f:
        technique_name = pickle.load(f)
    
    with open('pred_result/pred_result.pickle', 'rb') as f:
        df = pickle.load(f)
    f.close()

    vo_pred = []
    y_true=[]
    # titlefile = ['Text', 'list', 'Group', 'Soft', 'v_o', g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t]
    for i, row in df.iterrows():
        vo_pred.append(generate_ttplst(technique_name, row['vo_t']))
        y_true.append(row['list'])

    evaluation_save(y_true, vo_pred, "vo_pred", technique_name)
