from attack_pattern_inference import get_embedding, attack_pattern_inference
from technique_evaluation import evaluation
import pandas as pd
import pickle
import numpy as np

def infer_report(vo_pair_embeddings):
    with open('data_source/tacticTruth_srl_ontology_feature_6', 'rb') as f:
        df = pickle.load(f)
    df = df.iloc[:2, :]

    pred_result = df
    pred_result['g_t'] = ""
    pred_result['s_t'] = ""
    pred_result['vo_t'] = ""
    pred_result['g_vo_t'] = ""
    pred_result['s_vo_t'] = ""
    pred_result['g_s_vo_t'] = ""
    pred_result['must_ttp'] = ""
    pred_result['rm_ttp'] = ""
    # df_titlefile = ['Text' 'list' 'Group' 'Soft' 'v_o']
    # pred_result =  ['Text', 'list', 'Group', 'Soft', 'v_o', g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t]

    for i, row in df.iterrows():
        print("report no = ", i)
        # g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t = attack_pattern_inference(row['Group'], row['Soft'], row['v_o'], row['Text'], row['srl'], row['tactic_list])
        g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t, must_ttp, rm_ttp = attack_pattern_inference(vo_pair_embeddings, row['Group'], row['Soft'], row['v_o'], row['Text'], row['srl'], row['list_tactic'])
        pred_result.iloc[i]['g_t'] = g_t
        pred_result.iloc[i]['s_t'] = s_t
        pred_result.iloc[i]['vo_t'] = vo_t
        pred_result.iloc[i]['g_vo_t'] = g_vo_t
        pred_result.iloc[i]['s_vo_t'] = s_vo_t
        pred_result.iloc[i]['g_s_vo_t'] = g_s_vo_t
        pred_result.iloc[i]['must_ttp'] = must_ttp
        pred_result.iloc[i]['rm_ttp'] = rm_ttp

   
    with open('pred_result/pred_result.pickle', 'wb') as f:
        pickle.dump(pred_result, f)
        print("Data written to the file successfully.")

    with open('pred_result/pred_result.pickle', 'rb') as f:
        #pickle.dump(pred_result, f)
        loaded_data = pickle.load(f)
        print(loaded_data)
    f.close()
    

if __name__=='__main__':
    vo_pair_embeddings = get_embedding()
    infer_report(vo_pair_embeddings)
    Technique_name = "Technique_name"
    evaluation(Technique_name) # Technique_name: 113 techniques/ttpdrill_Technique_name: 106 techniques
