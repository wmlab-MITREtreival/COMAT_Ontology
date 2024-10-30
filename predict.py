import time
import torch
import numpy as np
from filter_bert_usage import topic_classifier
from Preprocess.preprocess import query_node_extract
from attack_pattern_inference import get_embedding, attack_pattern_inference

# Load document for processing
with open('input.txt', 'r', encoding="utf-8") as f:
    doc = f.read()
print("Original Document:")
print(doc)

# Step 1: Preprocessing
start_time = time.time()
print("===========================================Start Preprocessing===========================================")
doc = topic_classifier(doc)
print("Preprocessed Document:", doc)
print("Preprocessing Time:", time.time() - start_time, "seconds")
print("===========================================Finish Preprocessing===========================================")

# Step 2: Extract Query Nodes
start_time = time.time()
print("===========================================Start Extracting Query nodes===========================================")
group, software, all_node, v_o = query_node_extract(doc)
print("Extracted Nodes - Group:", group, "Software:", software, "Verb-Object Pairs:", v_o)
print("Query Node Extraction Time:", time.time() - start_time, "seconds")
print("===========================================Finish Extracting Query nodes===========================================")

# Step 3: Query COMAT
start_time = time.time()
print("===========================================Start Query COMAT===========================================")

# Ensure verb-object pairs are limited to a maximum of 2 per entry
for index, feature_v_o in enumerate(v_o):
    if np.shape(feature_v_o)[0] >= 2:
        v_o[index] = feature_v_o[:2]
    elif np.shape(feature_v_o)[0] < 2:
        v_o[index].append(" ")

# Initialize embeddings and run inference
vo_pair_embeddings = get_embedding()
COMAT_result = attack_pattern_inference(vo_pair_embeddings, group, software, v_o, doc, all_node)

# Separate results for analysis
l1_ans_cor = COMAT_result[0] + COMAT_result[1] + COMAT_result[2]
l2_ans_cor = COMAT_result[3] + COMAT_result[4]
l3_ans_cor = COMAT_result[5]
must_ttp = COMAT_result[6]
must_not_ttp = COMAT_result[7]

# Placeholder for storing results in the technique list format (update this as per your actual Technique_name list)
Technique_name = ["technique1", "technique2", "technique3"]  # replace with the full list
temp1 = torch.zeros((len(Technique_name)))
temp2 = torch.zeros((len(Technique_name)))
temp3 = torch.zeros((len(Technique_name)))
temp4 = torch.zeros((len(Technique_name)))
temp5 = torch.zeros((len(Technique_name)))

# Map results to Technique_name
for l1_ans in l1_ans_cor:
    if l1_ans.upper() in Technique_name:
        temp1[Technique_name.index(l1_ans.upper())] = 1
for l2_ans in l2_ans_cor:
    if l2_ans.upper() in Technique_name:
        temp2[Technique_name.index(l2_ans.upper())] = 1
for l3_ans in l3_ans_cor:
    if l3_ans.upper() in Technique_name:
        temp3[Technique_name.index(l3_ans.upper())] = 1
for l4_ans in must_ttp:
    if l4_ans.upper() in Technique_name:
        temp4[Technique_name.index(l4_ans.upper())] = 1
for l5_ans in must_not_ttp:
    if l5_ans.upper() in Technique_name:
        temp5[Technique_name.index(l5_ans.upper())] = 1


print("Total Query COMAT Time:", time.time() - start_time, "seconds")
print("===========================================Finish Query COMAT===========================================")

