from allennlp.predictors import Predictor
from allennlp_models import structured_prediction
from nltk.stem.wordnet import WordNetLemmatizer
from Preprocess.Ontology_Feature.Dict import load_lists,fpath
from Preprocess.Ontology_Feature.Dict_IoC import iocs
from Preprocess.Ontology_Feature.SVO_Extraction import findSVOs, nlp
from Preprocess.Ontology_Feature.Passive2Active import pass2acti
import re
import copy
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


predictor = Predictor.from_path("structured-prediction-srl-bert.2020.12.15.tar.gz")
#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp/models/semantic_role_labeler-2018.04.05.tar.gz")

'''
 * V_O_parser()-Get all verb object pair in sentence in sentence list
 * @sentence_list: List of sentences
 * @return: All verb object pair and SRL result
'''
def V_O_parser(sentence_list): #this is for parsing threat action(Verb-Object) from a report
    my_vo_triplet = []
    all_nodes = []
    all_sentences_nodes=[]
    main_verbs = load_lists(fpath)['verbs']
    main_verbs = main_verbs.replace("'", "").strip('][').split(', ')
    for sent_i in range(len(sentence_list)):
        print(len(sentence_list[sent_i].split(' ')))
        if(len(sentence_list[sent_i].split(' '))>500):
            continue
        predictions = predictor.predict(sentence_list[sent_i])
        
        
        lst = []
        nodes = []
        for k in predictions['verbs']:
            if k['description'].count('[') > 1:
                lst.append(k['description'])
        for jj in range(len(lst)):
            nodes.append([])
            for j in re.findall(r"[^[]*\[([^]]*)\]", lst[jj]):
                nodes[jj].append(j)
        print("*****sentence:",sentence_list[sent_i],'*****nodes: ',nodes)
        del lst,predictions
        for lis_ in nodes:
            for indx in range(len(lis_)):
                if lis_[0].split(":", 1)[0].lower().strip() == "v" and lis_[0].split(":", 1)[1].lower().strip() in main_verbs:
                    n = len(lis_)
                    for j in range(1, len(lis_)):
                        if lis_[j].split(":", 1)[0].lower() != "v":
                            if len(iocs.list_of_iocs(lis_[j].split(":", 1)[1])) > 0:
                                lis_.insert(0, " ARG-NEW: *")
        maxlength = 0
        if nodes:
            maxlength = max((len(i) for i in nodes))
        if nodes == [] or maxlength < 3:
            print("****DP SVO****")
            tokens = nlp(sentence_list[sent_i])
            svos = findSVOs(tokens)
            if svos:
                for sv in range(len(svos)):
                    if len(svos[sv]) == 3:
                        print('Dependency SVO(s):', ["ARG0: " + svos[sv][0], "V: " + svos[sv][1], "obj: " + svos[sv][2]])
                        nodes.append(["ARG0: " + svos[sv][0], "V: " + svos[sv][1], "obj: " + svos[sv][2]])
            print("Dependency-SVO added nodes: ", nodes)

            print("****Naive SVO****")
            breakers = []
            subj, obj = '', ''
            doc = nlp(sentence_list[sent_i])
            for token in doc:
                if token.pos_ == 'VERB':
                    breakers.append(str(token))
            if len(breakers) != 0:
                for vb in breakers:
                    subj = "subj: " + sentence_list[sent_i].split(vb)[0]
                    obj = "obj: " + sentence_list[sent_i].split(vb)[1]
                    vrb = "v: " + vb
                    lst = []
                    lst.append(subj)
                    lst.append(vrb)
                    lst.append(obj)
                    nodes.append(lst)
            print("Naive Nodes: ", nodes)

        if nodes != []:
            zero_dunplicate_removed = []
            for i in nodes:
                
                zero_dunplicate_removed.append(list(dict.fromkeys(i)))
            no_zero_nodes = []
            for i in zero_dunplicate_removed:
                if '.' in i:
                    i.remove('.')
                    no_zero_nodes.append(i)
                else:
                    no_zero_nodes = zero_dunplicate_removed

            no_zero_nodes_plus_3 = []
            for i in no_zero_nodes:
                if len(i) > 2:
                    no_zero_nodes_plus_3.append(i)
            #lammarizer
            for i in range(len(no_zero_nodes_plus_3)):
                
                for index, item in enumerate(no_zero_nodes_plus_3[i]):
                    if item.split(': ')[0] == 'V' or item.split(': ')[0] == 'v':
                        word = item.split(': ')[1]
                        no_zero_nodes_plus_3[i][index] = "V: " + WordNetLemmatizer().lemmatize(item.split(": ")[1].lower(), 'v')

            in_all_sentence=copy.deepcopy(no_zero_nodes_plus_3)
            all_sentences_nodes.append([sentence_list[sent_i],in_all_sentence])

            for i in range(len(no_zero_nodes_plus_3)):
                if no_zero_nodes_plus_3[i]:
                    for index, item in enumerate(no_zero_nodes_plus_3[i]):
                        if ('ARG3' in item or 'ARG4' in item or'ARGM-MOD:' in item or 'ARGM-ADJ:' in item or 'R-ARG2:' in item or 'ARGM-PRD' in item or 'ARGM-ADV:' in item or 'ARGM-TMP:' in item or 'ARGM-MNR:' in item or 'R-ARG1:' in item or 'R-ARG0:' in item or 'ARGM-DIS:' in item or 'ARGM-PRP:' in item or 'ARGM-LOC:' in item) :
                            del no_zero_nodes_plus_3[i][index]
            for i in range(len(no_zero_nodes_plus_3)):
                if no_zero_nodes_plus_3[i]:
                    for index, item in enumerate(no_zero_nodes_plus_3[i]):
                        if ('ARGM-GOL' in item or 'ARGM-CAU' in item or 'ARGM-NEG' in item or 'ARG2: ' in item or 'ARGM-MOD:' in item or 'ARGM-ADV:' in item or  'ARGM-TMP:' in item or 'ARGM-MNR:' in item or 'R-ARG1:' in item or 'R-ARG0:' in item or 'R-ARG2:' in item or 'ARGM-LOC:' in item or 'ARGM-EXT:' in item or 'subj:' in item or 'ARG0:' in item):
                            del no_zero_nodes_plus_3[i][index]

            v_unlink = ['delete', 'clear', 'remove', 'erase', 'wipe','purge','expunge', 'drop','drops']
            v_write = ['store', 'place', 'write','add','adds','modify','modifies','append','appends','record','records','set']
            v_read = ['read','obtain','acquire','check','checks']
            v_exec = [ 'use', 'execute', 'executed', 'run', 'ran', 'launch', 'call', 'perform', 'list', 'invoke', 'inject', 'implant', 'open', 'opened','target','resume','exec']
            v_fork = ['clone', 'clones','spawned','spawn','spawns', 'fork']
            v_send = ['send', 'sent','transfer','post','postsinformation','postsinformations', 'transmit','deliver','push','redirect','redirects']
            v_receive = ['receive','accept','get','gets']
            v_collect = ['collect', 'gather', 'extract','extracts']
            v_connect = ['browse', 'browses', 'connect', 'connected', 'portscan', 'connects','communicates','communicate']
            v_chmod = ['chmod', 'change permission','changes permission', 'permision-modifies', 'modifies permission','modify permission']
            v_load = ['load', 'loads']
            v_exit = ['terminate', 'terminates','stop','stops','end','finish','break off','abort','conclude']
    




            for i in range(len(no_zero_nodes_plus_3)):
                fv=False
                fo=False
                if i>=len(no_zero_nodes_plus_3):
                    break
                for index, item in enumerate(no_zero_nodes_plus_3[i]):
                    if item.split(': ')[0] == 'V':
                        fv=True
                        
                        if item.split(': ')[1] in v_unlink:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'remove'
                        elif item.split(': ')[1] in v_write:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'write'
                        elif item.split(': ')[1] in v_read:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'read'
                        elif item.split(': ')[1] in v_exec:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'execute'
                        elif item.split(': ')[1] in v_fork:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'fork'
                        elif item.split(': ')[1] in v_send:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'send'
                        elif item.split(': ')[1] in v_receive:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'receive'
                        elif item.split(': ')[1] in v_connect:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'connect'
                        elif item.split(': ')[1] in v_chmod:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'chmod'
                        elif item.split(': ')[1] in v_load:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'load'
                        elif item.split(': ')[1] in v_exit:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'exit'
                        elif item.split(': ')[1] in v_collect:
                            no_zero_nodes_plus_3[i][index] = 'V: ' + 'collect'
                        if item.split(': ')[1] =='be':
                            del no_zero_nodes_plus_3[i][index]
                        #print("jjj ",no_zero_nodes_plus_3[i][index])
                    elif ('obj' in item.split(': ')[0]  or 'ARG1' in item.split(': ')[0]):
                        fo=True                        
                        objective=item.split(': ')[1].strip()
                        if objective=='.':
                            del no_zero_nodes_plus_3[i][index]
                        else:
                            no_zero_nodes_plus_3[i][index] = 'Obj: ' + objective
                no_zero_nodes_plus_3[i]=sorted(no_zero_nodes_plus_3[i],reverse=True)
                if fv==False or fo==False or len(no_zero_nodes_plus_3[i])<2:
                    del no_zero_nodes_plus_3[i]

        else:
            continue
        all_nodes += no_zero_nodes_plus_3
        if my_vo_triplet:
            all_nodes += my_vo_triplet
        del no_zero_nodes_plus_3
    for node_iter,node in enumerate(all_nodes):
        for v_o_iter ,v_o in enumerate(node):
            if(len(v_o.split(': '))>1):
                all_nodes[node_iter][v_o_iter]=v_o.split(': ')[1]

    return all_sentences_nodes, all_nodes