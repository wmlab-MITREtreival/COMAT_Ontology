#For CPU
from filter_bert import BERTClass
import torch
import pickle
from nltk import sent_tokenize
import statistics
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

def remove_non_ascii(s):
    return "".join(c for c in s if ord(c) < 128)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = remove_non_ascii(text)
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'text': text
        }

def test_preprocess(text):
    model_config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_dict=False)
    model = BERTClass(model_config)
    MAX_LEN = 512
    model.load_state_dict(torch.load("bert_binary.pt", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    
    df_list = sent_tokenize(text)
    df1 = pd.DataFrame({'Text': df_list})
    training_set = CustomDataset(df1, tokenizer, MAX_LEN)
    test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    testing_loader = DataLoader(training_set, **test_params)
    fin_outputs = []
    new_text = ""
    all_text = ""

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to('cpu', dtype=torch.long)
            mask = data['mask'].to('cpu', dtype=torch.long)
            token_type_ids = data['token_type_ids'].to('cpu', dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            all_text += data['text'][0] + " "
            if outputs[0][1] > outputs[0][0]:
                new_text += data['text'][0] + " "
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        print(new_text)

def topic_classifier(txt):
    model_config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_dict=False)
    model = BERTClass(model_config)
    model.load_state_dict(torch.load("bert_binary_cy.pt", map_location=torch.device('cpu')), strict=False)
    model.to('cpu')
    model.eval()
    
    MAX_LEN = 512
    df_list = sent_tokenize(txt)
    df1 = pd.DataFrame({'Text': df_list})
    new_text = ""
    testing_set = CustomDataset(df1, tokenizer, MAX_LEN)
    test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    testing_loader = DataLoader(testing_set, **test_params)
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to('cpu', dtype=torch.long)
            mask = data['mask'].to('cpu', dtype=torch.long)
            token_type_ids = data['token_type_ids'].to('cpu', dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            if outputs[0][1] > outputs[0][0]:
                new_text += data['text'][0] + " "
    return new_text

def Preprocess(threshold: str):
    model_config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_dict=False)
    model = BERTClass(model_config)
    model.load_state_dict(torch.load("bert_binary_cy.pt", map_location=torch.device('cpu')))
    model.to('cpu')
    
    model.eval()
    with open('dl_train_feature_' + threshold, "rb") as f:
        train_ = pickle.load(f)
    with open("dl_test_feature_" + threshold, "rb") as of:
        test_ = pickle.load(of)
    
    train_test = pd.concat([train_, test_])
    sentences_list = []
    word_count_list = []
    for it in train_test['Text']:
        sentences = sent_tokenize(it)
        sentences_list.append(len(sentences))
        word_count = it.split()
        word_count_list.append(len(word_count))
    
    print("Original sentences number min= ", min(sentences_list))
    print("Original sentences number max= ", max(sentences_list))
    print("Original sentences number mean= ", statistics.mean(sentences_list))
    print("Original Words number min= ", min(word_count_list))
    print("Original Words number max= ", max(word_count_list))
    print("Original Words number mean= ", statistics.mean(word_count_list))
    
    MAX_LEN = 512
    whole_doc_list_train = []
    whole_doc_list_test = []

    for iter_doc in tqdm(train_['Text']):
        df_list = sent_tokenize(iter_doc)
        df1 = pd.DataFrame({'Text': df_list})
        training_set = CustomDataset(df1, tokenizer, MAX_LEN)
        test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
        testing_loader = DataLoader(training_set, **test_params)
        new_text = ""
        all_text = ""
        fin_outputs = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to('cpu', dtype=torch.long)
                mask = data['mask'].to('cpu', dtype=torch.long)
                token_type_ids = data['token_type_ids'].to('cpu', dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)
                all_text += data['text'][0] + " "
                if outputs[0][1] > outputs[0][0]:
                    new_text += data['text'][0] + " "
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            whole_doc_list_train.append(new_text)

    for iter_doc in tqdm(test_['Text']):
        df_list = sent_tokenize(iter_doc)
        df1 = pd.DataFrame({'Text': df_list})
        training_set = CustomDataset(df1, tokenizer, MAX_LEN)
        test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
        testing_loader = DataLoader(training_set, **test_params)
        new_text = ""
        all_text = ""
        fin_outputs = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to('cpu', dtype=torch.long)
                mask = data['mask'].to('cpu', dtype=torch.long)
                token_type_ids = data['token_type_ids'].to('cpu', dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)
                all_text += data['text'][0] + " "
                if outputs[0][1] > outputs[0][0]:
                    new_text += data['text'][0] + " "
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            whole_doc_list_test.append(new_text)

    new_train_df = pd.DataFrame({'Text': whole_doc_list_train})
    new_train_df['list'] = train_['list']
    new_test_df = pd.DataFrame({'Text': whole_doc_list_test})
    new_test_df['list'] = test_['list']
    
    new_train_df = new_train_df[new_train_df['Text'].apply(lambda x: len(sent_tokenize(x)) > 0)].reset_index(drop=True)
    new_test_df = new_test_df[new_test_df['Text'].apply(lambda x: len(sent_tokenize(x)) > 0)].reset_index(drop=True)
    
    with open("dl_train_feature_" + threshold + "_clean", "wb") as nf:
        pickle.dump(new_train_df, nf)
    with open("dl_test_feature_" + threshold + "_clean", "wb") as nf_t:
        pickle.dump(new_test_df, nf_t)

if __name__ == "__main__":
    test_preprocess("The secret, APT10 turned out, was a hardcoded list of passwords")


