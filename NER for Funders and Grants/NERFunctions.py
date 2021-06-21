import pandas as pd
import numpy as np
import torch

#Add [CLS] and [SEP] tokens, pad until "pad_len" chars.
def add_and_pad(lst,pad_len,cls,sep,pad):
    new_lst = []
    for item in lst:
        new_item = [cls] + item + [sep]
        while len(new_item) != pad_len:
            new_item.append(pad)
        new_lst.append(new_item)
    return new_lst

#Class for funding bodies dataset
class FB_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels,at_mask,seq_lens):
        self.encodings = encodings
        self.labels = labels
        self.at_mask = at_mask
        self.seq_lens = seq_lens

    def __getitem__(self, idx):
        item = dict()
        item['input_ids'] = torch.tensor(self.encodings[idx])
        item['attention_mask'] = torch.tensor(self.at_mask[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        item['seq_len'] =self.seq_lens[idx]
        return item

    def __len__(self):
        return len(self.labels)

#This function again tokenizes the input but the aim is to prepare the BERT
#model's input
def tokenize_input_bert(df,input_name,tokenizer):
    tokenized = []
    labels = []
    encoded = []
    for index, row in df.iterrows():
        #Tokenize with BERT
        x1 = tokenizer(row[input_name],return_offsets_mapping=True)
        tokens_encoded = x1['input_ids'][1:-1]
        token_spans = x1['offset_mapping'][1:-1]
        words = tokenizer.tokenize(row[input_name]) 
        
        spans = [token_spans[0]]
        for i in range(1,len(words)):
            this_span = token_spans[i]
            this_word = words[i]
            if len(this_word)>1:
                if this_word[:2] == '##':
                    spans.append(-1)
                    attached_word = i-1
                    while spans[attached_word] == -1:
                        attached_word-=1
                    spans[attached_word] = (spans[attached_word][0],this_span[1])
                else:
                    spans.append(this_span)
            else:
                spans.append(this_span)
        for i in range(len(spans)):
            if spans[i] != -1:
                spans[i] = (spans[i][0]+row['Start_Idx'],spans[i][1]+row['Start_Idx'])
            
        tokenized.append(words)
        labels.append(spans)
        encoded.append(tokens_encoded)
    return tokenized, labels, encoded

def split_long_sentences_old(df,max_len):
    #Extract too long sentences from dataset
    too_long_idxs = []
    for index, row in df.iterrows():
        if len(row['Token_Encoding'])>max_len:
            too_long_idxs.append(index)
    print("Too long sents: ",len(too_long_idxs))
    
    #Separate too long sentences from the dataset
    too_long_df = df[df.index.isin(too_long_idxs)].copy(deep=True)
    df = df[~df.index.isin(too_long_idxs)].copy(deep=True)
    
    #Define the new columns for the new subset of the  dataset
    #This will point to the removed index so we understand which sentences were actually together.
    ID = []
    Sentence = []
    Sentence_Tokenized = []
    Token_Spans = []
    Token_Encoding = []
    for index, row in too_long_df.iterrows():
        #Start index of the subsentence
        start_idx=0
        #End index of the subsentence
        cut_idx=0
        #Length to cover
        remaining = len(row['Sentence_Tokenized'])
        #While we have not covered the whole sentence
        while cut_idx<remaining:
            #Determine where to cut from
            #Ideally, we would like to cut from the longest possible subsentence
            cut_idx = start_idx+max_len
            #If we do not have that many tokens left, we limit the cut to the 
            #length of the sentence
            if cut_idx >= len(row['Sentence_Tokenized']):
                cut_idx = len(row['Sentence_Tokenized'])
            #If we are separating a word, we reduce the index until we are not 
            #separating a word
            if cut_idx != len(row['Sentence_Tokenized']):
                while row['Token_Spans'][cut_idx] == -1:
                    cut_idx -=1
            #We add the information for this cut
            ID.append(index)
            Sentence.append(None)
            Sentence_Tokenized.append(row['Sentence_Tokenized'][start_idx:cut_idx])
            Token_Spans.append(row['Token_Spans'][start_idx:cut_idx])
            Token_Encoding.append(row['Token_Encoding'][start_idx:cut_idx])
            #The next cut will start where this cut ends
            start_idx = cut_idx
    d = {'ID':ID,'Sentence':Sentence,'Sentence_Tokenized':Sentence_Tokenized,'Token_Spans':Token_Spans,'Token_Encoding':Token_Encoding}
    too_long_df_cut = pd.DataFrame(data=d)
        
    #Now we get sentence, start index and end index
    sentence=[]
    start_index=[]
    end_index=[]
    for index, row in too_long_df_cut.iterrows():
        s_i = row['Token_Spans'][0][0]
        start_index.append(int(s_i))
        get_end_idx = -1
        while row['Token_Spans'][get_end_idx] == -1:
            get_end_idx -=1
        e_i = row['Token_Spans'][get_end_idx][1]
        end_index.append(int(e_i))
        old_sent = too_long_df[too_long_df.index==row['ID']].Sentence.values[0]
        old_sent_s_i = too_long_df[too_long_df.index==row['ID']].Start_Idx.values[0]
        sentence.append(old_sent[(s_i-old_sent_s_i):(e_i-old_sent_s_i)])
    too_long_df_cut['Start_Idx'] = start_index
    too_long_df_cut['End_Idx'] = end_index
    too_long_df_cut['Sentence'] = sentence
    
    #Merge with the original training set
    df=df.append(too_long_df_cut,ignore_index=True)
    return df, too_long_idxs

def get_labels(df):
    labels = []
    for index, row in df.iterrows():
        this_lbl = []
        for elt in row['Sentence_Tokenized']:
            if len(elt)>1:
                if elt[:2] == '##':
                    this_lbl.append(-1)
                else:
                    this_lbl.append(0)
            else:
                this_lbl.append(0)
        labels.append(this_lbl)
    return labels


def tokenize_with_bert(df,tokenizer):
    tokenized_sents = []
    token_spans= []
    for index, row in df.iterrows():
        words = tokenizer.tokenize(row['Sentence'])
        spans = tokenizer(row['Sentence'],return_offsets_mapping=True)['offset_mapping'][1:-1]
        new_words = []
        new_spans = []
        for i in range(len(words)):
            if words[i][:2] == '##':
                prev_word = i-1
                if len(words[prev_word])>1:
                    while words[prev_word][:2] == "##":
                        prev_word-=1
                new_words[prev_word] = new_words[prev_word]+words[i][2:]
                new_spans[prev_word] = (new_spans[prev_word][0],spans[i][1])
                new_words.append(-1)
                new_spans.append(-1)
            else:
                new_words.append(words[i])
                new_spans.append(spans[i])
        new_words = [word for word in new_words if word != -1]
        new_spans = [span for span in new_spans if span != -1]
        for i in range(len(new_spans)):
            new_spans[i] = (new_spans[i][0]+row['Start_Idx'],new_spans[i][1]+row['Start_Idx'])
        tokenized_sents.append(new_words) 
        token_spans.append(new_spans)
    return tokenized_sents,token_spans