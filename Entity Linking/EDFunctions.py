from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset
import torch

#Functions
ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    # mention_tokens = [Ms] mention [Me]
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add
    
    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    
    # mention_tokens = [CLS] left context [Ms] mention [Me] right context [SEP]
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]
def process_mention_data_2(samples,tokenizer):
    
    max_context_length=64
    mention_key="mention"
    context_key="context"
    ent_start_token="[unused0]"
    ent_end_token="[unused1]"
    
    processed_samples = []
    all_samples = []
    iter_ = samples

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(sample,tokenizer,max_context_length,mention_key,context_key,ent_start_token,ent_end_token)
                        
        record = {"context": context_tokens}
            
        processed_samples.append(record)
        all_samples.append(sample)
        
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "sample":all_samples
    }

    tensor_data = TensorDataset(context_vecs)
    return data, tensor_data
def get_thresholded_preds(th,pred,scores):
    thresholded_preds = []
    for i in range(len(pred)):
        if scores[i]>=th:
            thresholded_preds.append(str(pred[i]))
        else:
            thresholded_preds.append('None')
    return thresholded_preds