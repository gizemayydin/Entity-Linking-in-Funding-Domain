import numpy as np

#Gets two annotation instances and determine if they are matching
def match(instance,target,entity_pool):
    entity_match = target[2] in entity_pool[str(instance[2])]
    span_match = (instance[0] == target[0]) and  (instance[1] == target[1])
    
    if entity_match and span_match:
        return True
    else:
        return False

#Get the number of TP/FP/FN per document
def eval_doc(gold_ann,preds,entity_pool):
    #Store the counts here
    document_counts = {'tp':0,'fp':0,'fn':0}
    #We set the index to 1 if we match the prediction with a gold annotation
    already_used_results = np.zeros(len(preds),dtype=int)
    #This will store the match index
    #(initialized here to prevent obj creation)
    match_index = -1
    #Loop over all the gold annotations
    for gold in gold_ann:
        #Set match index to -1
        match_index = -1
        #Loop over predictions
        for idx, pred in enumerate(preds):
            #Check if prediction is matching
            is_matching = match(gold,pred,entity_pool)
            #If we found a match, we set the match_index to the index of the match
            #then we stop searching
            if is_matching:
                match_index = idx
                break
        #If there is a match
        if match_index != -1:
            #We have a true positive
            document_counts['tp'] += 1
            #We mark the matched predictio so we won't use it again
            already_used_results[match_index] = 1
        else:
            #If no match, this is a false negative
            document_counts['fn'] += 1
    #The unmatched predictions are false positives
    document_counts['fp'] =+ len(already_used_results) - int(np.sum(already_used_results))
    return document_counts


#This function calculates PRF scores for a given document
def calculate_measures(res):
    prec=None
    rec =None
    f1  =None
    
    #If we do not have any true positives
    if res['tp'] == 0:
        #If we do not have any false pos/neg as well
        #The annotation is correct
        if res['fp'] == 0 and res['fn'] == 0:
            prec=1.0
            rec =1.0
            f1  =1.0
        #If we have some false pos/neg annotation is wrong
        #and we do not have anything correct
        else:
            prec=0.0
            rec =0.0
            f1  =0.0
    #Else
    else:
        #PRF calculation
        prec = res['tp'] / (res['tp'] + res['fp']);
        rec  = res['tp'] / (res['tp'] + res['fn']);
        f1   = (2 * prec * rec) / (prec + rec)
    return [prec,rec,f1]

def Evaluate_End2End(all_gold_ann,all_preds,entity_pool):
    entity_pool['None'] = set(['None'])
    #Store the results of TP/FP/FN calculation for each document
    all_doc_res = []
    for doc_idx in range(len(all_gold_ann)):
        gold_ann = all_gold_ann[doc_idx]
        preds = all_preds[doc_idx]
        all_doc_res.append(eval_doc(gold_ann,preds,entity_pool))
    #Calculate micro scores
    #This is equivalent to treating these numbers as single document
    micro_results = {'tp':0,'fp':0,'fn':0}
    for item in all_doc_res:
        micro_results['tp'] += item['tp']
        micro_results['fp'] += item['fp']
        micro_results['fn'] += item['fn']
    micro_scores= calculate_measures(micro_results)
    print("Micro Scores: ")
    print("\tPrecision: ",micro_scores[0])
    print("\tRecall: ",micro_scores[1])
    print("\tF1: ",micro_scores[2])
    
    #Calculate macro scores
    #Calculate score for each document
    #Then average these scores
    all_doc_prec = []
    all_doc_rec = []
    all_doc_f1 = []
    for item in all_doc_res:
        this_doc_scores = calculate_measures(item)
        all_doc_prec.append(this_doc_scores[0])
        all_doc_rec.append(this_doc_scores[1])
        all_doc_f1.append(this_doc_scores[2])
    print("Macro Scores: ")
    print("\tPrecision: ",np.mean(all_doc_prec))
    print("\tRecall: ",np.mean(all_doc_rec))
    print("\tF1: ",np.mean(all_doc_f1))
    return micro_results