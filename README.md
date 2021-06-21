This repository contains the code for Gizem Aydin's Master's Thesis. The thesis is done for Radboud Univesity and Elsevier B.V.
* Supervisor: Dr. Faegheh Hasibi, Radboud University
* Supervisor: Dr. Seyed Amin Tabatabaei, Elsevier B.V.
* Second reader: Prof. Dr. Arjen P. de Vries, Radboud University

The full thesis: `Thesis.pdf`

# Domain Adaptation of BERT
First, a BERT [1] model is pretrained following the Task-Adaptive Pretraining (TAPT) schema proposed by [2]. The code can be found in `Pretraining BERT/TAPT_ScopusData_BERT.ipynb`. The code is copied from [this notebook by the authors of TAPT](https://github.com/allenai/dont-stop-pretraining/blob/master/scripts/run_language_modeling.py), and modified to fit the problem at hand. 

# Named Entity Recognition for Funding Organizations and Grant Numbers
Training and evaluating an NER component that can extract mentions of funding organizations and grant numbers. 
The labelled dataset and the trained model are not provided. 

* Training notebook: `NER for Funders and Grant/Train_BERT_Scopus_NER.ipynb`
    * This notebook requires a labelled dataset. Using that, the NER model is trained and saved to a file.
* Evaluation notebook: `NER for Funders and Grant/Evaluate_BERT.ipynb`
    * This notebook requires a labelled and a trained models. Using these, it evaluates the model.
* Prediction notebook: `NER for Funders and Grant/NER_Predictions.ipynb`
    * This notebook requires a trained model. With the input trained model, it extracts the mentions of funding organizations and grant numbers. It also provides some statistics such as the output tag probability distribution for each token.


# Entiy Disambiguation for Funding Organizations
Training and evaluating an Entity Disambiguation system for funder organization mentions. The labelled datasets, the trained models and the Knowledge Base is not included.
## Training
### Candidate Selection
First the candidate selector should be trained.
1. Training with random negatives: Run `ED for Funding Organizations/BiEncoder RandomNegative Training.ipynb`
2. The rest of the training will be with hard negatives. For this purpose, the predictions on training and dev sets should be obtained.
    1. Compute entity embeddings: Run `ED for Funding Organizations/Compute Entity Embeddings.ipynb`
    2. Run the notebook `ED for Funding Organizations/Hard Negative Mining.ipynb` twice. Once for training set, and a second time for the dev set.
    3. Run the notebook `ED for Funding Organizations/Number Random Negatives.ipynb` to see how many random negatives per mention will be sampled for the next round. This notebook also shows some statistics on the number of hard negatives found.
3. Train with hard negatives: Run `ED for Funding Organizations/BiEncoder HardNegative Training.ipynb`
4. Repeat steps (2-3) 3 times. (3=number of hard negative training rounds)
### Candidate Reranking
Candidate Reranking model needs to be trained after the candidate selector.
1. Run the notebook `ED for Funding Organizations/Prediction with Biencoder.ipynb` twice. Once for training set, and a second time for the dev set. This notebook retrieves the candidate entities for these datasets, which are later used for training.
2. Run the notebook `ED for Funding Organizations/Train GBM Reranker.ipynb`
## Prediction
Run the notebook `ED for Funding Organizations/Train GBM Reranker.ipynb`. This notebook can perform predictions on input mentions with the full model.
## Evaluation
The notebook `ED for Funding Organizations/Evalute ED Model.ipynb` contains the evaluation functions that are used for the thesis.

# Entity Linking for Funding Organizations and Named Entity Recognition for Grant Numbers

## Prediction
Run the notebook `Neural Entity Linking Predictions.ipynb`. For the given sentences, the organization and grant mentions are extracted. Then, the Disambiguation is performed on the organization mentions.
## Evaluation
The evaluation functions used in the thesis can be found in these Python scripts with the function name `Evaluate_End2End`:
* `EvaluationPoolStrict.py`: Strict matching, ``Normal`` setting.
* `EvaluationPoolStrictEE.py`: Strict matching, ``EE`` setting.
* `EvaluationPoolStrictInKB.py`: Strict matching, ``InKB`` setting.

How to use the function `Evaluate_End2End`?
* Inputs:
    * `all_gold_ann`: List of lists. The length of the main list is equal to the number of documents. For each document, a list stores the gold annotations. In this list, each annotation is indicated with another list of 3 elements. The first element is the start index of mention, the second element is the length of the mention, and the third element is the correct entity ID. Example gold annotation list for a document:
    ```python
    [
        [5,10,"Entity_A"],
        [25,3,None]
    ]
    ```
    According to the example, there are two mentions. One starts at character index 5 and has a length of 10. The correct link for this mention is `"Entity_A"`. The other mention starts at index 25 and has a length of 3. This is a NIL mention.
    * `all_preds`: Similar to `all_gold_ann`. Only difference is that it contains the predicted annotations instead of the gold ones.
    * `entity_pool`: See `ED for Funding Organizations/Hard Negative Mining.ipynb`
* Output: Prints Micro and Macro averaged Precision, Recall and F1 scores. Returns Micro averaged ones.

# Notes
* The instructions are included in the beginning of the notebooks.
* If you have any questions, contact: gizemaydin96@gmail.com

## Libraries and Versions
Python==3.7.9
* [annoy](https://github.com/spotify/annoy)==1.17.0
* [datasets](https://pypi.org/project/datasets/)==1.4.1
* [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)==0.18.0
* [lightgbm](https://github.com/microsoft/LightGBM)==2.3.0
* [numpy](https://numpy.org/)==1.19.2
* [pandas](https://pandas.pydata.org/)==1.1.3
* [scipy](https://www.scipy.org/)==1.5.2
* [seqeval](https://github.com/chakki-works/seqeval)==1.2.2
* [torch](https://pytorch.org/)==1.7.1
* [tqdm](https://github.com/tqdm/tqdm)==4.49.0
* [transformers](https://huggingface.co/transformers/)==3.5.1

# References

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational  Linguistics:  Human  Language  Technologies,  Volume  1  (Long  and Short  Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
[2] Suchin Gururangan,  Ana Marasovic,  Swabha Swayamdipta,  Kyle Lo,  Iz Beltagy, Doug Downey, and Noah A. Smith. Don't stop pretraining: Adapt language models to domains and tasks. In Proceedings of ACL, 2020.