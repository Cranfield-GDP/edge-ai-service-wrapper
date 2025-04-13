---
language: en
datasets:
- conll2003
license: mit
model-index:
- name: dslim/bert-base-NER
  results:
  - task:
      type: token-classification
      name: Token Classification
    dataset:
      name: conll2003
      type: conll2003
      config: conll2003
      split: test
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9118041001560013
      verified: true
    - name: Precision
      type: precision
      value: 0.9211550382257732
      verified: true
    - name: Recall
      type: recall
      value: 0.9306415698281261
      verified: true
    - name: F1
      type: f1
      value: 0.9258740048459675
      verified: true
    - name: loss
      type: loss
      value: 0.48325642943382263
      verified: true
---
# bert-base-NER

If my open source models have been useful to you, please consider supporting me in building small, useful AI models for everyone (and help me afford med school / help out my parents financially). Thanks!

<a href="https://www.buymeacoffee.com/dslim" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Model description

**bert-base-NER** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC). 

Specifically, this model is a *bert-base-cased* model that was fine-tuned on the English version of the standard [CoNLL-2003 Named Entity Recognition](https://www.aclweb.org/anthology/W03-0419.pdf) dataset. 

If you'd like to use a larger BERT-large model fine-tuned on the same dataset, a [**bert-large-NER**](https://huggingface.co/dslim/bert-large-NER/) version is also available. 

### Available NER models 
| Model Name | Description | Parameters |
|-------------------|-------------|------------------|
| [distilbert-NER](https://huggingface.co/dslim/distilbert-NER) **(NEW!)** | Fine-tuned DistilBERT - a smaller, faster, lighter version of BERT | 66M |
| [bert-large-NER](https://huggingface.co/dslim/bert-large-NER/) | Fine-tuned bert-large-cased - larger model with slightly better performance | 340M |
| [bert-base-NER](https://huggingface.co/dslim/bert-base-NER)-([uncased](https://huggingface.co/dslim/bert-base-NER-uncased)) | Fine-tuned bert-base, available in both cased and uncased versions | 110M |


## Intended uses & limitations

#### How to use

You can use this model with Transformers *pipeline* for NER.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)
```

#### Limitations and bias

This model is limited by its training dataset of entity-annotated news articles from a specific span of time. This may not generalize well for all use cases in different domains. Furthermore, the model occassionally tags subword tokens as entities and post-processing of results may be necessary to handle those cases. 

## Training data

This model was fine-tuned on English version of the standard [CoNLL-2003 Named Entity Recognition](https://www.aclweb.org/anthology/W03-0419.pdf) dataset. 

The training dataset distinguishes between the beginning and continuation of an entity so that if there are back-to-back entities of the same type, the model can output where the second entity begins. As in the dataset, each token will be classified as one of the following classes:

Abbreviation|Description
-|-
O|Outside of a named entity
B-MISC |Beginning of a miscellaneous entity right after another miscellaneous entity
I-MISC | Miscellaneous entity
B-PER |Beginning of a person’s name right after another person’s name
I-PER |Person’s name
B-ORG |Beginning of an organization right after another organization
I-ORG |organization
B-LOC |Beginning of a location right after another location
I-LOC |Location


### CoNLL-2003 English Dataset Statistics
This dataset was derived from the Reuters corpus which consists of Reuters news stories. You can read more about how this dataset was created in the CoNLL-2003 paper. 
#### # of training examples per entity type
Dataset|LOC|MISC|ORG|PER
-|-|-|-|-
Train|7140|3438|6321|6600
Dev|1837|922|1341|1842
Test|1668|702|1661|1617
#### # of articles/sentences/tokens per dataset
Dataset |Articles |Sentences |Tokens
-|-|-|-
Train |946 |14,987 |203,621
Dev |216 |3,466 |51,362
Test |231 |3,684 |46,435

## Training procedure

This model was trained on a single NVIDIA V100 GPU with recommended hyperparameters from the [original BERT paper](https://arxiv.org/pdf/1810.04805) which trained & evaluated the model on CoNLL-2003 NER task. 

## Eval results
metric|dev|test
-|-|-
f1 |95.1 |91.3
precision |95.0 |90.7
recall |95.3 |91.9

The test metrics are a little lower than the official Google BERT results which encoded document context & experimented with CRF. More on replicating the original results [here](https://github.com/google-research/bert/issues/223).

### BibTeX entry and citation info

```
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
```
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
```
