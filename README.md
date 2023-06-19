# Caseformer

Source code of our submitted paper:

Caseformer: Pre-training for Legal Case Retrieval

**This GitHub repository has been anonymized.**



### The file structure of this repository:

```
.
├── demo_data
│   ├── legal_documents
│   │   ├── legal_documents.jsonl
│   │   └── file_format.txt
│   └── preprocessed_training_data
│       ├── LJP_task.jsonl
│       ├── FDM_task.jsonl
│       └── file_format.txt
├── data_preprocess
│   ├── law_article_extration.py
│   └── crime_extraction.py
├── pre-training_data_generation
│   ├── demo_data
│   │   ├── bm25_top100.jsonl
│   │   ├── extracted_law_articles.jsonl
│   │   └── extracted_crimes.jsonl
│   ├── generate_LJP_task_data.py
│   ├── generate_FDM_task_data.py
│   └── calc_LP-ICF_score.py
├── pre-training
│   └── pre-train.sh

```





### Pre-installation

```
git clone git@github.com:caseformer/caseformer.git
cd caseformer
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```



### Extract structured information from legal documents

#### Extract law articles

```
cd caseformer
python ./data_preprocess/law_article_extraction.py \
--path_to_documents your_path \
--output_path your_path
```

--path_to_documents 



Format:

```
{"docID":string,"content":string}
{"docID":string,"content":string}
{"docID":string,"content":string}
......
{"docID":string,"content":string}
```





#### Extract Crimes



```
cd caseformer
python ./data_preprocess/crime_extraction.py \
--path_to_documents your_path \
--output_path your_path
```

--path_to_documents 



Format:

```
{"docID":string,"content":string}
{"docID":string,"content":string}
{"docID":string,"content":string}
......
{"docID":string,"content":string}
```





### Prepare the Training Data

#### LJP Task



#### FDM Task



### Running Pre-training

Pretrain directly using the demo pre-training data