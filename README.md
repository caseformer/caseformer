# Caseformer

## Source code of our submitted paper

## Caseformer: Pre-training for Legal Case Retrieval


### Some notes about this anonymous repository
**This GitHub repository has been anonymized.**
**The core code of this paper are publicly available in this GitHub repository. As the paper is currently under submission, once it is accepted, we will disclose the complete code and data in this repository.**
**Some of the code in this repository involves absolute paths. Once the paper is accepted, we will make all the files corresponding to these paths publicly available.**

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

Format of the input documents:

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

Format of the input documents:

```
{"docID":string,"content":string}
{"docID":string,"content":string}
{"docID":string,"content":string}
......
{"docID":string,"content":string}
```





### Prepare the Training Data

#### LJP Task
```
cd caseformer
python ./pre-training_data_generation/generate_LJP_task_data.py \
--BM25_top_100  path \
--law_articles path \
--crimes path \
--output_path your_path
```


#### FDM Task
```
cd caseformer
python ./pre-training_data_generation/generate_FDM_task_data.py \
--LP-ICF_top_100  path \
--law_articles path \
--crimes path \
--output_path your_path
```


### Running Pre-training
#### Pre-train Caseformer_retriever
```
cd caseformer
bash ./pre-training/pre-train_retriever.sh
```

shell script:
```
CUDA_VISIBLE_DEVICES=6 python /home/swh/dense/projects/tensorboard_dense/src/dense/driver/train_with_tensorboard.py \
  --output_dir /home/swh/dense/projects/tensorboard_dense/ckpts/finetune_roberta2 \
  --model_name_or_path /home/swh/huggingface_models/chinese_roberta \
  --do_train \
  --save_steps 1000 \
  --train_dir /home/swh/legal/project/sigir/dense/finetune_dense/train_data \
  --tokenizer_name /home/swh/huggingface_models/chinese_roberta \
  --fp16 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --train_n_passages 40 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --inference_result_path /home/swh/dense/projects/tensorboard_dense/codes/data/finetune_roberta2 \
  --inference_in_path /home/swh/dense/projects/tensorboard_dense/codes/lecard/data/recall_inf_without_train.json \
  --tensorboard_path /home/swh/dense/projects/tensorboard_dense/tensorboard/finetune_roberta2
```


#### Pre-train Caseformer_reranker
```
cd caseformer
bash ./pre-training/pre-train_reranker.sh
```

shell script:
```
CUDA_VISIBLE_DEVICES=6 python /home/swh/dense/projects/tensorboard_dense/src/dense/driver/train_with_tensorboard.py \
  --output_dir /home/swh/dense/projects/tensorboard_dense/ckpts/finetune_roberta2 \
  --model_name_or_path /home/swh/huggingface_models/chinese_roberta \
  --do_train \
  --save_steps 1000 \
  --train_dir /home/swh/legal/project/sigir/dense/finetune_dense/train_data \
  --tokenizer_name /home/swh/huggingface_models/chinese_roberta \
  --fp16 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --train_n_passages 40 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --inference_result_path /home/swh/dense/projects/tensorboard_dense/codes/data/finetune_roberta2 \
  --inference_in_path /home/swh/dense/projects/tensorboard_dense/codes/lecard/data/recall_inf_without_train.json \
  --tensorboard_path /home/swh/dense/projects/tensorboard_dense/tensorboard/finetune_roberta2
```

