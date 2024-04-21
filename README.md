# Meta Graph Learning for Long-tail Recommendation

This repository is the official implementation of MGL.

## Requirements

To install requirements:

```setup
python 3.8.19

# may not work
conda env create -f environment.yaml -n mgl

# CUDA 11.0
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# CPU Only
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch

pip install torch-sparse==0.6.9 torch-scatter==2.0.7 tqdm==4.54.1 pandas==1.4.4 scikit-learn==0.23.2 rectorch==0.0.9b0
```

## Data Process

To prepare the data for the model training:

```setup
old:
python data_process.py

new:
python data_process2.py
```

Book crossing:
```
item_feature.pkl:
  ['item', 'title', 'Book-Author', 'Year-Of-Publication', 'Publisher']

user_feature.pkl:
  ['user', 'location', 'age']

interact_train.pkl:
  ['userid', 'itemid', 'score', 'timestamp'] (timestamp = 0)
```

Movielens-1M:
```
item_feature.pkl:
  ['item', 'title', 'genres', 'year']

user_feature.pkl:
   ['user', 'gender', 'age', 'occupation', 'zip-code']

interact_train.pkl:
  ['userid', 'itemid', 'score', 'timestamp']
```

## Training

To train the model(s) in the paper:

```setup
old:
python train.py

new:
python train2.py --dataset bx --epoch 100
```

> Output: the file "model/model.tar"

## Prediction
Expected model/model.tar prediction result (5 epochs):
```
python train2.py --dataset bx --epoch 0 --model model-bx-5.tar

Namespace(K_list=[10, 20, 50], L=3, batch_size=128, beta=0.1, convergence=40, cross_validate=None, dataset_name='bx', dense_embedding_dim=16, embedding_size=8, epoch=0, id_embedding_size=64, implcit_bottom=None, item_fre_threshold=None, link_topk=10, loadFilename=None, load_mode='test_set', local_lr=0.01, lr=0.001, reg_lambda=0.02, seperate_rate=0.2, social_data=False, split=None, top_rate=0.1, user_fre_threshold=None, weight_decay=0.01)
Meta_final_2
NDCG@10: 0.006996426316734284
RECALL@10: 0.012595019096869426
MRR@10: 0.00819188843448712

head_NDCG@10: 0.008838922592907143
head_RECALL@10: 0.015313736551401025

tail_NDCG@10: 0.00293742884303968
tail_RECALL@10: 0.006057617915142586

body_NDCG@10: 0.00288486284928376
body_RECALL@10: 0.006238979522285444
NDCG@20: 0.008726406606767318
RECALL@20: 0.018344177466961147
MRR@20: 0.009076320064120113

head_NDCG@20: 0.01201136437219491
head_RECALL@20: 0.026289055461668175

tail_NDCG@20: 0.004282807906525026
tail_RECALL@20: 0.011031905464592193

body_NDCG@20: 0.003758954297817896
body_RECALL@20: 0.009295088336636082
NDCG@50: 0.012911715887406866
RECALL@50: 0.03581372760343694
MRR@50: 0.010197982058791515

head_NDCG@50: 0.018453605944794102
head_RECALL@50: 0.055301413378220164

tail_NDCG@50: 0.006253641418941265
tail_RECALL@50: 0.020242293774728953

body_NDCG@50: 0.0063220311907828735
body_RECALL@50: 0.020980212766697214
```


