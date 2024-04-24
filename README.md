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
Expected model/model.tar prediction result (user item freq threshold = 6, 0 epochs):
```
python train2.py --dataset bx --epoch 0 --model model-bx-0.tar

Namespace(K_list=[10, 20, 50], L=3, batch_size=128, beta=0.1, convergence=40, cross_validate=None, dataset_name='bx', dense_embedding_dim=16, embedding_size=8, epoch=0, id_embedding_size=64, implcit_bottom=None, item_fre_threshold=None, link_topk=10, loadFilename=None, load_mode='test_set', local_lr=0.01, lr=0.001, reg_lambda=0.02, seperate_rate=0.2, social_data=False, split=None, top_rate=0.1, user_fre_threshold=None, weight_decay=0.01)
Meta_final_2
NDCG@10: 0.0019885282841520103
RECALL@10: 0.00439453125
MRR@10: 0.0012845801297949736

head_NDCG@10: 0.004150366277809452
head_RECALL@10: 0.008951822916666666

tail_NDCG@10: 0.0013591006028556213
tail_RECALL@10: 0.0029296875

body_NDCG@10: 0.0012532526898807823
body_RECALL@10: 0.0026041666666666665
NDCG@20: 0.0031701756732269797
RECALL@20: 0.009114583333333334
MRR@20: 0.0016017261455875582

head_NDCG@20: 0.006419071032709525
head_RECALL@20: 0.01806640625

tail_NDCG@20: 0.0019404580967747021
tail_RECALL@20: 0.005208333333333333

body_NDCG@20: 0.0018075892880403059
body_RECALL@20: 0.0048828125
NDCG@50: 0.005976962380900827
RECALL@50: 0.0234375
MRR@50: 0.002038660399545376

head_NDCG@50: 0.011853943397438225
head_RECALL@50: 0.045868844696969696

tail_NDCG@50: 0.002730372342562085
tail_RECALL@50: 0.00927734375

body_NDCG@50: 0.003669716198958233
body_RECALL@50: 0.014485677083333334
```


