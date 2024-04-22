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
Expected model/model.tar prediction result (user item freq threshold = 6, 5 epochs):
```
python train2.py --dataset bx --epoch 0 --model model-bx-5.tar

Namespace(K_list=[10, 20, 50], L=3, batch_size=128, beta=0.1, convergence=40, cross_validate=None, dataset_name='bx', dense_embedding_dim=16, embedding_size=8, epoch=0, id_embedding_size=64, implcit_bottom=None, item_fre_threshold=None, link_topk=10, loadFilename=None, load_mode='test_set', local_lr=0.01, lr=0.001, reg_lambda=0.02, seperate_rate=0.2, social_data=False, split=None, top_rate=0.1, user_fre_threshold=None, weight_decay=0.01)
Meta_final_2
NDCG@10: 0.006966148958437254
RECALL@10: 0.011959912802840434
MRR@10: 0.00811387191938179

head_NDCG@10: 0.009377201626060061
head_RECALL@10: 0.016366029004803676

tail_NDCG@10: 0.0032788977566884335
tail_RECALL@10: 0.006154057017543859

body_NDCG@10: 0.002730639562476644
body_RECALL@10: 0.005056994700292397
NDCG@20: 0.009284879313552616
RECALL@20: 0.019256429985616805
MRR@20: 0.009346940040194393

head_NDCG@20: 0.012977246956064751
head_RECALL@20: 0.028644861740050882

tail_NDCG@20: 0.0047920756004418905
tail_RECALL@20: 0.011880653782894737

body_NDCG@20: 0.0038804170440125216
body_RECALL@20: 0.00913585169855729
NDCG@50: 0.013821622865658485
RECALL@50: 0.038197207516068664
MRR@50: 0.01053075667999894

head_NDCG@50: 0.021316176024078815
head_RECALL@50: 0.06562929286767691

tail_NDCG@50: 0.00733227699575873
tail_RECALL@50: 0.0239625632951264

body_NDCG@50: 0.006448544463836808
body_RECALL@50: 0.021245955935511856
```


