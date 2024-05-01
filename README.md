# Meta Graph Learning for Long-tail Recommendation

[This repository](https://github.com/Shaun1280/MGL-test) is the re-implementation of [MGL](https://github.com/weicy15/MGL).

## File Descriptions:
- ml-1m/transform_ml.py: data preprocessing of movielens-1m dataset
- bx/transform_bx.py: data preprocessing of bookcrossing dataset
- dataset/bx: preprocessed bookcrossing data
- dataset/ml: preprocessed movielens-1m data
- data_process2.py: another preprocessing, mainly for creating train, val, and test sets
- load_data2.py: data loader and feature processing
- metric.py: for calculating metrics
- model2.py: for embeddings, prediction and loss
- train2.py: for training, validating and testing

## Experiment Environment
- ubuntu 18.04
- GPU L20(48GB) * 1
- CPU20 vCPU Intel(R) Xeon(R) Platinum 8457C
- RAM 100GB

## Requirements

To install requirements:

```setup
python 3.8.19

# this may not work
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
python train2.py --dataset bx --epoch 100

or:
python train2.py --dataset bx --epoch 100 --output <model_name> --model <model_name>
```

To test a model:
```setup
python train2.py --dataset bx --epoch 0 --model <model_name>
```

> Output: the file "model/model.tar" or "model/<model_name>"


