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
python transform_dataset.py bx ratings_wo_duplicates.csv BX_users.csv BX_Books.csv

output_path: dataset/
```

```setup
python data_process.py
```

Book crossing
```
item_feature.pkl:
  Index(['item', 'title', 'Book-Author', 'Year-Of-Publication', 'Publisher'], dtype='object')

user_feature.pkl:
   Index(['user', 'location', 'age'], dtype='object')

interact_train.pkl:
  Index(['userid', 'itemid', 'score', 'timestamp'], dtype='object')
```

## Training

To train the model(s) in the paper:

```setup
python train.py
```
> Output: the file "model.tar"


