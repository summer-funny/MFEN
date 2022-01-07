# MFEN 
MFEN: Multi-metrics Feature Extraction Network for Few-Shot Relation Prediction

# Requirements

python 3.6

Pytorch == 1.1.0

CUDA: 9.0

# How to run

#### Nell
```
python trainer.py --prefix nell_5shot
```
#### Wiki

```
python trainer.py --dataset wiki --embed_dim 50 --prefix wiki_5shot
```



# How to test

#### Nell
```
python trainer.py --prefix nell_5shot_best --test
```

#### Wiki
```
python trainer.py --dataset wiki --embed_dim 50 --prefix wiki_5shot_best --test
```
