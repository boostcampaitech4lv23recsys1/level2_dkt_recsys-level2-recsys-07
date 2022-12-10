# Graph Embeded Smoothing을 활용한 모델

<img src="https://github.com/zhuty16/GES/blob/master/framework.jpg?raw=true">

[Graph-Based Embedding Smoothing for Sequential Recommendation 논문](https://ieeexplore.ieee.org/abstract/document/9405450)  

[Graph-Based Embedding Smoothing 원작자 깃허브](https://github.com/zhuty16/GES)

## Dependency Install

``` bash
chmod +x install.sh 
./install.sh
```
## Prepare Dataset

``` bash

python prepare.py

```

-> create fe_train_data.csv

-> create fe_test_data.csv

## Train Run 
``` bash
python train.py --model NAME
```

## Inference
``` bash
python inference.py --model NAME

```

## Deafault Parameter

### 모델 파라미터

``` python
--max_seq_len, default=200, type=int, help="max sequence length"
--hidden_dim, default=256, type=int, help="hidden dimension size"
--n_layers, default=2, type=int, help="number of layers"
--n_heads, default=4, type=int, help="number of heads"
--drop_out, default=0.4, type=float, help="drop out rate"
--gcn_n_layes, default=2, type=int, help="gcn layers"
--alpha, type=float, default=1.0, help="weight of seq Adj"
--beta, type=float, default=1.0, help="weight of sem Adj"
```



### 훈련

``` python
--n_epochs, default=1, type=int, help="number of epochs"
--batch_size, default=32, type=int, help="batch size"
--lr, default=0.000001, type=float, help="learning rate"
--clip_grad, default=10, type=int, help="clip grad"
--patience, default=10, type=int, help="for early stopping"
```

