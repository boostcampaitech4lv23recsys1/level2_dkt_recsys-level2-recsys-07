import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from CatBoost_data import cat_data_load, cat_data_split
from CatBoost_model import CatBoost

# import wandb


def main(args):
    # setSeeds(args.SEED)
    now = time.localtime()

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    # wandb.init(project = args.MODEL, name = time.strftime('%m/%d %H:%M:%S',now), config={"epochs": args.EPOCHS, "batch_size": 1024})
    if args.MODEL == 'cat':
        data = cat_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL=='cat':
        data = cat_data_split(args, data)
    else:
        pass

    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='cat':
        model = CatBoost(args, data)
    else:
        pass

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    # wandb.init(project="f'{model.name}'")
    model.train()

    ######################## INFERENCE
    print(f'--------------- {args.MODEL} PREDICT ---------------')
    if args.MODEL=='cat':
        predicts = model.predict()
    else:
        pass

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ("cat"):
        submission['prediction'] = predicts
    else:
        pass

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('output/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='/opt/ml/input/data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, default='cat', help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')

    ############### TRAINING OPTION
    arg('--EPOCHS', type=int, default=1000, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=0.1, help='Learning Rate를 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    args = parser.parse_args()
    main(args)