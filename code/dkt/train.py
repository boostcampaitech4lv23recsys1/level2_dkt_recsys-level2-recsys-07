import os

import torch
import wandb
import numpy as np
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds
from sklearn.model_selection import KFold


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    # train_data, valid_data = preprocess.split_data(args, train_data)

    wandb.init(project="dkt", config=vars(args))
    model = trainer.get_model(args).to(args.device)
    # trainer.run(args, train_data, valid_data, model)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cnt = 1
    for train_idx, valid_idx in kf.split(train_data):
        print(f'===================================  iter: {cnt}  ===================================\n')
        train = np.array([train_data[i] for i in train_idx]).squeeze()
        valid = np.array([train_data[valid_idx]]).squeeze()
        # valid = valid[valid['userID'] != valid['userID'].shift(-1)]

        trainer.run(args, train, valid, model)

        cnt += 1


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
