import os

import torch
import wandb
import numpy as np
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds
from datetime import datetime


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    # train_data, valid_data = preprocess.split_data(args, train_data)

    name = datetime.now().strftime('%H%M%S')+"_"+ args.model
    wandb.init(project="dkt", config=vars(args), name=name)
    model = trainer.get_model(args).to(args.device)
    # trainer.run(args, train_data, valid_data, model)
    trainer.run_with_vaild_loss(args, train_data, valid_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
