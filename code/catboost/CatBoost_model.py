import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import  StratifiedKFold, KFold,TimeSeriesSplit 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


class CatBoost:

    def __init__(self, args, data):
        super().__init__()

        self.tr = data['train']
        self.test = data['test']
        self.users = data['users']
        self.cat_features = self.test.select_dtypes(include=['int','object']).columns.tolist()

        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.seed = args.SEED

        self.model = CatBoostRegressor(iterations=self.epochs, depth=6, learning_rate=self.learning_rate, random_seed=42,
            # reg_lambda=86, random_strength=43, leaf_estimation_iterations= 1, bagging_temperature=0.1, 
            verbose=50, eval_metric='AUC', task_type='GPU')


    def train(self):
    # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        tsp = TimeSeriesSplit(n_splits=5)
        AUC = []
        cnt = 1
        FEATS = self.test.columns.tolist()
        for train_idx, valid_idx in tsp.split(self.users):
            print(f'===================================  iter: {cnt}  ===================================\n')
            tr = self.tr[self.tr['userID'].isin(train_idx)]
            valid = self.tr[self.tr['userID'].isin(valid_idx)]
            valid = valid[valid['userID'] != valid['userID'].shift(-1)]

            X_train = tr.drop(['answerCode'], axis=1)
            y_train = tr['answerCode']
            X_valid = valid.drop(['answerCode'], axis=1)
            y_valid = valid['answerCode']

            self.model.fit(
                X_train, y_train,
                cat_features=self.cat_features,
                eval_set=(X_valid, y_valid),
                # plot=True,
                early_stopping_rounds=100
            )
            AUC.append(self.model.get_best_score()['validation']['AUC'])
            cnt += 1

            # explainer = shap.TreeExplainer(self.model)
            # shap_values = explainer.shap_values(X_train[FEATS])

        print(f'average AUC: {sum(AUC)/len(AUC)}\n')

        #Create arrays from feature importance and feature names
        importance = self.model.get_feature_importance()
        imp = []
        for i, n in enumerate(importance):
            imp.append((str(X_train.columns[i]), importance[i]))
        imp = sorted(imp, key=lambda x:-x[1])
        print(f'===================================  Feature Importance  ===================================')
        print(imp)
        print('\n')

        # shap.summary_plot(shap_values, self.test[FEATS])

    
    def predict(self):
        predicts = self.model.predict(self.test)
        # print(self.model.get_all_params)
        return predicts#[:,1]