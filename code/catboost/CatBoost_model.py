import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import  StratifiedKFold, KFold

class CatBoost:

    def __init__(self, args, data):
        super().__init__()

        # self.X_train = data['X_train']
        # self.y_train = data['y_train']
        # self.X_valid = data['X_valid']
        # self.y_valid = data['y_valid']
        self.tr = data['train']
        self.test = data['test']
        self.users = data['users']
        # self.sub = data['sub']
        self.cat_features = list(range(0, self.tr.shape[1]-1))

        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.seed = args.SEED

        self.model = CatBoostClassifier(iterations=self.epochs, depth=6, learning_rate=self.learning_rate, random_seed=42,
            verbose=50, eval_metric='AUC', task_type='GPU')


    def train(self):
    # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        AUC = []
        cnt = 1
        for train_idx, valid_idx in kf.split(self.users):
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
                early_stopping_rounds=500
            )
            AUC.append(self.model.get_best_score()['validation']['AUC'])
            cnt += 1

        print(f'average AUC: {sum(AUC)/len(AUC)}')


    def predict(self):
        predicts = self.model.predict(self.test)
        # print(self.model.get_all_params)
        return predicts