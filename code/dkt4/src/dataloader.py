import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "large_paper_number2idx", "hour", "dayofweek"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        # TODO


        # 시험지 대분류
        def get_large_paper_number(x):
            return x[1:4]
        df['large_paper_number2idx'] = df['assessmentItemID'].apply(lambda x : get_large_paper_number(x))

        # 문제 푸는데 걸린 시간
        def get_now_elapsed(df):
            
            diff = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
            diff = diff.fillna(pd.Timedelta(seconds=0))
            diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
            df['now_elapsed'] = diff
            df['now_elapsed'] = df['now_elapsed'].apply(lambda x : x if x < 650 and x >=0 else 0)
            df['now_elapsed'] = df['now_elapsed']

            return df

        df = get_now_elapsed(df = df)

        # 문항별 정답률
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_mean_answerCode'] = df[df['answerCode'] != -1].groupby('assessmentItemID').mean()['answerCode']
        df = df.reset_index(drop = False)

        # 문항별 정답률 표준편차
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_std_answerCode'] = df[df['answerCode'] != -1].groupby('assessmentItemID').std()['answerCode']
        df = df.reset_index(drop = False)

        # 올바르게 푼 사람들의 문항별 풀이 시간 평균
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_mean_now_elapsed'] = df[df['answerCode'] == 1].groupby('assessmentItemID').mean()['now_elapsed']
        df = df.reset_index(drop = False)

        # 올바르게 푼 사람들의 문항별 풀이 시간 표준 편차
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_std_now_elapsed'] = df[df['answerCode'] == 1].groupby('assessmentItemID').std()['now_elapsed']
        df = df.reset_index(drop = False)

        # 문제 푼 시간
        df['hour'] = df['Timestamp'].dt.hour

        # 문제 푼 요일
        df['dayofweek'] = df['Timestamp'].dt.dayofweek

        # # 문제 푼 시간
        # sin_val = np.sin(2 * np.pi * np.array([i for i in range(1, 25)]) / 24)
        # cos_val = np.cos(2 * np.pi * np.array([i for i in range(1, 25)]) / 24)

        # df['num_hour'] = 2 * np.pi * (df['hour'] + 1) / 24
        # df['sin_num_hour'] = np.sin(df['num_hour']) + 2 * abs(sin_val.min())
        # df['cos_num_hour'] = np.cos(df['num_hour']) + 2 * abs(cos_val.min())

        # # 문제 푼 요일
        # sin_val = np.sin(2 * np.pi * np.array([i for i in range(1, 8)]) / 7)
        # cos_val = np.cos(2 * np.pi * np.array([i for i in range(1, 8)]) / 7)

        # df['num_dayofweek'] = 2 * np.pi * (df['dayofweek'] + 1) / 7
        # df['sin_num_dayofweek'] = np.sin(df['num_dayofweek']) + 2 * abs(sin_val.min())
        # df['cos_num_dayofweek'] = np.cos(df['num_dayofweek']) + 2 * abs(cos_val.min())

        return df


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])  # , nrows=100000)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
# ["assessmentItemID", "testId", "KnowledgeTag", "large_paper_number2idx", "hour", "dayofweek"]
        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        self.args.n_large_paper_number2idx = len(
            np.load(os.path.join(self.args.asset_dir, "large_paper_number2idx_classes.npy"))
        )
        self.args.n_hour = len(
            np.load(os.path.join(self.args.asset_dir, "hour_classes.npy"))
        )
        self.args.n_dayofweek = len(
            np.load(os.path.join(self.args.asset_dir, "dayofweek_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "KnowledgeTag", "large_paper_number2idx", "hour", "dayofweek", "now_elapsed", "assessmentItemID_mean_now_elapsed", "assessmentItemID_std_now_elapsed", "assessmentItemID_mean_answerCode", "assessmentItemID_std_answerCode", 'answerCode']
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r['large_paper_number2idx'].values,
                    r['hour'].values,
                    r['dayofweek'].values,
                    r['now_elapsed'].values,
                    r['assessmentItemID_mean_now_elapsed'].values,
                    r['assessmentItemID_std_now_elapsed'].values,
                    r['assessmentItemID_mean_answerCode'].values,
                    r['assessmentItemID_std_answerCode'].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, large_paper_number2idx, hour, dayofweek, now_elapsed, assessmentItemID_mean_now_elapsed, assessmentItemID_std_now_elapsed, assessmentItemID_mean_answerCode, assessmentItemID_std_answerCode, correct = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]
        
        cate_cols = [test, question, tag, large_paper_number2idx, hour, dayofweek, now_elapsed,
         assessmentItemID_mean_now_elapsed, assessmentItemID_std_now_elapsed,
         assessmentItemID_mean_answerCode, assessmentItemID_std_answerCode, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
