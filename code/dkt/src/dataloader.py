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

    def split_data(self, args, data, ratio=0.8, shuffle=True):
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
        # cate_cols = df.columns.tolist()
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        

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

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df, is_train):
        # TODO
        # if is_train:
        #     # 푼 문제 수 누적합
        #     df['past_count'] = df.groupby('userID').cumcount()
        #     # user별 푼 문제 총합
        #     total_count = df.groupby('userID')[['testId']].count().rename(columns={"testId":"total_count"})
        #     df = pd.merge(df, total_count, on='userID', how='left')
        #     # user별 푼 시험지, 문제, 태그 종류 수
        #     test_count = df.groupby('userID')[['testId']].nunique().rename(columns={"testId":"test_count"})
        #     assess_count = df.groupby('userID')[['assessmentItemID']].nunique().rename(columns={"assessmentItemID":"assess_count"})
        #     tag_count = df.groupby('userID')[['KnowledgeTag']].nunique().rename(columns={"KnowledgeTag":"tag_count"})
        #     test_correct = (df.groupby('testId')[['answerCode']].sum()*1000 / 
        #                 df.groupby('testId')[['answerCode']].count()).rename(columns={'answerCode':'test_correct'})
        #     assess_correct = (df.groupby('assessmentItemID')[['answerCode']].sum()*1000 / 
        #                     df.groupby('assessmentItemID')[['answerCode']].count()).rename(columns={'answerCode':'assess_correct'})
        #     df = pd.merge(df, test_count, on='userID', how='left')
        #     df = pd.merge(df, assess_count, on='userID', how='left')
        #     df = pd.merge(df, tag_count, on='userID', how='left')
        #     df = pd.merge(df, test_correct, on='testId', how='left')
        #     df = pd.merge(df, assess_correct, on='assessmentItemID', how='left')
        #     df['test_correct'] = df['test_correct'].fillna(0).astype(int)
        #     df['assess_correct'] = df['assess_correct'].fillna(0).astype(int)

        #     # 과거에 맞춘 문제 수
        #     df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
        #     df['past_correct'] = df.groupby('userID')['shift'].cumsum()
        #     # 과거 평균 정답률
        #     # df['average_correct'] = (df['past_correct'] / df['past_count']).fillna(0)
        #     df['avg_cor'] = (df.groupby(['userID'])['answerCode'].sum()*1000 / df.groupby(['userID'])['answerCode'].count())
        #     df['avg_cor'] = df['avg_cor'].fillna(0).astype(int)

        #     df = df.drop(['shift', "past_count", "assess_count", "past_correct","total_count", "test_count", "assess_count", "tag_count", 'test_correct', 'assess_correct', 'avg_cor'], axis=1) 
        #     #
        #     print(f'features: {df.columns.tolist()}')
        # else:
        #     # df = df.drop([ "test_count", "tag_count", 'test_correct', 'assess_correct', 'avg_cor'], axis=1)
        #     #"total_count",
        #     pass
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        # self.args.total_count = df['total_count'].max() + 1         #TODO
        # self.args.test_count = df['test_count'].max() + 1
        # self.args.tag_count = df['tag_count'].max() + 1
        # self.args.test_correct = df['test_correct'].max() + 1
        # self.args.assess_correct = df['assess_correct'].max() + 1
        # self.args.avg_cor = df['avg_cor'].max() + 1


        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag",        #TODO
        #         "total_count", "test_count", "tag_count", 'test_correct', 'assess_correct', 'avg_cor']
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,                 
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    # r["total_count"].values,          #TODO
                    # r["test_count"].values,
                    # r["tag_count"].values,
                    # r["test_correct"].values,
                    # r["assess_correct"].values,
                    # r["avg_cor"].values,
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

        # test, question, tag, correct, total_cnt, test_cnt, tag_cnt, test_cor, ass_cor, avg_cor =\
        # row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]              #TODO
        test, question, tag, correct = row[0], row[1], row[2], row[3]

        # cate_cols = [test, question, tag, correct, total_cnt, test_cnt, tag_cnt, test_cor, ass_cor, avg_cor]        #TODO
        cate_cols = [test, question, tag, correct]

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
