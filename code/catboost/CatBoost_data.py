import os
import random
import numpy as np
import pandas as pd
from catboost import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

def cat_data_load(args):

    ######################## DATA LOAD
    train = pd.read_csv(args.DATA_PATH + 'train_data2.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_data2.csv')
    # sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    train['past_count'] = train.groupby('userID').cumcount()
    # user별 푼 문제 총합
    # total_count = train.groupby('userID')[['testId']].count().rename(columns={"testId":"total_count"})
    # train = pd.merge(train, total_count, on='userID', how='left')
    # 과거에 맞춘 문제 수
    train['shift'] = train.groupby('userID')['answerCode'].shift().fillna(0)
    train['past_correct'] = train.groupby('userID')['shift'].cumsum()
    train = train.drop(['shift'], axis=1)
    # 과거 평균 정답률
    train['average_correct'] = (train['past_correct'] / train['past_count']).fillna(0)

    # test["past_count"] = train.groupby("userID")["past_count"].mean()
    # test["total_count"] = train.groupby("userID")["total_count"].count()
    # test["past_correct"] = train.groupby("userID")["past_correct"].mean()
    test["average_correct"] = train.groupby("userID")["average_correct"].mean()

    # test["past_count"] = test["past_count"].fillna(0).astype(int)
    # test["total_count"] = test["total_count"].fillna(0).astype(int)
    # test["past_correct"] = test["past_correct"].fillna(0).astype(int)
    test["average_correct"] = test["average_correct"].fillna(0).astype(int)

    # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", 'past_count', "total_count", "past_correct", "average_correct"]
    cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "average_correct"]
    
    for col in cate_cols:
        le = LabelEncoder()
        # For UNKNOWN class
        a = train[col].unique().tolist() + ["unknown"]
        le.fit(a)
        le_path = os.path.join("/opt/ml/input/code/catboost/asset/", col + "_classes.npy")
        np.save(le_path, le.classes_)

        # 모든 컬럼이 범주형이라고 가정
        train[col] = train[col].astype(str)
        trans = le.transform(train[col])
        train[col] = trans
        
        if col in ["assessmentItemID", "testId", "KnowledgeTag"]:
            test[col] = test[col].astype(str)
            trans = le.transform(test[col])
            test[col] = trans


    train = train.drop(['Timestamp', 'past_count', "past_correct"], axis=1)
    test = test.drop(['answerCode', 'Timestamp'], axis=1)

    data = {
            'train': train,
            'test': test,
            }

    return data

def cat_data_split(args, data, ratio=0.8):    
    users = list(zip(data['train']['userID'].value_counts().index, data['train']['userID'].value_counts()))
    data['users'] = users
    # random.seed(args.SEED)
    # random.shuffle(users)
    
    # max_train_data_len = ratio*len(data['train'])
    # sum_of_train_data = 0
    # user_ids =[]

    # for user_id, count in users:
    #     sum_of_train_data += count
    #     if max_train_data_len < sum_of_train_data:
    #         break
    #     user_ids.append(user_id)

    # train = data['train'][data['train']['userID'].isin(user_ids)]
    # valid = data['train'][data['train']['userID'].isin(user_ids) == False]

    # #valid데이터셋은 각 유저의 마지막 interaction만 추출
    # valid = valid[valid['userID'] != valid['userID'].shift(-1)]

    # X_train = train.drop(['answerCode'], axis=1)
    # y_train = train['answerCode']
    # X_valid = valid.drop(['answerCode'], axis=1)
    # y_valid = valid['answerCode']
    # data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data