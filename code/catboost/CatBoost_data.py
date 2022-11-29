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
    total_count = train.groupby('userID')[['testId']].count().rename(columns={"testId":"total_count"})
    train = pd.merge(train, total_count, on='userID', how='left')
    # user별 푼 시험지, 문제, 태그 종류 수
    test_count = train.groupby('userID')[['testId']].nunique().rename(columns={"testId":"test_count"})
    assess_count = train.groupby('userID')[['assessmentItemID']].nunique().rename(columns={"assessmentItemID":"assess_count"})
    tag_count = train.groupby('userID')[['KnowledgeTag']].nunique().rename(columns={"KnowledgeTag":"tag_count"})
    test_correct = (train.groupby('testId')[['answerCode']].sum()*10000 / 
                train.groupby('testId')[['answerCode']].count()).rename(columns={'answerCode':'test_correct'})
    assess_correct = (train.groupby('assessmentItemID')[['answerCode']].sum()*10000 / 
                    train.groupby('assessmentItemID')[['answerCode']].count()).rename(columns={'answerCode':'assess_correct'})
    train = pd.merge(train, test_count, on='userID', how='left')
    train = pd.merge(train, assess_count, on='userID', how='left')
    train = pd.merge(train, tag_count, on='userID', how='left')
    train = pd.merge(train, test_correct, on='testId', how='left')
    train = pd.merge(train, assess_correct, on='assessmentItemID', how='left')
    train['test_correct'] = train['test_correct'].fillna(0).astype(int)
    train['assess_correct'] = train['assess_correct'].fillna(0).astype(int)

    # 과거에 맞춘 문제 수
    train['shift'] = train.groupby('userID')['answerCode'].shift().fillna(0)
    train['past_correct'] = train.groupby('userID')['shift'].cumsum().astype(int)
    train = train.drop(['shift'], axis=1)
    # 과거 평균 정답률
    # train['average_correct'] = (train['past_correct'] / train['past_count']).fillna(0).astype(int)
    train['avg_cor'] = (train.groupby(['userID'])['answerCode'].sum()*10000 / train.groupby(['userID'])['answerCode'].count())
    train['avg_cor'] = train['avg_cor'].fillna(0).astype(int)

    test["past_count"] = train.groupby("userID")["past_count"].mean()
    test["total_count"] = train.groupby("userID")["total_count"].count()
    test = pd.merge(test, test_count, on='userID', how='left')
    test = pd.merge(test, assess_count, on='userID', how='left')
    test = pd.merge(test, tag_count, on='userID', how='left')
    test = pd.merge(test, test_correct, on='testId', how='left')
    test = pd.merge(test, assess_correct, on='assessmentItemID', how='left')
    test['test_correct'] = test['test_correct'].fillna(0).astype(int)
    test['assess_correct'] = test['assess_correct'].fillna(0).astype(int)

    test["past_correct"] = train.groupby("userID")["past_correct"].mean()
    # test["average_correct"] = train.groupby("userID")["average_correct"].mean()
    test["avg_cor"] = train.groupby("userID")["avg_cor"].mean()

    test["past_count"] = test["past_count"].fillna(0).astype(int)
    test["past_correct"] = test["past_correct"].fillna(0).astype(int)
    # test["average_correct"] = test["average_correct"].fillna(0).astype(int)
    test['avg_cor'] = test["avg_cor"].fillna(0).astype(int)

    cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
    
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
        
        test[col] = test[col].astype(str)
        trans = le.transform(test[col])
        test[col] = trans


    train = train.drop(['Timestamp', "past_count", "assess_count", "past_correct"], axis=1)
    test = test.drop(['answerCode', 'Timestamp', "past_count", "assess_count", "past_correct"], axis=1)

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