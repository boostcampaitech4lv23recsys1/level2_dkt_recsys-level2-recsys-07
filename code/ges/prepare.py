# preprocess.py

import random
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder



def get_count(tp, id):
    count_id = tp[[id, 'rating']].groupby(id, as_index=False)
    return count_id.size()


def filter(tp, min_user_count, min_item_count):
    item_count = get_count(tp, 'iid')
    # tp = tp[tp['iid'].isin(item_count.index[item_count >= min_item_count])]

    user_count = get_count(tp, 'uid')
    # tp = tp[tp['uid'].isin(user_count.index[user_count >= min_user_count])]

    user_count, item_count = get_count(tp, 'uid'), get_count(tp, 'iid')
    return tp, user_count, item_count


def numerize(tp, user2id):
    
    uid = list(map(lambda x: user2id[x], tp['uid']))
    tp['uid_new'] = uid
    
    
    le1 = LabelEncoder()
    id_lists = tp["iid"].unique().tolist() + ["unknown"]
    le1.fit(id_lists)
    tp['iid_new'] = tp['iid']
    iid_new = le1.transform(tp['iid_new'].astype(str))
    tp['iid_new'] = iid_new
    
    le2 = LabelEncoder()
    tag_lists = tp["tid"].unique().tolist() + ["unknown"]
    le2.fit(tag_lists)
    tp['tid_new'] = tp['tid']
    tid_new = le2.transform(tp['tid_new'].astype(str))
    tp['tid_new'] = tid_new
    
    return tp

if __name__ == '__main__':
    print('data preprocessing...')
    
    data_dir = '/opt/ml/input/data/'
    train_file_path = os.path.join(data_dir, 'train_data.csv')
    test_file_path = os.path.join(data_dir, 'test_data.csv')

    train_df = pd.read_csv(train_file_path, parse_dates=['Timestamp'])
    test_df = pd.read_csv(test_file_path, parse_dates=['Timestamp']) 
    pre_df = pd.concat([train_df, test_df])

    pre_df = pre_df.sort_values(by=["userID", "Timestamp"], axis=0)
    
    tp = pd.DataFrame()
    tp[['uid', 'iid', 'rating', 'timestamp', 'tid']] = pre_df[['userID','assessmentItemID', 'answerCode', 'Timestamp', 'KnowledgeTag']]
    

    MIN_USER_COUNT = 20
    MIN_ITEM_COUNT = 20
    
    tp, user_count, item_count = filter(tp, min_user_count=MIN_USER_COUNT, min_item_count=MIN_ITEM_COUNT)

    sparsity = float(tp.shape[0]) / user_count.shape[0] / item_count.shape[0]
    print('num_user: %d, num_item: %d, num_interaction: %d, sparsity: %.4f%%' % (user_count.shape[0], item_count.shape[0], tp.shape[0], sparsity * 100))
    
    
    unique_uid = user_count.index
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    all_tp = numerize(tp, user2id)
    
    
    
    
    print('data splitting...')

    all_tp_sorted = all_tp.sort_values(by=['uid_new', 'timestamp', 'iid_new'])

    users, items = np.array(all_tp_sorted['uid_new'], dtype=np.int32), np.array(all_tp_sorted['iid_new'], dtype=np.int32)

    num_user, num_item = max(users) + 1, max(items) + 1

    all_data = defaultdict(list)
    for n in range(len(users)):
        all_data[users[n]].append(items[n])

    train_dict = dict()

    random.seed(42)
    for u in all_data:
        train_dict[u] = all_data[u][:-2]

    print('preprocessed data save at /data/dataset...')
    np.save('/opt/ml/input/code/ges/dataset/preprocessed_data', np.array([train_dict, num_user, num_item]))
    
    
    print('preprocessed data save at /data/dataset_rel...')
    tag_tp_sorted = all_tp.sort_values(by=['tid_new', 'iid_new'])
    grouped_tag = tag_tp_sorted.groupby('tid_new').apply(lambda r: list(set(r['iid_new'].values)))
    rel_dict = grouped_tag.to_dict()
    np.save('/opt/ml/input/code/ges/dataset/preprocessed_data_rel', np.array([rel_dict]))
    
    print('prepare done.')
