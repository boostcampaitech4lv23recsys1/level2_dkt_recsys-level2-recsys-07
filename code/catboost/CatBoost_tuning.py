import pandas as pd
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv("/opt/ml/input/data/train_data.csv", parse_dates=['Timestamp'])
test = pd.read_csv('/opt/ml/input/data/test_data.csv', parse_dates=['Timestamp'])

df = pd.merge(train, test, how='outer').sort_values(by=['userID','Timestamp']).reset_index(drop=True)

## col_name를 기준으로 mean, std, sum을 추가하는 함수.

# col_name를 기준으로 answerCode의 mean, std, sum을 추가하는 함수.
def new_feature_answer(df, col_name:str, new_feature_name:str):
    
    grouped_df = df.groupby(col_name)
    
    mean_series = grouped_df.mean()['answerCode']
    std_series = grouped_df.std()['answerCode']
    sum_series = grouped_df.sum()['answerCode']
    
    
    series2mean = dict()
    for i, v in zip(mean_series.keys(), mean_series.values):
        series2mean[i] = v
        
    series2std = dict()
    for i, v in zip(std_series.keys(), std_series.values):
        series2std[i] = v
        
    series2sum = dict()
    for i, v in zip(sum_series.keys(), sum_series.values):
        series2sum[i] = v
        
    df[f'{new_feature_name}_ans_mean'] = df[col_name].map(series2mean)
    df[f'{new_feature_name}_ans_std'] = df[col_name].map(series2std)
    df[f'{new_feature_name}_ans_sum'] = df[col_name].map(series2sum)
    
    return df


## col_name를 기준으로 elap_time의 mean, std, sum을 추가하는 함수.
def new_feature_time(df, col_name:str, new_feature_name:str):
    
    grouped_df = df.groupby(col_name)
    
    mean_series = grouped_df.mean()['elap_time']
    std_series = grouped_df.std()['elap_time']
    sum_series = grouped_df.sum()['elap_time']
    median_series = grouped_df.median()['elap_time']
    
    
    series2mean = dict()
    for i, v in zip(mean_series.keys(), mean_series.values):
        series2mean[i] = v
        
    series2std = dict()
    for i, v in zip(std_series.keys(), std_series.values):
        series2std[i] = v
        
    series2sum = dict()
    for i, v in zip(sum_series.keys(), sum_series.values):
        series2sum[i] = v

    series2median = dict()
    for i, v in zip(median_series.keys(), median_series.values):
        series2median[i] = v
        
    df[f'{new_feature_name}_time_mean'] = df[col_name].map(series2mean)
    df[f'{new_feature_name}_time_std'] = df[col_name].map(series2std)
    df[f'{new_feature_name}_time_sum'] = df[col_name].map(series2sum)
    df[f'{new_feature_name}_time_median'] = df[col_name].map(series2median)
    
    return df


# 난이도 설정을 위한 ELO 사용
def ELO_function(df):
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name="assessmentItemID"):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print("Parameter estimation is starting...")

        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values,
                answers_df.answerCode.values,
            )
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters

    def gou_func(theta, beta):
        return 1 / (1 + np.exp(-(theta - beta)))

    df["left_asymptote"] = 0

    print(f"Dataset of shape {df.shape}")
    print(f"Columns are {list(df.columns)}")

    student_parameters, item_parameters = estimate_parameters(df)

    prob = [
        gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
        for student, item in zip(df.userID.values, df.assessmentItemID.values)
    ]

    df["elo_prob"] = prob

    return df


def feature_engineering(df):
    # ELO 적용
    df = ELO_function(df)

    # user의 문제 정답수, 풀이수, 정답률을 시간순으로 누적해서 계산 / 전체 평균 정답률
    df['user_cum_correct'] = df.groupby('userID')['answerCode'].cumsum()
    df['user_cum_cnt'] = df.groupby('userID')['answerCode'].cumcount()+1
    df['user_acc'] = (df['user_cum_correct']/df['user_cum_cnt'])
    stu_groupby = df.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': 'sum'
    		  })
    stu_groupby['user_mean'] = stu_groupby['answerCode'] / stu_groupby['assessmentItemID']
    stu_groupby = stu_groupby.reset_index()
    df = df.merge(stu_groupby[['userID','user_mean']], on='userID', how='left')

    # user별 풀어본 총 문제수(total_count)
    total_count = df.groupby('userID')[['testId']].count().rename(columns={"testId":"total_count"})
    df = pd.merge(df, total_count, on='userID', how='left')

    # test 별 정답의 mean, std, sum, user별 푼 종류수, 누적 개수
    df = new_feature_answer(df, 'testId', 'test')
    test_count = df.groupby('userID')[['testId']].nunique().rename(columns={"testId":"test_count"})
    df = pd.merge(df, test_count, on='userID', how='left')
    df["test_cum_correct"] = df.groupby(["userID", "testId"])['answerCode'].cumsum()
    df["test_cum_cnt"] = df.groupby(["userID", "testId"]).cumcount()+1
    df["test_acc"] = (df['test_cum_correct']/df['test_cum_cnt'])

    # tag 별 정답의 mean, std, sum, user별 푼 종류수, 누적 개수
    df = new_feature_answer(df, 'KnowledgeTag', 'tag')
    tag_count = df.groupby('userID')[['KnowledgeTag']].nunique().rename(columns={"KnowledgeTag":"tag_count"})
    df = pd.merge(df, tag_count, on='userID', how='left')
    df["tag_cum_correct"] = df.groupby(["userID", "KnowledgeTag"])['answerCode'].cumsum()
    df["tag_cum_cnt"] = df.groupby(["userID", "KnowledgeTag"]).cumcount()+1
    df["tag_acc"] = (df['tag_cum_correct']/df['tag_cum_cnt'])

    # test 대분류(prefix) 별 정답의 mean, std, sum, user별 푼 종류수, 누적 개수
    df['prefix'] = df.assessmentItemID.map(lambda x: int(x[2:3]))
    df = new_feature_answer(df, 'prefix', 'prefix')
    prefix_count = df.groupby('userID')[['prefix']].nunique().rename(columns={"prefix":"prefix_count"})
    df = pd.merge(df, prefix_count, on='userID', how='left')
    df["prefix_cum_correct"] = df.groupby(["userID", "prefix"])['answerCode'].cumsum()
    df["prefix_cum_cnt"] = df.groupby(["userID", "prefix"]).cumcount()+1
    df["prefix_acc"] = (df['prefix_cum_correct']/df['prefix_cum_cnt'])

    # assessmentID 별 정답의 mean, std, sum, user별 푼 종류수
    df = new_feature_answer(df, 'assessmentItemID', 'assess')
    assess_count = df.groupby('userID')[['assessmentItemID']].nunique().rename(columns={"assessmentItemID":"assess_count"})
    df = pd.merge(df, assess_count, on='userID', how='left')
    df["assess_cum_correct"] = df.groupby(["userID", "assessmentItemID"])['answerCode'].cumsum()
    df["assess_cum_cnt"] = df.groupby(["userID", "assessmentItemID"]).cumcount()+1
    df["assess_acc"] = (df['assess_cum_correct']/df['assess_cum_cnt'])

    # assessmentID 뒤 3자리(suffix) 정답의 mean, std, sum, user별 푼 종류수
    df['suffix'] = df.assessmentItemID.map(lambda x: int(x[-3:]))
    df = new_feature_answer(df, 'suffix', 'suffix')
    suffix_count = df.groupby('userID')[['suffix']].nunique().rename(columns={"suffix":"suffix_count"})
    df = pd.merge(df, suffix_count, on='userID', how='left')
    df["suffix_cum_correct"] = df.groupby(["userID", "suffix"])['answerCode'].cumsum()
    df["suffix_cum_cnt"] = df.groupby(["userID", "suffix"]).cumcount()+1
    df["suffix_acc"] = (df['suffix_cum_correct']/df['suffix_cum_cnt'])

    # weekday, month, day, hour 별 정답의 mean, std, sum
    df['weekday'] = df['Timestamp'].dt.weekday
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df = new_feature_answer(df,'weekday', 'weekday' )
    df = new_feature_answer(df,'month', 'month' )
    df = new_feature_answer(df,'day', 'day' )
    df = new_feature_answer(df,'hour', 'hour' )

    # 시간관련 feature engineering
    # user가 각 문제를 푸는 데 걸린 시간
    solving_time = df[['userID', 'Timestamp']].groupby('userID').diff(periods=-1).fillna(pd.Timedelta(seconds=0))
    solving_time = solving_time['Timestamp'].apply(lambda x: x.total_seconds())
    df['elap_time'] = -solving_time
    df['elap_time'] = df['elap_time'].map(lambda x: int(x) if 0 < x <= 3600 else int(89))

    # 각 기본 feature에 대한 elap_time의 mean, std, sum, median, 상대 시간(중앙값 - 풀이 시간)
    df = new_feature_time(df, 'userID', 'user')
    df = new_feature_time(df, 'testId', 'test')
    df = new_feature_time(df, 'KnowledgeTag', 'tag')
    df = new_feature_time(df, 'prefix', 'prefix')
    df = new_feature_time(df, 'assessmentItemID', 'assess')
    df = new_feature_time(df, 'suffix', 'suffix')
    
    df['user_relatvie_time'] = df['user_time_median'] - df['elap_time']
    df['test_relatvie_time'] = df['test_time_median'] - df['elap_time']
    df['tag_relatvie_time'] = df['tag_time_median'] - df['elap_time']
    df['prefix_relatvie_time'] = df['prefix_time_median'] - df['elap_time']
    df['assess_relatvie_time'] = df['assess_time_median'] - df['elap_time']
    df['suffix_relatvie_time'] = df['suffix_time_median'] - df['elap_time']

    # 맞은 row와 틀린 row를 분리
    df_o = df[df['answerCode']==1]
    df_x = df[df["answerCode"]==0]
    basic_feats = ['userID','assessmentItemID',"testId",'answerCode','Timestamp','KnowledgeTag']

    user_o_elp = new_feature_time(df_o, 'userID', 'user_o')
    user_o_elp['user_o_relatvie_time'] = user_o_elp['user_o_time_median'] - user_o_elp['elap_time']
    df = pd.merge(df, user_o_elp[basic_feats+['user_o_time_mean', 'user_o_time_std', 'user_o_time_sum', 'user_o_time_median', 'user_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    user_x_elp = new_feature_time(df_x, 'userID', 'user_x')
    user_x_elp['user_x_relatvie_time'] = user_x_elp['user_x_time_median'] - user_x_elp['elap_time']
    df = pd.merge(df, user_x_elp[basic_feats+['user_x_time_mean', 'user_x_time_std', 'user_x_time_sum', 'user_x_time_median', 'user_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)

    test_o_elp = new_feature_time(df_o, 'testId', 'test_o')
    test_o_elp['test_o_relatvie_time'] = test_o_elp['test_o_time_median'] - test_o_elp['elap_time']
    df = pd.merge(df, test_o_elp[basic_feats+['test_o_time_mean', 'test_o_time_std', 'test_o_time_sum', 'test_o_time_median', 'test_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    test_x_elp = new_feature_time(df_x, 'testId', 'test_x')
    test_x_elp['test_x_relatvie_time'] = test_x_elp['test_x_time_median'] - test_x_elp['elap_time']
    df = pd.merge(df, test_x_elp[basic_feats+['test_x_time_mean', 'test_x_time_std', 'test_x_time_sum', 'test_x_time_median', 'test_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)

    tag_o_elp = new_feature_time(df_o, 'KnowledgeTag', 'tag_o')
    tag_o_elp['tag_o_relatvie_time'] = tag_o_elp['tag_o_time_median'] - tag_o_elp['elap_time']
    df = pd.merge(df, tag_o_elp[basic_feats+['tag_o_time_mean', 'tag_o_time_std', 'tag_o_time_sum', 'tag_o_time_median', 'tag_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    tag_x_elp = new_feature_time(df_x, 'KnowledgeTag', 'tag_x')
    tag_x_elp['tag_x_relatvie_time'] = tag_x_elp['tag_x_time_median'] - tag_x_elp['elap_time']
    df = pd.merge(df, tag_x_elp[basic_feats+['tag_x_time_mean', 'tag_x_time_std', 'tag_x_time_sum', 'tag_x_time_median', 'tag_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)

    prefix_o_elp = new_feature_time(df_o, 'prefix', 'prefix_o')
    prefix_o_elp['prefix_o_relatvie_time'] = prefix_o_elp['prefix_o_time_median'] - prefix_o_elp['elap_time']
    df = pd.merge(df, prefix_o_elp[basic_feats+['prefix_o_time_mean', 'prefix_o_time_std', 'prefix_o_time_sum', 'prefix_o_time_median', 'prefix_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    prefix_x_elp = new_feature_time(df_x, 'prefix', 'prefix_x')
    prefix_x_elp['prefix_x_relatvie_time'] = prefix_x_elp['prefix_x_time_median'] - prefix_x_elp['elap_time']
    df = pd.merge(df, prefix_x_elp[basic_feats+['prefix_x_time_mean', 'prefix_x_time_std', 'prefix_x_time_sum', 'prefix_x_time_median', 'prefix_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)

    assess_o_elp = new_feature_time(df_o, 'assessmentItemID', 'assess_o')
    assess_o_elp['assess_o_relatvie_time'] = assess_o_elp['assess_o_time_median'] - assess_o_elp['elap_time']
    df = pd.merge(df, assess_o_elp[basic_feats+['assess_o_time_mean', 'assess_o_time_std', 'assess_o_time_sum', 'assess_o_time_median', 'assess_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    assess_x_elp = new_feature_time(df_x, 'assessmentItemID', 'assess_x')
    assess_x_elp['assess_x_relatvie_time'] = assess_x_elp['assess_x_time_median'] - assess_x_elp['elap_time']
    df = pd.merge(df, assess_x_elp[basic_feats+['assess_x_time_mean', 'assess_x_time_std', 'assess_x_time_sum', 'assess_x_time_median', 'assess_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)

    suffix_o_elp = new_feature_time(df_o, 'suffix', 'suffix_o')
    suffix_o_elp['suffix_o_relatvie_time'] = suffix_o_elp['suffix_o_time_median'] - suffix_o_elp['elap_time']
    df = pd.merge(df, suffix_o_elp[basic_feats+['suffix_o_time_mean', 'suffix_o_time_std', 'suffix_o_time_sum', 'suffix_o_time_median', 'suffix_o_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    suffix_x_elp = new_feature_time(df_x, 'suffix', 'suffix_x')
    suffix_x_elp['suffix_x_relatvie_time'] = suffix_x_elp['suffix_x_time_median'] - suffix_x_elp['elap_time']
    df = pd.merge(df, suffix_x_elp[basic_feats+['suffix_x_time_mean', 'suffix_x_time_std', 'suffix_x_time_sum', 'suffix_x_time_median', 'suffix_x_relatvie_time']],
            on=basic_feats, how="left").fillna(0)
    
    # test 푸는데 걸린 시간 총합
    user_solve_test_time = df.groupby(['userID', 'testId'])['elap_time'].sum()\
                                                 .groupby(level=0).cumsum().reset_index()
    user_solve_test_time.rename(columns={"elap_time":"test_solve_time"}, inplace=True)
    df = df.merge(user_solve_test_time, on=['userID','testId'], how='left')

    # 현 시점 user의 학습 누적 시간
    df['total_used_time'] = df.groupby('userID')['elap_time'].cumsum()

    # user의 최근 3문제, 최근 5문제 문제풀이 평균시간
    df['recent3_elap_time'] = df.groupby(['userID'])['elap_time'].rolling(3).mean().fillna(0).values

    # user나 prefix가 바뀔때마다 시간 리셋
    time_df = df[["userID", "prefix", "Timestamp"]].sort_values(by=["userID", "prefix", "Timestamp"])
    time_df["userID_reset"] = time_df["userID"] != time_df["userID"].shift(1)
    time_df["prefix_reset"] = time_df["prefix"] != time_df["prefix"].shift(1)
    time_df["first"] = time_df[["userID_reset", "prefix_reset"]].any(axis=1).apply(lambda x: 1 - int(x))
    time_df["reset_time"] = time_df["Timestamp"].diff().fillna(pd.Timedelta(seconds=0))
    time_df["reset_time"] = (
        time_df["reset_time"].apply(lambda x: x.total_seconds()) * time_df["first"]
    )
    df["reset_time"] = time_df["reset_time"]#.apply(lambda x: math.log(x + 1))
    
    return df

df = feature_engineering(df)
train = df[df['answerCode']!=-1]
test = df[df['answerCode']==-1]

train = train.drop(['left_asymptote',], axis=1)
test = test.drop(['answerCode','left_asymptote'], axis=1)

users = list(zip(train['userID'].value_counts().index, train['userID'].value_counts()))
sampler = TPESampler(seed=1)

def objective(trial):

    cbrm_param = {
        'iterations':trial.suggest_int("iterations", 1000, 10000), # best: 4575, 4936, 
        # 'od_wait':trial.suggest_int('od_wait', 1, 500), # 변화 없음
        'learning_rate' : trial.suggest_uniform('learning_rate',0.05, 1), # best: 0.6014863476848088, 0.4603434159510499
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100), # best: 86.00892526265588
        # 'subsample': trial.suggest_uniform('subsample',0,1), # Bayesian에서는 적용 안됨
        'random_strength': trial.suggest_uniform('random_strength',0,50), # best: 43.206283882145115
        'depth': trial.suggest_int('depth',1, 15), # best: 6 (default)
        # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30), # 변화 없음
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,100), # best: 1
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), # best: 0.09717630812870576
        # 'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.01, 1.0), # GPU에서 적용 안됨
        'task_type':'GPU'
    }

    model = CatBoostRegressor(**cbrm_param)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    AUC = []
    cnt = 1
    for train_idx, valid_idx in kf.split(users):
        print(f'===================================  iter: {cnt}  ===================================\n')
        tr = train[train['userID'].isin(train_idx)]
        valid = train[train['userID'].isin(valid_idx)]
        valid = valid[valid['userID'] != valid['userID'].shift(-1)]

        X_train = tr.drop(['answerCode'], axis=1)
        y_train = tr['answerCode']
        X_valid = valid.drop(['answerCode'], axis=1)
        y_valid = valid['answerCode']

        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', ],
            early_stopping_rounds=1000,
            verbose=100
        )
        cnt += 1

    # Generate model
    
                           
	# 평가지표 원하는 평가 지표가 있을 시 바꾸어 준다.
    AUC = roc_auc_score(y_valid, model.predict(X_valid))
    return AUC


opt_study = optuna.create_study(direction='maximize', sampler=sampler)
opt_study.optimize(objective, n_trials=50)
opt_trial = opt_study.best_trial
opt_trial_params = opt_trial.params
print('Best Trial: score {},\nparams {}'.format(opt_trial.value, opt_trial_params))