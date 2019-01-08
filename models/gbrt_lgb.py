from some_libs import *
# import ctypes
from ctypes import *

np.set_printoptions(linewidth=np.inf)
def display_importances(feature_importance_df):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()

def Y_t(y_train,np_type):
    # print(type(y_train))
    if type(y_train) is pd.Series:
        np_target = y_train.values.astype(np_type)
    else:
        np_target = y_train.astype(np_type)
    return np_target

def X_t(X_train_0,np_type):
    # mort_fit需要feature优先
    X_train = np.asfortranarray(X_train_0)
    # Transpose just changes the strides, it doesn't touch the actual array
    data = X_train.astype(np_type)  # .transpose()
    return data

#v0.2
def LiteMORT_test( X_train_0, y_train,X_test, y_test, flag ):
    gc.collect()
    print("====== LiteMORT_test X_train_0={} y_train={}......".format(X_train_0.shape,y_train.shape))
    train_y = Y_t(y_train, np.float32)
    eval_y = Y_t(y_test, np.float32)
    train_X = X_t(X_train_0,np.float32)
    eval_X = X_t(X_test, np.float32)
    nTrain, nFeat,nTest = train_X.shape[0], train_X.shape[1],eval_X.shape[0]

    dll_path = 'F:/Project/LiteMORT/LiteMORT.dll'
    if False:
        arr_path = "../input/df_ndarray.csv"
        np.savetxt(arr_path, data, delimiter="", fmt='%12g', )
        print("====== arr_file@{} size={} dll={}".format(arr_path, data.shape, dll_path))
    dll = cdll.LoadLibrary(dll_path)
    mort_fit = dll.LiteMORT_fit
    mort_fit.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t,POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
    mort_fit(train_X.ctypes.data_as(POINTER(c_float)), train_y.ctypes.data_as(POINTER(c_float)), nFeat, nTrain,
             eval_X.ctypes.data_as(POINTER(c_float)), eval_y.ctypes.data_as(POINTER(c_float)),nTest, 0)             # 1 : classification
    quit()

def lgb_test(X_train_0,y_train_0,X_test,  y_test):
    X_train, X_eval, y_train, y_eval = train_test_split(X_train_0, y_train_0, random_state=42, test_size=0.1, shuffle=True)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval)
    print( "train[{}] head=\n{}\neval[{}] head=\n{} ".
           format(X_train.shape, X_train[0:5, :], X_eval.shape, X_eval[0:5, :]))
    # specify your configurations as a dict
    params = {  #https://lightgbm.readthedocs.io/en/latest/Parameters.html
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        #'metric': {'l2', 'auc'},
        'metric': 'l2',
        'metric': 'rmse',       #'l2_root'
        'num_leaves': 31,
        #'max_depth': 1,
        'learning_rate': 0.01,
        #'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': 0,
        'nthread': 6,
    }

    print('Start training...')
    # train
    evals_result = {}  # to record eval results for plotting
    t0=time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    verbose_eval=1,
                    valid_sets=lgb_eval,
                    evals_result=evals_result,
                    early_stopping_rounds=20)

    print('|y_test|={:.3g} best_iter={}, time={} Save model...'.format(np.linalg.norm(y_test),gbm.best_iteration,time.time()-t0))
    # save model to file
    gbm.save_model('model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    if True:
        print('Plot metrics recorded during training...')
        ax = lgb.plot_metric(evals_result)
        plt.show()

    print('Plot feature importances...')
    ax = lgb.plot_importance(gbm)
    plt.show()

    if False:
        print('Plot 84th tree...')  # one tree use categorical feature to split
        ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
        plt.show()
        print('Plot 84th tree with graphviz...')
        graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
        graph.render(view=True)

    return y_pred