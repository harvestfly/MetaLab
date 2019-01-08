'''
    https://www.kaggle.com/shep312/lightgbm-with-weighted-averages-dropout-787/code
    https://www.kaggle.com/davidsalazarv95/fast-ai-pytorch-starter/notebook
    Adaptive normalization
    https://discuss.pytorch.org/t/adaptive-normalization/9157/6
    https://www.zhihu.com/question/274716261
    Face Recognition via Centralized Coordinate Learning(对特征层做白化) https://blog.csdn.net/u014230646/article/details/79489795
'''
import lightgbm as lgbm
from some_libs import *
from torch.utils.data import DataLoader as torch_dl
from torch.utils.data import Dataset
from torch import nn
from torch import optim

import torch.onnx
import torchvision
from SomeModules import *
# https://github.com/facebookresearch/visdom/blob/master/example/demo.py vis.line的一些参数
# 启动python -m visdom.server
import visdom
import pickle
from config import *
from utils.visualize import *
import scipy

isNN = True
version = "v0.1"
input_dir = os.path.join(os.pardir, 'input')


def init_emb_size(categorical_feats, train_df, test_df):
    assert len(categorical_feats) > 0
    merged_df = pd.concat([train_df, test_df])
    print("\tmerged_df={} train_df={} test_df={}".format(merged_df.shape, train_df.shape, test_df.shape))
    emb_szs = get_embs_dims(merged_df, categorical_feats)
    del merged_df;
    gc.collect()
    print("cat={}\nemb_szs={}".format(categorical_feats, emb_szs))
    return emb_szs


class EmbeddingDataset(Dataset):
    ### This dataset will prepare inputs cats, conts and output y
    ### To be feed into our mixed input embedding fully connected NN model
    ### Stacks numpy arrays to create nxm matrices where n = rows, m = columns
    ### Gives y 0 if not specified
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        # self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n, 1))
        # self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))

        self.conts = conts.astype(np.float32)
        self.n_conts = self.conts.shape[1]
        self.y = np.zeros((n, 1)) if y is None else y.astype(np.float32)  # binary_cross_entropy需要FloatTensor
        # self.y = np.zeros((n, 1)) if y is None else y[:, None].astype(np.int64)
        self.cats = cats

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.cats is None:
            return [self.conts[idx], self.y[idx]]
        else:
            return [self.cats[idx], self.conts[idx], self.y[idx]]

    # @classmethod
    # def from_data_frames(cls, df_cat, df_cont, y=None):
    #   cat_cols = [c.values for n, c in df_cat.items()]
    #   cont_cols = [c.values for n, c in df_cont.items()]
    #   return cls(cat_cols, cont_cols, y)


class ModelData():
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path, self.trn_dl, self.val_dl, self.test_dl = path, trn_dl, val_dl, test_dl


class EmbeddingModelData(ModelData):
    ### This class provides training and validation dataloaders,    Which we will use in our model
    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None):
        self.n_conts = trn_ds.n_conts
        test_dl = torch_dl(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        super().__init__(path, torch_dl(trn_ds, batch_size=bs, shuffle=True, num_workers=1)
                         , torch_dl(val_ds, batch_size=bs, shuffle=True, num_workers=1), test_ds)

    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, batch_size, test_df=None):
        assert cat_flds is None
        # test_ds = EmbeddingDataset.from_data_frame(test_df, cat_flds) if test_df is not None else None
        # eds_1 = EmbeddingDataset.from_data_frame(trn_df, cat_flds, trn_y)
        # eds_2 = EmbeddingDataset.from_data_frame(val_df, cat_flds, val_y)
        eds_1 = EmbeddingDataset(None, trn_df, trn_y)
        eds_2 = EmbeddingDataset(None, val_df, val_y)
        test_ds = EmbeddingDataset(None, test_df, None) if test_df is not None else None
        return cls(path, eds_1, eds_2, batch_size, test_ds=test_ds)

    @classmethod
    def from_data_frame(cls, path, val_idxs, trn_idxs, df, y, cat_flds, bs, test_df=None):
        val_df, val_y = df.iloc[val_idxs], y[val_idxs]
        trn_df, trn_y = df.iloc[trn_idxs], y[trn_idxs]
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, test_df)

    # 感觉比RandomOverSampler好一些
    def get_imbalance(self, config_, dataset,
                      rPosi=0.5):  # Get random selection of data for batch GD. Upsample positive classes to make it balanced in the training batch
        batch_size = config_.batch_size
        nPosi = int(np.round(batch_size * rPosi));
        assert nPosi > 0 and nPosi < batch_size
        y = dataset.y
        pos_idx = np.random.choice(np.where(y[:, 0] == 1)[0], size=nPosi)
        neg_idx = np.random.choice(np.where(y[:, 0] == 0)[0], size=batch_size - nPosi)
        idx = np.concatenate([pos_idx, neg_idx])
        x_cats = torch.LongTensor(dataset.cats[idx, :])
        x_conts = torch.Tensor(dataset.conts[idx, :])
        y = torch.Tensor(y[idx, :])
        return x_cats, x_conts, y


def embedding_train(config_,vis,env_title, model, model_data, optimizer, criterion, epochs, eval_y):
    print("\n========embedding_train env_title={} config=......\n{}\n".format(env_title, config_.__dict__))
    if config_.use_gpu:
        model.cuda()
    trn_set = model_data.trn_dl.dataset
    n_iterations = (int)(trn_set.y.shape[0] / config_.batch_size)
    nSamp = eval_y.shape[0]
    all_predict_path="F:/Project/MetaLab/checkpoints/BN({})/PREDICT_{}.pickle".format(config_.use_bn,config_.loss_curve_title)

    # print(scipy.stats.itemfreq(eval_y))
    t0 = time.time()
    for epoch in range(epochs):
        no, loss_1, loss_2 = 0, 0, 0
        # for iteration in range(n_iterations):
        # x_cats, x_conts, y = model_data.get_imbalance(config_,trn_set,rPosi=0.5)
        model.train()
        for data in iter(model_data.trn_dl):
            x_cats = None
            x_conts, y = data
            if config_.use_gpu:
                x_conts = x_conts.cuda();                y = y.cuda();
            # wrap with variable
            # x_cats, x_conts, y = Variable(x_cats), Variable(x_conts), Variable(y)

            optimizer.zero_grad()
            # if no==0:                print("x_cats={}".format(x_cats))      #print("x_cats={}\ny={}".format(x_cats,y))
            no = no + 1
            outputs = model(x_cats, x_conts)
            loss = criterion(outputs, y)
            loss.backward();
            optimizer.step()
            a = loss.item();
            loss_1 += a
            if no % 2000 == 0:
                print("\t\t{}\tloss({})={}".format(no, loss.grad_fn, a))
        loss_1/=no
        if True:
            model.eval()
            x_cats = None       #torch.LongTensor(model_data.val_dl.dataset.cats);
            x_conts = torch.Tensor(model_data.val_dl.dataset.conts)
            y_eval = torch.Tensor(model_data.val_dl.dataset.y)
            if config_.use_gpu:
                x_conts = x_conts.cuda();       y_eval = y_eval.cuda();
            preds = model(x_cats, x_conts)
            # PyTorch的mseloss并不一致   https://discuss.pytorch.org/t/how-is-the-mseloss-implemented/12972/14
            loss = criterion(preds, y_eval);
            loss_2 = loss.item()
            # squeeze    Returns a tensor with all the dimensions of input of size 1 removed.
            preds = preds.detach().squeeze().cpu().numpy();
            assert nSamp == preds.shape[0]
            if False:
                # print("eval_y={}, preds={}".format(eval_y, preds))
                score = roc_auc_score(eval_y, preds)
                fpr, tpr, thresholds = roc_curve(eval_y, preds)
                loss_1 /= trn_set.y.shape[0];
            dis = np.linalg.norm(eval_y - preds)
            dis = np.sqrt(dis*dis/eval_y.size)
            loss_3 = config_.user_loss(config_,nSamp,preds,eval_y, x_conts.cpu().numpy(),plot_each=epoch>3) if config_.user_loss is not None else 0;
            print( "{:8d} nDevice={} len(Y)={} \tloss=[{:.2f},{:.4f}] user_loss={:.4f} |Y'-Y0|={:.4f} time={:.2f}".
                    format(epoch,nSamp,eval_y.size,loss_1,loss_2,loss_3,dis,time.time() - t0))
            model.loss_curve.append([epoch,loss_3])
            vis_plot(config_,vis,epoch,loss_3,env_title)
        with open(all_predict_path, "ab") as fp:  # Pickling
            pickle.dump([epoch,loss_3], fp)
            pickle.dump(preds, fp)

        save_to_path = "{}.pickle".format(config_.loss_curve_title)
        with open(save_to_path, "wb") as fp:  # Pickling
            pickle.dump(model.loss_curve, fp)
        if epoch%10==0:
            model_path = "F:/Project/MetaLab/checkpoints/BN({})/{}_{}.state".format(config_.use_bn,config_.loss_curve_title,epoch)
            torch.save(model.state_dict(), model_path)
            print("---- epoch={} Save model @@@\"{}\"".format(epoch, model_path))


def weighted_binary_cross_entropy(output, target, weights=None):
    # weights=[0.1,0.9]
    if weights is not None:  # https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def get_embs_dims(data, cats):
    cat_sz = [len(data[c].unique()) for c in cats]
    return [(c, min(50, (c + 1) // 2)) for c in cat_sz]
    # return [(c, 2) for c in cat_sz]


def export_graph(config_,model):
    if True:
        dummy_input = (torch.randn(10, 3, 224, 224)).cuda()
        alex_net = torchvision.models.alexnet(pretrained=True).cuda()
        torch.onnx.export(alex_net, dummy_input, "alexnet.proto", verbose=True)
    dummy_input = torch.randn(32, 256)
    if config_.use_gpu:
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model,None, dummy_input, "NanoNet.onnx", verbose=True)

def EmbedNN_test(config_,trainX, trainY, testX, testY, nPt):
    print("train[{}] head=\n{}\ntest[{}] head=\n{} ".
          format(trainX.shape, trainX[0:5, :], testX.shape, testX[0:5, :]))
    nIn = trainX.shape[1]
    nOut = trainY.shape[1]

    _params = {
        'lr': 0.00001,  # 0.02
        'weight_decay': 1e-4,
        'use_gpu': True,
        'batch_size': 256,
        'train_data_root': '',
        'test_data_root': '',
        'pkl_path': '',
    }
    assert config_ is not None
    config_.parse(_params)
    #if config_.use_bn!="none":
    #    config_.lr *= 5 #BN support much hihger learning rate

    emb_szs, cat_vars = None, None
    print("\n======EmbedNN_test. Train={}, test={}\ntrain_head={}".format(trainX.shape, testX.shape, trainX[0:5, :]))
    with timer("### training ###"):
        # pytorch_env(42)
        # cat_sz = [(c, len(train_df[c].cat.categories) + 1) for c in cat_vars]
        # print("cat_sz[{}]={}".format(len(cat_sz), cat_sz))
        test_ratio = 100/trainX.shape[0]
        #rain_x, eval_x, train_y, eval_y = train_test_split(trainX, trainY, random_state=42, test_size=test_ratio,shuffle=True)
        train_x, eval_x, train_y, eval_y = trainX,testX, trainY,  testY
        data_loader = EmbeddingModelData.from_data_frames('./tmp', train_x, eval_x, train_y, eval_y, cat_vars,
                                                          batch_size=config_.batch_size, test_df=testX)
        trn_set = data_loader.trn_dl.dataset
        nSamp = eval_y.shape[0]
        '''
            放大每层，对于困难模型(85 _lenda(240.0-2000.0）_H(5.0-50.0)_N(10)_xita(85)_model(v0)_25600000_)确实能提高准确率
            增加层[nIn, 128, 64, 64,64,32, 32, 32, 16]差不多
        '''
        #layer_szs = [nIn, 128, 64, 64, 32, 32, 16]
        #layer_szs = [nIn, 128 * 2, 64 * 2, 64 * 2, 32 * 2, 32 * 2, 16 * 2]
        #layer_szs = [nIn, 256, 512, 256,32,32, 16]
        #layer_szs = [nIn, 256, 512, 256,64, 16]     #big2
        #layer_szs = [nIn, 128, 256, 128]
        layer_szs = [nIn, 128, 256, 128,32]
        env_title = "{}_{}_p{}_F{}_x{}_[{}_{}]_[{}]_{}".format(config_.use_bn,config_.model,config_.polar,
            config_.noFeat4CMM, config_.xita, trn_set.__len__(), nSamp,layer_szs,config_.batch_size)
        config_.env_title = env_title
        vis = visdom.Visdom(env=env_title)
        emb_model = Embedding_NN(emb_szs, data_loader.n_conts, 0.5, nOut,
                                 layer_szs, drops=[0,0,0,0], classify=False,use_bn=config_.use_bn,isDrop=True)
        # emb_model = Embedding_NN(emb_szs, data_loader.n_conts, 0.05, 1, [500,250,250],[0.1,0.1,0.1], classify=False,isDrop=False)
        # opt = optim.SGD(emb_model.parameters(), lr=config_.lr, weight_decay=config_.weight_decay)
        # export_graph(config_,emb_model)

        opt = optim.Adam(emb_model.parameters(), lr=config_.lr, weight_decay=config_.weight_decay)
        # crit = F.nll_loss
        # crit = nn.BCELoss()
        # crit = weighted_binary_cross_entropy
        crit = nn.MSELoss()
        #emb_model.load_state_dict(torch.load('Loss__lenda(240.0-2000.0)_H(5.0-50.0)_N(10)_xita(85)_polar(0)_model(v1)_BN(none)__0.state'))
        embedding_train(config_,vis,env_title, emb_model, data_loader, opt, crit, epochs=500, eval_y=eval_y)


if __name__ == "__main__":
    submission_file_name = "cys_20.csv"
    with timer("Full model run"):
        main(num_rows=10000)
