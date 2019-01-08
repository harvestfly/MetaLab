'''
    1) log all data to csv and use jupyter (ipython) notebooks to visualize. This is something I learned from some power users of torch/pytorch.
    2) https://github.com/lanpa/tensorboard-pytorch
'''

from some_libs import *
import visdom
import time
import numpy as np
import torch
import pickle
import seaborn as sns;      sns.set()
from matplotlib.colors import LogNorm

# https://seaborn.pydata.org/generated/seaborn.lineplot.html
def compare_loss_curve(files,hue_col,style_col=None):
    no,nPt = 0,sys.maxsize
    datas_0,datas_1 = [],[]
    # style_col = 'Incident Angle'
    for path,bn,xita in files:
        with open(path, "rb") as fp:  # Pickling
            #legend = 'adaptive BN' if no==0 else 'standard BN'
            loss_curve = pickle.load(fp)
        nPt = min(nPt, len(loss_curve))
        assert nPt>10
        df = pd.DataFrame.from_dict(loss_curve)
        #print(df.info())
        df = df.rename(columns={df.columns[0]: 'epoch', df.columns[1]: 'loss'})
        df[hue_col]=bn
        if xita is not None:
            df[style_col] = xita
        datas_0.append(df)
        no = no+1
    for df in datas_0:
        df = df[0:nPt]
        datas_1.append(df)
    df = pd.concat(datas_1)
    if True:
        writer = pd.ExcelWriter('figure_4.xlsx')
        df.to_excel(writer, 'sheet_1')
        writer.save()

# http://hookedondata.org/Better-Plotting-in-Python-with-Seaborn/
    #sns.set_style("whitegrid")  #five options for the background of your plot; the default one is darkgrid
    #sns.scatterplot(x="epoch",y="loss",data=df)
    if style_col is None:
        ax = sns.lineplot(x="epoch", y="loss", data=df, hue_norm=LogNorm(), hue=hue_col, lw=1) \
            .set_title('Relative Spectrogram Loss')
    else:
        ax = sns.lineplot(x="epoch", y="loss", data = df, hue_norm=LogNorm(),hue=hue_col,style=style_col,lw=1)\
            .set_title('Relative Spectrogram Loss')

    print("OK")

def vis_plot(config_, vis, epoch, loss_3,title):
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss_3]), win='loss',
             opts=dict(
                 legend=[title], #[config_.use_bn],
                 fillarea=False,
                 showlegend=True,
                 width=1600,
                 height=800,
                 xlabel='Epoch',
                 ylabel='L2 distance',
                 # ytype='log',
                 title='L2 distance between target & predicted spectrogram',
                 # marginleft=30,
                 # marginright=30,
                 # marginbottom=80,
                 # margintop=30,
             ),
             update='append' if epoch > 0 else None)


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)
        print(self.log_text)

    def __getattr__(self, name):
        return getattr(self.vis, name)


if __name__ == '__main__':
    folder = 'F:/Project/MetaLab/Result/10_28_2018/'
    files = [[folder+'loss_curve_adaptive.pickle','N=10000',None],
             [folder+'Relative Spectrogram Loss_adaptive_[5000].pickle','N=5000',None],
             [folder+'Relative Spectrogram Loss_adaptive_[1000].pickle','N=1000',None],
             #'loss_curve_bn.pickle'
             ]
    legend_title = 'set size'
    #compare_loss_curve(files, legend_title)

    files = [['F:/Project/MetaLab/loss_curve_adaptive.pickle', 'adaptive BN'],
             ['F:/Project/MetaLab/loss_curve_bn.pickle', 'standard BN'],
             ]

    files = [[folder+'Loss_H(5.0-50.0)_N(10)_xita(60)_BN(adaptive)_.pickle', 'adaptive-BN', 'angle=60'],
             [folder+'Loss_H(5.0-50.0)_N(10)_xita(60)_BN(bn)_.pickle', 'standard-BN', 'angle=60'],
             [folder+'Loss_H(5.0-50.0)_N(10)_xita(30)_BN(adaptive)_.pickle', 'adaptive-BN', 'angle=30'],
             [folder+'Loss_H(5.0-50.0)_N(10)_xita(30)_BN(bn)_.pickle', 'standard-BN', 'angle=30'],
             [folder+'loss_curve_adaptive.pickle', 'adaptive-BN', 'angle=0'],
             [folder+'loss_curve_bn.pickle', 'standard-BN', 'angle=0'],
             [folder + 'Loss__lenda(240.0-2000.0)_H(5.0-50.0)_N(10)_xita(0)_polar(0)_model(v0)_BN(none)_.pickle', 'No BN', 'angle=0'],
             [folder + 'Loss__lenda(240.0-2000.0)_H(5.0-50.0)_N(10)_xita(30)_polar(0)_model(v0)_BN(none)_.pickle', 'No BN', 'angle=30'],
             [folder + 'Loss__lenda(240.0-2000.0)_H(5.0-50.0)_N(10)_xita(60)_polar(0)_model(v0)_BN(none)_.pickle', 'No BN', 'angle=60'],
             ]
    legend_title = 'Batch Normalization'
    compare_loss_curve(files,legend_title,'Incident Angle')


