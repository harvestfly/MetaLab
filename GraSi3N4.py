from some_libs import *
import warnings
from jreftran_rt import *
from config import DefaultConfig
from Polynomial import *
from spec_gram import *
from numpy.linalg import norm

'''
    入射角 85' 60'             85'确实有问题，简单的MLP无法收敛
    极化      TE,TM
    厚度  [50,50,50,50,50]    [10,10,10,10,10] [10,10,50,10,10]`[49,9.5,7,50,20]
    折射率 上入射 1.46    NSi  2.0
    
'''
def plt_curve_structure(config,curves,title):
    # plt.gcf().clear()
    fig1 = plt.figure(figsize=(12, 6))
    # ax1 = fig1.add_subplot(111, aspect='equal')
    # ax1.add_patch( patches.Rectangle((0, 0), 50, 0.1, linewidth=1, edgecolor='r', facecolor='none'))
    # https://stackoverflow.com/questions/21445005/drawing-rectangle-with-border-only-in-matplotlib
    currentAxis = plt.gca()
    for curve,label,thicks in curves:
        plt.plot(config.lenda_tic, curve,  label=label) # '.',
    #plt.plot(config.lenda_tic, curve_0, '-', label='Target')
    ymin, ymax = currentAxis.get_ylim()
    xmin, xmax = currentAxis.get_xlim()
    left, width, delta = (xmin + xmax) / 2, 300, (ymax - ymin) * 0.4
    bottom = ymax - delta * 1.1;       curY=bottom
    scale = delta / 150         #   scale = delta/thick_sum
    no = 0
    for curve, label, thicks in curves:
        curY = bottom
        if thicks is not None:
            for thick in thicks:
                color='r' if no%2==0 else 'y'
                lw = 1 if no%2==0 else 0.1
                thick = 2 if no % 2 == 0 else thick
                currentAxis.add_patch(patches.Rectangle((left, curY), width, thick*scale, alpha=1,facecolor=color) )
                curY+=thick*scale
                no=no+1
            left += 400

    plt.title(title)
    plt.legend(loc='best')
    plt.show(block=True)

def findnearest(array, value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 125893.0:
        return array[idx]
    else:
        return array[idx]
    idx1 = np.searchsorted(array, value, side='right')
    if idx1 < 2e10:
        return array[idx1]
    else:
        return array[idx1 - 1]


class N_Dict(object):
    map2 = {}

    def __init__(self,config):
        self.dicts = {}
        self.config = config
        return

    def InitMap2(self, maters, lendas):
        for mater in maters:
            for lenda in lendas:
                self.map2[mater, lenda] = self.Get(mater, lenda, isInterPolate=True)

    def Load(self, material, path, scale=1):
        df = pd.read_csv(path, delimiter="\t", header=None, names=['lenda', 're', 'im'], na_values=0).fillna(0)
        if scale is not 1:
            df['lenda'] = df['lenda'] * scale
            df['lenda'] = df['lenda'].astype(int)
        self.dicts[material] = df
        rows, columns = df.shape
        # if columns==3:
        print("{}@@@{} shape={}\n{}".format(material, path, df.shape, df.head()))

    def Get(self, material, lenda, isInterPolate=False):
        # lenda = 1547
        n = 1 + 0j
        if material == "air":
            return n
        if material == "Si3N4":
            if self.config.model=='v1':
                return 2. + 0j
            else:
                return 2.46 + 0j
            #return 2.0
        assert self.dicts.get(material) is not None
        df = self.dicts[material]
        assert df is not None
        pos = df[df['lenda'] == lenda].index.tolist()
        # assert len(pos)>=1
        if len(pos) == 0:
            if isInterPolate:
                A = df.as_matrix(columns=['lenda'])  # df.as_matrix(columns=['lenda'])
                idx = (np.abs(A - lenda)).argmin()
                if idx == 0:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                else:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx - 1], df['re'].loc[idx - 1], df['im'].loc[idx - 1]
                lenda_2, re_2, im_2 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                re = np.interp(lenda, [lenda_1, lenda_2], [re_1, re_2])
                im = np.interp(lenda, [lenda_1, lenda_2], [im_1, im_2])
            else:
                return None
        elif len(pos) > 1:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        else:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        n = re + im * 1j
        return n


class Thin_Layer(object):
    def __init__(self, mater, thick_=np.nan):
        self.material = mater
        self.thick = thick_
        self.n_ = 1 + 0j


class Thin_Film_Filters(object):
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        # for k, v in self.__class__.__dict__.items():
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print("\t{}:    {}".format(k, getattr(self, k)))

    def __init__(self, layers):
        self.layers = layers


def Thick2Title(thicks, isAir=True):
    nLyaer = len(thicks)
    title = "{} layers, thick=[".format(nLyaer)
    # self.layers.append(Thin_Layer('air'))
    no = 0
    for thick in thicks:
        mater = 'Graphene' if no % 2 == 0 else 'Si3N4'
        title += "{:.2f},".format(thick)
        no = no + 1
    # self.layers.append(Thin_Layer('air'))
    title += "](nm)"
    return title


class GraSi3N4(Thin_Film_Filters):
    def InitThick(self, thicks=None, h_G=0.35):
        # h_G,hSi=0.35,5        # 计算石墨烯折射率的时候取的厚度是0.35nm
        nLayer = self.config.nLayer
        hSi_1, hSi_2 = self.config.thick_0, self.config.thick_1
        if thicks is None:
            thicks = []
            for no in range(nLayer):
                if no % 2 == 0:
                    thicks.append(h_G)
                else:
                    hSi = random.uniform(hSi_1, hSi_2)
                    thicks.append(hSi)
        self.thicks = thicks
        return self.thicks

    def __init__(self, n_dict, config, thicks=None):
        assert (config is not None)
        self.config = config
        self.n_dict = n_dict
        self.xita = config.xita #angle of incidence
        self.InitThick(thicks)
        self.InitLayers()
        # self.lenda = lenda

        # layer.n_ =
        return

    def InitLayers(self, isAir=True):
        thicks = self.thicks
        nLyaer = len(thicks)
        title = "{} layers, thick=[".format(nLyaer)
        # self.layers.append(Thin_Layer('air'))
        self.layers = []
        no = 0
        for thick in thicks:
            if thick is np.nan:
                mater = 'air'
            else:
                mater = 'Graphene' if no % 2 == 0 else 'Si3N4'
            self.layers.append(Thin_Layer(mater, thick))
            title += "{:.2f},".format(thick)
            no = no + 1
        # self.layers.append(Thin_Layer('air'))
        title += "](nm)"
        self.title = title

    # 基底材料
    def OnSubstrate(self, d0, n0):
        if self.config.model == 'v1':
            d = np.concatenate([np.array([np.nan]), d0, np.array([np.nan])])
            n = np.concatenate([np.array([1.46 + 0j]), n0, np.array([1.0 + 0j])])
        elif self.config.model == 'v0':
            d_air, n_air = np.array([np.nan]), np.array([1 + 0j])
            d = np.concatenate([d_air, d0, d_air])
            n = np.concatenate([n_air, n0, n_air])
        else:
            print( "OnSubstrate:: !!!config.model is {}!!!".format(self.config.model) )
            sys.exit(-66)
        return d, n

    def nMostPt(self):
        # nMost = int((self.config.lenda_1-self.config.lenda_0)/self.config.lenda_step+1)
        nMost = len(self.config.lenda_tic)
        return nMost

    def Chebyshev(self):
        nRow, nCol = self.dataX.shape
        chebX = self.dataX[0, 0:nCol - 1]
        lenda = self.dataX[:, nCol - 1]
        chebY = cheb_fitcurve(lenda, self.dataY[:, 0], 64)
        return

    def CMM(self):
        t0 = time.time()
        nMostRow, nLayer = self.nMostPt(), len(self.layers)
        dataX = np.zeros((nMostRow, nLayer + 1))
        dataY = np.zeros((nMostRow, 3))
        row = 0
        map2 = self.n_dict.map2
        polar = self.config.polar
        for lenda in self.config.lenda_tic:
            i = 0
            # dict_n = {}
            d, n = np.zeros(nLayer), np.zeros(nLayer, dtype=complex)

            for layer in self.layers:
                d[i] = layer.thick
                # if layer.material not in dict_n:
                #    dict_n[layer.material] = self.n_dict.Get(layer.material, lenda, isInterPolate=True)
                # n[i] = dict_n[layer.material]   #layer.n_
                if layer.material=='air':
                    n[i] = 1
                else:
                    n[i] = map2[layer.material, lenda]
                dataX[row, i] = d[i]
                i = i + 1
            dataX[row, nLayer] = lenda
            d, n = self.OnSubstrate(d, n)
            #print("d={}\nn={}".format(d, n))
            r, t, R, T, A = jreftran_rt(lenda, d, n, self.xita, polar)
            if False:
                sum = R + T + A
                if abs(sum - 1) > 1.0e-7:
                    print("sum={} R={} T={} A={}".format(sum, R, T, A))
                assert abs(sum - 1) < 1.0e-7
            dataY[row, 0] = R
            dataY[row, 1] = T
            dataY[row, 2] = A

            row = row + 1
            if row >= nMostRow:            break

        # Sepectrogram(dataY[:, 0])
        # make_melgram(dataY[:, 0],256)
        self.dataX = dataX[0:row, :];
        self.dataY = dataY[0:row, :]

        # print("======N={} time={:.3f}".format(row, time.time() - t0))
        return

    def plot_scatter(self,user_title=None):
        xlabel, ylabel = 'lenda (nm)', ['Reflectivity', 'Transmissivity', 'Absorptivity']
        nFigure = len(ylabel)
        plt.gcf().clear()
        # plt.figure(figsize=( 7.195,3.841 ), dpi=200)
        plt.rcParams["figure.figsize"] = [16, 9]
        if False:
            line = plt.figure()
            np.random.seed(5)
            x = np.arange(1, 101)
            y = 20 + 3 * x + np.random.normal(0, 60, 100)
            assert arrX.shape[0] == arrY.shape[0]
        nLayer = len(self.layers)
        arrX = self.dataX[:, nLayer]
        for no in range(nFigure):
            # plt.figure(no)
            plt.plot(arrX, self.dataY[:, no], label=ylabel[no])
            plt.xlabel(xlabel)
            # plt.ylabel(ylabel[no])
            # break
        plt.legend(loc='best')
        plt.title(self.title)
        if user_title is not None:
            plt.title(user_title)
        plt.show(block=True)

#网络输出层的结果需要适当变换
def Thicks_from_output(config,thicks_net_output):
    nLayer = thicks_net_output.shape[0]
    if config.fix_graphene_thick:       # graphene_thick固定为0.35
        nLayer *= 2
        thicks_1 = np.zeros(nLayer)
        thicks_1.fill(0.35)
        thicks_1[1:nLayer:2] = thicks_net_output[:]

    if config.normal_Y == 1:
        s = (config.thick_1) / 2
        thicks = (thicks_1 + 1) * s
    else:
        thicks = thicks_1
    return thicks


def Loss_GraSi3N4_spectra(config, nCase, thicks_predict,thicks_target, curves, plot_each=False):
    if nCase==1:    #真麻烦
        thicks_predict = thicks_predict.reshape(1, len(thicks_predict))
        thicks_target = thicks_target.reshape(1, len(thicks_target))
        curves = curves.reshape(1, len(curves))
    else:
        assert nCase == thicks_predict.shape[0]
    nLayer = thicks_predict.shape[1]
    noFeat = config.noFeat4CMM
    loss = 0

    if config.fix_graphene_thick:       # graphene_thick固定为0.35
        nLayer *= 2
        thicks_1 = np.zeros((nCase, nLayer))
        thicks_1.fill(0.35)
    for case in range(nCase):
        thicks_1[case,1:nLayer:2] = thicks_predict[case,:]

    if config.fix_graphene_thick:
        thicks_0 = thicks_1

    if config.normal_Y == 1:
        s = (config.thick_1) / 2
        thicks = (thicks_0 + 1) * s
    else:
        thicks = thicks_0

    for case in range(nCase):
        p0 = case * nLayer;
        p1 = p0 + nLayer
        thick_sum = 0
        for thick in thicks[case, :]:   thick_sum+=thick
        filter = GraSi3N4(config.n_dict, config, thicks[case, :])
        filter.CMM()
        #filter.plot_scatter()
        curve_1, curve_0 = filter.dataY[:, noFeat], curves[case, 0:]
        assert curve_1.shape == curve_0.shape
        off = np.linalg.norm(curve_1 - curve_0) / np.linalg.norm(curve_0)
        plt_title = filter.title
        if config.model=="v1":
            plt_title = "predict={}, target={}".format(thicks_predict[case,:],thicks_target[case,:])
        # if off>0.2:
        #if off < 0.01:
        if False:
            fig1 = plt.figure(figsize=(12, 6))
            # ax1 = fig1.add_subplot(111, aspect='equal')
            # ax1.add_patch( patches.Rectangle((0, 0), 50, 0.1, linewidth=1, edgecolor='r', facecolor='none'))
            # https://stackoverflow.com/questions/21445005/drawing-rectangle-with-border-only-in-matplotlib
            currentAxis = plt.gca()
            plt.plot(config.lenda_tic, curve_1, '.', label='Design')
            plt.plot(config.lenda_tic, curve_0, '-', label='Target')
            if False:   #draw the mulit-layer structure
                ymin, ymax = currentAxis.get_ylim()
                xmin, xmax = currentAxis.get_xlim()
                left, width, delta = (xmin + xmax) / 2, 500, (ymax - ymin) * 0.4
                bottom = ymax - delta * 1.2;       curY=bottom
                scale = delta / 200         #   scale = delta/thick_sum
                no = 0
                for thick in thicks[case, :]:
                    color='r' if no%2==0 else 'y'
                    lw = 1 if no%2==0 else 0.1
                    thick = 2 if no % 2 == 0 else thick
                    currentAxis.add_patch(patches.Rectangle((left, curY), width, thick*scale, alpha=1,facecolor=color) )
                    curY+=thick*scale
                    no=no+1
            plt.title("{}: off={:.3f} {}".format(case, off,plt_title ))
            plt.legend(loc='best')
            plt.savefig('F:/Project/MetaLab/Result/{}/{:.3g}@{}_.png'.format("11_15",off,config.env_title))
            #plt.show(block=True)

        loss += off * off
    loss = np.sqrt(loss / nCase)
    if nCase==1:
        return loss,curve_1
    return loss


def GraSi3N4_init(r_seed,model, config=None):
    np.set_printoptions(linewidth=np.inf)
    random.seed(r_seed - 1)
    np.random.seed(r_seed)
    if config is None:
        config = DefaultConfig(42)
    else:
        config = config
    if config.model!=model:
        print("\n!!!GraSi3N4_init model CHANGE!!! {}=>{}\n".format(config.model,model))
        config.model=model

    t0 = time.time()
    n_dict = N_Dict(config)
    n_dict.Load("Si3N4", "./data/Si3N4_310nm-14280nm.txt", scale=1000);
    n_dict.Load("Graphene", "./data/Graphene_240nm-30000nm.txt");
    n_dict.InitMap2(["Si3N4", "Graphene"], config.lenda_tic)
    config.n_dict = n_dict
    return config


def GraSi3N4_sample(nCase, r_seed, sKeyTitle,config):
    print("========GraSi3N4_sample nCase={} config=......\n{}\n".format(nCase,config.__dict__))
    np.set_printoptions(linewidth=np.inf)
    random.seed(r_seed - 1)
    np.random.seed(r_seed)
    '''
    if config is None:
        config = DefaultConfig(42)
    else:
        config = config
    '''

    t0 = time.time()
    n_dict = N_Dict(config)
    n_dict.Load("Si3N4", "./data/Si3N4_310nm-14280nm.txt", scale=1000);
    n_dict.Load("Graphene", "./data/Graphene_240nm-30000nm.txt");
    n_dict.InitMap2(["Si3N4", "Graphene"], config.lenda_tic)
    arrX, arrY = [], []
    t0 = time.time()
    # 进一步并行     https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
    for case in range(nCase):
        # t1 = time.time()
        filter = GraSi3N4(n_dict, config)
        filter.CMM()
        # filter.Chebyshev()
        #filter.plot_scatter()
        arrX.append(filter.dataX), arrY.append(filter.dataY)
        # gc.collect()
        if case % 100 == 0:
            print("\rno={} time={:.3f} arrX={}\t\t".format(case, time.time() - t0, filter.dataX.shape), end="")

    # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    mX = np.vstack(arrX)
    mY = np.vstack(arrY)
    nRow = mY.shape[0]
    print("Y[{}] head=\n{} ".format(mY.shape, mY[0:5, :]))
    print("X[{}] head={} ".format(mX.shape, mX[0:5, :]))
    l0, l1, h0, h1 = config.lenda_0, config.lenda_1, config.thick_0, config.thick_1
    out_path = "./data/{}".format(sKeyTitle)

    # out_path="./data/___"
    if (nRow < 10000):
        np.savetxt("{}_X{}_.csv".format(out_path, mX.shape), mX, delimiter="\t", fmt='%.8f')
        np.savetxt("{}_Y{}_.csv".format(out_path, mY.shape), mY, delimiter="\t", fmt='%.8f')
    pathZ = "{}_{}_.npz".format(out_path, nRow)
    np.savez_compressed(pathZ, X=mX, Y=mY)
    gc.collect()
    return pathZ

#https://stackoverflow.com/questions/36267936/normalizing-rows-of-a-matrix-python?noredirect=1&lq=1
def matrix_normal(X):
    nRow = X.shape[0]
    l2norm = norm(X, axis=1, ord=2)
    X = X / l2norm.reshape(nRow, 1)
    l2 = norm(X, axis=1, ord=2)
    for nrm in l2:
        assert(abs(nrm-1.0)<1.0e-7)
    return X

def GraSi3N4_check_similar_0(config, mX,mY,nPt):
    mX_0 = np.copy(mX)
    nCase = (int)(mX.shape[0] / nPt)
    mA = mX[:,0];       mB = mX[:,1]
    mA = matrix_normal( np.reshape(mA,(nCase,nPt)) )
    # mB = matrix_normal( np.reshape(mB, (nCase, nPt)) )
    mAA = np.matmul(mA, mA.T)
    nSame,thrsh=0,0.001
    for case in range(nCase):
        struc_0 = mY[case,:]
        curve_0 = mX_0[case*nPt:(case+1)*nPt,0]
        for next in range(nCase):
            off = abs(mAA[case,next]-1)
            curve_1 = mX_0[next * nPt:(next + 1) * nPt, 0]
            if case==next:
                assert off<1.0e-7
            elif off<thrsh:
                struc_1 = mY[next, :]
                delata = struc_1-struc_0
                nSame+=1
    return

def GraSi3N4_check_similar_1(config, mX,mY,nPt):
    mX_0 = np.copy(mX)
    nCase = (int)(mX.shape[0] / nPt)
    mA = mX[:,0];       mB = mX[:,1]
    mA = np.reshape(mA, (nCase, nPt))
    mB = np.reshape(mB, (nCase, nPt))
    nSame,thrsh=0,0.01
    t0 = time.time()
    for case in range(nCase):
        nSame, thrsh,list,figs = 0, 0.01,[],[]
        struc_0 = mY[case,:]
        A_0 = mA[case,:]
        len_0 = np.linalg.norm(A_0)
        for next in range(nCase):
            off_A = np.linalg.norm(A_0-mA[next,:])/len_0
            struc_1 = mY[next, :]
            if case==next:
                assert off_A<1.0e-7
            elif off_A<thrsh:
            #elif abs(sum(struc_0)-sum(struc_1)) < thrsh*sum(struc_0) and off_A>0.05:
                figs = []

                off_struc = struc_1-struc_0
                delta = np.linalg.norm(struc_1-struc_0)/np.linalg.norm(struc_0)
                B_0 = mB[case,:]
                off_B = np.linalg.norm(B_0 - mB[next,:]) / np.linalg.norm(B_0)
                list.append(delta)
                nSame+=1
                figs.append([A_0, "A_0", struc_0])
                figs.append([mA[next,:], "A_1", None])
                figs.append([B_0, "B_0", struc_1])
                figs.append([mB[next,:], "B_1", None])
                title = "[{}]-[{}] |A|={:.4f} |B|={:.4f} delta={:.2f} H_A={:.2f},H_B={:.2f}".format(case,next,off_A,off_B,delta,sum(struc_0),sum(struc_1))
                plt_curve_structure(config, figs, title)
        if nSame>0 or case%100==0:
            print("{}:\tsame={} time={} [{}]".format(case,nSame,time.time()-t0,list))


    return

'''
    v0.1    cys
        11/14/2018
'''
def High_Aborb_test(config):
    #thicks =[.35, 49, 0.35, 9.5, 0.35, 7, 0.35, 50, 0.35, 20]
    filter = GraSi3N4(config.n_dict, config, None)
    thicks = filter.thicks
    ticks = config.lenda_tic
    if False:   #some testing
        config.xita = 85
        #thicks = [0.35, 50, 0.35, 50, 0.35, 50, 0.35, 50, 0.35, 50]
        filter = GraSi3N4(config.n_dict, config, thicks)
        filter.CMM()
        title = "Incident Angle={}".format(config.xita)
        filter.plot_scatter(user_title=title)
        sys.exit(-101)

    xitas = list(range(90))
    M_absorb = np.zeros((len(xitas),len(ticks)))
    no=len(xitas)
    thick_info=''
    for xita in xitas:
        config.xita = xita
        filter = GraSi3N4(config.n_dict, config, thicks)
        filter.CMM()
        thick_info = filter.title
        no=no-1
        M_absorb[no,:]=filter.dataY[:, 2]
    assert no==0
    #plt.imshow(absorb)
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('Absorbance\n'+thick_info)
    cmap = 'coolwarm'     #"plasma"  #https://matplotlib.org/examples/color/colormaps_reference.html
    if True:
        ticks = np.linspace(0, 1, 10)
        ylabels = [int(i) for i in np.linspace(0, 90, 10) ]
        xlabels = [int(i) for i in np.linspace(240, 2000, 10)]
        ax = sns.heatmap(M_absorb,square =True,cmap=cmap, yticklabels=ylabels[::-1], xticklabels=xlabels)
        y_limit = ax.get_ylim();        x_limit = ax.get_xlim()
        ax.set_yticks(ticks * y_limit[0])
        ax.set_xticks(ticks * x_limit[1])
        #ax.set_xticks(np.arange(1, 256, 1))
    else:
        plt.imshow(M_absorb)
        ax.set_aspect(256/90.0)
        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
    plt.show()
    #sys.exit(-100)

#用户设定一组宽度，生成过程参见GraSi3N4_sample
def user_absorb_test(config,r_seed,thicks):
    print("========user_absorb_test config=......\n{}\n".format(config.__dict__))
    np.set_printoptions(linewidth=np.inf)
    random.seed(r_seed - 1)
    np.random.seed(r_seed)
    if thicks==None:
        thicks=[
            [0.35,50,0.35,50,0.35,50,0.35,50,0.35,50],
            [0.35,10,0.35,10,0.35,10,0.35,10,0.35,10],
            [0.35,10,0.35,10,0.35,50,0.35,10,0.35,10],
            [0.35,49,0.35,9.5,0.35,7,0.35,50,0.35,20]
        ]

    t0 = time.time()
    n_dict = N_Dict(config)
    n_dict.Load("Si3N4", "./data/Si3N4_310nm-14280nm.txt", scale=1000);
    n_dict.Load("Graphene", "./data/Graphene_240nm-30000nm.txt");
    n_dict.InitMap2(["Si3N4", "Graphene"], config.lenda_tic)
    arrX, arrY = [], []
    t0 = time.time()
    # 进一步并行     https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
    for layers_thick in thicks:
        assert len(layers_thick)==10
        filter = GraSi3N4(n_dict, config,layers_thick)
        filter.CMM()
        # filter.Chebyshev()
        filter.plot_scatter()
        arrX.append(filter.dataX), arrY.append(filter.dataY)

    # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    mX = np.vstack(arrX)
    mY = np.vstack(arrY)
    nRow = mY.shape[0]
    nHead=min(5,len(thicks))
    print("Y[{}] head=\n{} ".format(mY.shape, mY[0:nHead, :]))
    print("X[{}] head={} ".format(mX.shape, mX[0:nHead, :]))
    print("========user_absorb_test config=......OK\n\n\n".format( ))
    return len(thicks),mX,mY

if __name__ == '__main__':
    if True:
        config = GraSi3N4_init(2018, 'v1')
        config.polar = 1
        for cas in range(0,10):
            High_Aborb_test(config)
    else:
        config.xita=60
        thicks = [0.35,15.33,0.35,34.31,0.35,44.56,0.35,13.58,0.35,26.85]
        filter = GraSi3N4(config.n_dict, config, thicks)
        filter.CMM()
        title = "Incident Angle={}".format(config.xita)
        filter.plot_scatter(user_title=title)