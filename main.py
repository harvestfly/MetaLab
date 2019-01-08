#from visualdl import LogWriter
from jreftran_rt import *
from GraSi3N4 import *
from Inverse import *
from models.Embed_NN import *
from numpy import genfromtxt
from models.gbrt_lgb import *

isTestLiteMORT = False
isInverse = True
def FindPeakValue(config,mX, mY, nPt):
    nCase = (int)(mX.shape[0] / nPt)
    noY = config.noFeat4CMM  # 0,1,2   三条曲线
    n0, n1= 0, 0
    peak,fMax = 0,0
    #iX = np.zeros((nCase, nPt))
    nLayer = 5 if config.fix_graphene_thick else 10
    #iY = np.zeros((nCase, nLayer))
    x_tic = mX[0:nPt, 10]
    for case in range(nCase):
        n1 = n0 + nPt
        #iY[case, :] = mX[n0, 1:10:2] if nLayer == 5 else mX[n0, 0:10]
        #iX[case, :] = mY[n0:n1, noY]
        curve = mY[n0:n1, noY]
        peak = max(curve)
        n0 = n1
        if peak>fMax :
            fMax = peak
            if peak>0.7:
                plt.plot(x_tic, curve)
                plt.show(block=True)
    os._exit(0)
    #np.savez_compressed(pathZ, X=iX, Y=iY)
    return


if __name__ == '__main__':
    for arg in sys.argv:
        if arg == 'mort':   isTestLiteMORT = True

    config = GraSi3N4_init(2018,'v1')
    if config.model=='v0':
        nTest, nTrainCase = 1000, 10000
    else:
        nTest,nTrainCase = 1000, 50000
    #pathTest = GraSi3N4_sample(nTest, 1997, config)
    nLayer = config.nLayer

    sKeyTitle = "_lenda({:.1f}-{:.1f})_H({:.1f}-{:.1f})_N({})_xita({})_polar({})_model({})".format(
        config.lenda_0, config.lenda_1,config.thick_0, config.thick_1,nLayer,config.xita, config.polar, config.model)
    if nLayer==20:
        pathZ = './data/{}_2560000_.npz'.format(sKeyTitle)
        pathTest = './data/{}_25600_.npz'.format(sKeyTitle)
    else:
        pathZ = './data/{}_{}_.npz'.format(sKeyTitle,nTrainCase*256)
        pathTest = './data/{}_{}_.npz'.format(sKeyTitle,nTest*256)
        # pathZ = './data/_L(240.0-2000.0）_H(5.0-50.0)_2560000_.npz'
        # pathTest = './data/_L(240.0-2000.0）_H(5.0-50.0)_25600_.npz'
        # pathZ = './data/_L(240.0-400.0）_H(5.0-100.0)_25600000_.npz'
        # pathTest = './data/_L(240.0-400.0）_H(5.0-100.0)_25600_.npz'
        # pathZ = './data/_L(240.0-2000.0）_H(5.0-20.0)_2560000_.npz'
        # pathTest = './data/_L(240.0-2000.0）_H(5.0-20.0)_25600_.npz'
    #pathZ = None
    if pathZ is None or not(os.path.isfile(pathZ)):
        pathZ = GraSi3N4_sample(nTrainCase, 42,sKeyTitle,config)
        pathTest = GraSi3N4_sample(nTest, 1997,sKeyTitle,config)
        print("GraSi3N4_sample@@@{} nTrain={} nTest={}".format(pathZ,10000,nTest))
        os._exit(0)


    print("\n======Load Train@@@\"{}\"......".format(pathZ))
    loaded = np.load(pathZ)
    mX, mY = loaded['X'], loaded['Y']
    if False:   #some tesing samples
        nTest,testX, testY = user_absorb_test(config,1997,None)
    else:
        print("\n======Load Test@@@\"{}\"......".format(pathTest))
        loaded = np.load(pathTest)
        testX, testY = loaded['X'], loaded['Y']
    nPt = (int)(testX.shape[0] / nTest)
    assert nPt==256
    #FindPeakValue(config,mX, mY, nPt)
    if isInverse:
        trainX, trainY = X_Curve_Y_thicks(config,mX, mY, nPt)
        if False:   # 用来测试小样本的表现       10/6/2018   cys
            trainX, trainY = trainX[0:5000], trainY[0:5000]
        config.loss_curve_title="{}_{}_BN({})_".format(config.loss_curve_title,sKeyTitle,config.use_bn)
        # 用来校验数据是否有重复       10/6/2018   cys
        # GraSi3N4_check_similar_1(config, mY,trainY, nPt)
        evalX, evalY = X_Curve_Y_thicks(config,testX, testY, nPt)
        config.user_loss = Loss_GraSi3N4_spectra
        with open("INVERSE_{}_.pickle".format(sKeyTitle), "wb") as fp:  # Pickling
            pickle.dump(evalX, fp)
            pickle.dump(evalY, fp)
    else:
        trainX, trainY = X_thicks_Y_Curve(mX, mY, nPt)
        evalX, evalY = X_thicks_Y_Curve(testX, testY, nPt)
        config.user_loss = None

    EmbedNN_test( config,trainX, trainY, evalX, evalY, nPt )
    # lgb_test(iTX, iTY, iEX, iEY)
    # nverseDesign(iTX, iTY, iEX, iEY)
    exit()

    title = Thick2Title(testX[0:1, 0:nLayer].squeeze())
    print(
        "train[{}] head=\n{}\ntest[{}] nPt={} head=\n{} ".format(mX.shape, mX[0:5, :], testX.shape, nPt, testX[0:5, :]))
    plt.rcParams["figure.figsize"] = [16, 9]
    if isTestLiteMORT:
        LiteMORT_test(mX, mY[:, 1], testX, testY[:, 1], 0x0)
    else:
        y_pred = lgb_test(mX, mY[:, 1], testX, testY[:, 1])

    xlabel, ylabel = 'lenda (nm)', ['Reflectivity', 'Transmissivity', 'Absorptivity']
    for no in range(nTest):
        plt.gcf().clear()
        y0 = testY[no * nPt:(no + 1) * nPt, 1]
        y2 = y_pred[no * nPt:(no + 1) * nPt]
        lenda = testX[no * nPt:(no + 1) * nPt, nLayer]
        err_2 = np.linalg.norm(y2 - y0) / np.linalg.norm(y0)
        title = '\"Transmissivity\" err={:.4f} \n{}'.format(err_2, Thick2Title(
            testX[no * nPt:no * nPt + 1, 0:nLayer].squeeze()))
        #
        plt.plot(lenda, y0, '-', label='True')
        plt.plot(lenda, y2, '.', label='Predict')
        plt.xlabel(xlabel), plt.title(title, fontsize=20), plt.legend(loc='best', fontsize=18)
        plt.show(block=True)
