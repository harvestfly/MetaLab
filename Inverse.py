from GraSi3N4 import *


def X_Curve_Y_thicks(config, mX, mY, nPt, pick_thick=-1):
    pathZ = "./data/_INVERSE_{}_pick={}.npz".format(mX.shape, pick_thick)
    # if os.path.isfile(pathZ):
    if False:
        loaded = np.load(pathZ)
        iX, iY = loaded['X'], loaded['Y']
        return iX, iY

    nCase = (int)(mX.shape[0] / nPt)
    noY = config.noFeat4CMM  # 0,1,2   三条曲线
    n0, n1, pos = 0, 0, pick_thick
    iX = np.zeros((nCase, nPt))
    nLayer = 5 if config.fix_graphene_thick else 10
    iY = np.zeros(nCase) if pick_thick >= 0 else np.zeros((nCase, nLayer))
    x_tic = mX[0:nPt, 10]
    for case in range(nCase):
        n1 = n0 + nPt
        if pos >= 0:
            thick = mX[n0, pos]
            for n in range(n0, n1):
                assert (thick == mX[n0, pos])
            iY[case] = thick
        else:
            iY[case, :] = mX[n0, 1:10:2] if nLayer == 5 else mX[n0, 0:10]
        curve0, curve1 = mY[n0:n1, 0], mY[n0:n1, 1]
        #iX[case, :] = np.concatenate([curve0, curve1])  # mY[n0:n1,noY]
        iX[case, :] = mY[n0:n1,noY]

        n0 = n1
        if False:
            plt.plot(x_tic, iX[case, :])
            plt.show(block=True)

    if config.normal_Y == 1:
        s = (config.thick_1) / 2
        iY = (iY) / s - 1

    np.savez_compressed(pathZ, X=iX, Y=iY)
    return iX, iY


def X_thicks_Y_Curve(mX, mY, nPt, pick_thick=-1):
    noFeat = 1  # 0,1,2   三条曲线
    pathZ = "./data/_X(thicks)_Y(Curve)_{}_feat={}_pick={}.npz".format(mX.shape, noFeat, pick_thick)
    if os.path.isfile(pathZ):
        # if True:
        loaded = np.load(pathZ)
        iX, iY = loaded['X'], loaded['Y']
        return iX, iY

    nCase = (int)(mX.shape[0] / nPt)
    n0, n1, pos = 0, 0, pick_thick
    iY = np.zeros((nCase, nPt))
    iX = np.zeros(nCase) if pick_thick >= 0 else np.zeros((nCase, 10))
    x_tic = mX[0:nPt, 10]
    for case in range(nCase):
        n1 = n0 + nPt
        if pos >= 0:
            thick = mX[n0, pos]
            for n in range(n0, n1):
                assert (thick == mX[n0, pos])
            iX[case] = thick
        else:
            iX[case, :] = mX[n0, 0:10]
        iY[case, :] = mY[n0:n1, noFeat]

        n0 = n1
        if False:
            plt.plot(x_tic, iY[case, :])
            plt.show(block=True)

    np.savez_compressed(pathZ, X=iX, Y=iY)
    return iX, iY


def InverseDesign(trainX, trainY, evalX, eavlY):
    nTest, nLayer = 100, 10
    nPt = (int)(evalX.shape[0] / nTest)
    title = Thick2Title(evalX[0:1, 0:nLayer].squeeze())
    print(
        "train[{}] head=\n{}\ntest[{}] nPt={} head=\n{} ".format(mX.shape, mX[0:5, :], testX.shape, nPt, testX[0:5, :]))
    return
