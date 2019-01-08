from jreftran_rt import *
from GraSi3N4 import *
from Inverse import *
from models.Embed_NN import *
import pickle
import seaborn as sns;      sns.set()
from GraSi3N4 import *

def ReadResult(path):
    vPreds = []
    with open(path, "rb") as fp:
        while True:
            try:
                [epoch, loss_] = pickle.load(fp)
                preds = pickle.load(fp)
                vPreds.append(preds)
            except (EOFError, pickle.UnpicklingError):
                break
    return vPreds

if __name__ == '__main__':
    doStatistic = False
    all_errs=[]
    config = GraSi3N4_init(2018, 'v1')
    test='lenda(240.0-2000.0)_H(5.0-50.0)_N(10)_xita(85)_polar(0)_model(v1)'
    test="lenda({:.1f}-{:.1f})_H({:.1f}-{:.1f})_N({})_xita({})_polar({})_model({})".format(
        config.lenda_0, config.lenda_1,config.thick_0, config.thick_1,10,config.xita, config.polar, config.model)
    print("======Predict @{}...".format(test))
    #bn_types=['BN(none)','BN(adaptive)','BN(none)']
    bn_types = ['BN(none)','BN(adaptive)','BN(bn)']
    bn_titles = ['No BN', 'adaptive-BN', 'standard-BN']
    feat = 'Transmission' if config.noFeat4CMM==0 else 'absoptity'
    with open("F:/Project/MetaLab/Result/12_1_2018/{}/INVERSE__{}_.pickle".format(feat,test), "rb") as fp:  # Pickling
        evalX = pickle.load(fp)
        evalY = pickle.load(fp)
    nCase,nPt = evalX.shape[0],evalX.shape[1]
    assert nPt==256
    all_Preds={}
    nEpoch = 100000000
    for bn in bn_types:
        path = 'F:/Project/MetaLab/Result/12_1_2018/{}/{}/PREDICT_Loss__{}_{}_.pickle'.format(feat,bn,test,bn)
        vPreds = ReadResult(path)
        all_Preds[bn] = vPreds
        nEpoch = min(nEpoch,len(vPreds))
    #for epoch in range(nEpoch):
    epoch = min(100,nEpoch)

    style_col = None
    hue_col = None
    off_max=0
    picks=[]

    for case in range(nCase):
        if case % 100==0:
            print("{}...".format(case));
        df,df_thick=pd.DataFrame(),pd.DataFrame()
        curves = []
        err_compare=[]
        thicks_target = evalY[case,:]
        #thicks_target = thicks_target.reshape(1, len(thicks_target))
        curve_target = evalX[case,:]
        df['lenda'] = config.lenda_tic
        df['target'] = curve_target
        df_thick['target'] = Thicks_from_output(config,thicks_target)
        curves.append(pd.DataFrame(curve_target))
        #curve_target = curve_target.reshape(1, len(curve_target))

        for no in range(len(bn_types)):
            vPreds = all_Preds[bn_types[no]][epoch]
            thicks_predict = vPreds[case,:]
            #thicks_predict = thicks_predict.reshape(1, len(thicks_predict))
            err_1,curve_p=Loss_GraSi3N4_spectra(config, 1, thicks_predict,thicks_target, curve_target, plot_each=False)
            curves.append(pd.DataFrame(curve_p))
            df[bn_titles[no]] = curve_p
            df_thick[bn_titles[no]] = Thicks_from_output(config,thicks_predict)
            #df['absorpity'] = curve_p
            #df[hue_col] = bn
            err_compare.append(err_1)
        if doStatistic:
            all_errs.append(err_compare)
            #if case>10: break
            continue
        else:
            off=err_compare[0]-err_compare[1]
            #if err_compare[1]<0.05 and off>0.2 and off<0.3:
            if True:
                picks.append(case)
                off_max = off
                #df = pd.concat(curves)
                if True:
                    writer = pd.ExcelWriter('F:/Project/MetaLab/Result/12_1_2018/{}.xlsx'.format(case))
                    df.to_excel(writer, 'absorptive spectrum')
                    df_thick.to_excel(writer, 'thickness of layers')
                    writer.save()

                #fig1 = plt.figure(figsize=(12, 6))
                plt.clf()
                plt.title("{}: off={:.3f} {}".format(case, off, ''))
                plt.plot(config.lenda_tic, curves[0], '-', label='Target')
                for i in range(3):
                    plt.plot(config.lenda_tic, curves[i+1], '.',
                             label="{}    err={:.2g}".format(bn_titles[i],err_compare[i]))
                plt.legend(loc='best')
                plt.savefig('F:/Project/MetaLab/Result/12_1_2018/{}_(off={:.3g}).png'.format( case,off))
            #plt.show(block=True)
    if doStatistic:
        dfErr = pd.DataFrame.from_records(all_errs, columns=bn_titles)
        print(dfErr.describe())
    sys.exit(0)

