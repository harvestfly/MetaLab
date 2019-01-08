from jreftran_rt import *
from GraSi3N4 import *
from Inverse import *
from models.Embed_NN import *
import pickle
import seaborn as sns;      sns.set()
from GraSi3N4 import *

excel_files=[
    'F:/Project/MetaLab/Result/12_1_2018/Transmission/compare_xita=0/817.xlsx',
    'F:/Project/MetaLab/Result/12_1_2018/Transmission/compare_xita=30/67.xlsx',
    'F:/Project/MetaLab/Result/12_1_2018/Transmission/compare_xita=60/544.xlsx',
    'F:/Project/MetaLab/Result/12_1_2018/absoptity/compare_xita=0/364.xlsx',
    'F:/Project/MetaLab/Result/12_1_2018/absoptity/compare_xita=30/251.xlsx',
    'F:/Project/MetaLab/Result/12_1_2018/absoptity/compare_xita=60/559.xlsx',
]

def excel2sns(file):
    df = pd.read_excel(io=file, sheet_name='absorptive spectrum')
    ax = sns.lineplot(x="lenda", y="target", data=df, color="r",hue_norm=LogNorm(), lw=1) \
        .set_title('Absorbance')
    cmap = sns.cubehelix_palette(as_cmap=True)

    ax = sns.scatterplot(x="lenda", y="adaptive-BN", data=df)
    plt.show()

if __name__ == '__main__':
    if False:
        excel2sns('F:/Project/MetaLab/Result/12_1_2018/some structures/39.xlsx')
        excel2sns('F:/Project/MetaLab/Result/12_1_2018/some structures/230.xlsx')
        os._exit(-1)

    compares=['No BN','adaptive-BN','standard-BN']
    for file in excel_files:
        df = pd.read_excel(io=file, sheet_name='absorptive spectrum')
        #print(df.head(5))  # print first 5 rows of the dataframe
        target = df['target']
        for col in df.columns:
            if col not in compares:
                continue;
            predict = df[col]
            off = np.linalg.norm(predict - target) / np.linalg.norm(target)
            print("{:16s}\t={:.3g}\t@{}".format(col,off,file))
        print("----------------------------------------")
