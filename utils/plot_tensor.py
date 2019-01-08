import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torchvision.utils import make_grid
import numpy as np
import sys

def plot_batch_grid( tensor, label=None ):
    nImg = tensor.shape[0]
    imglist=[]
    for i in range(nImg):
        imglist.append(tensor[i,:,:,:])
    grids = make_grid(imglist, padding=10, normalize=True, scale_each=False )
    np_grid = grids.cpu().numpy()
    plt.imshow(np.transpose(np_grid, (1, 2, 0)), interpolation='nearest')

def plot_tensor_4d_slide( tensor ):
    array = tensor.cpu().numpy()
    #array = np.random.rand(300,300,10,9)
    # assuming you have for each i=Temperature index and j =Opacity index
    # an image array(:,:,i,j)

    fig, ax = plt.subplots()
    l = ax.imshow(array[0,0,:,:], origin = 'lower')

    axT = fig.add_axes([0.2, 0.95, 0.65, 0.03])
    axO = fig.add_axes([0.2, 0.90, 0.65, 0.03])

    nItem=array.shape[0];        nChan=array.shape[1]
    sliderT = Slider(axT, 'item', 0, nItem, valinit=0, valfmt='%i')
    sliderO = Slider(axO, 'channel', 0, nChan, valinit=0, valfmt='%i')

    def update(val):
        i = int(sliderT.val)
        j = int(sliderO.val)
        # im = array[:,:,i,j]
        im = array[ i, j,:, :]
        l.set_data(im)
        fig.canvas.draw_idle()

    sliderT.on_changed(update)
    sliderO.on_changed(update)

    plt.show()

if __name__ == '__main__':
    array = np.random.rand(300, 300, 10, 9)
    plot_tensor_4d(array)
    #sys.exit(main())