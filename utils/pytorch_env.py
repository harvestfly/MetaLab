from some_libs import *
import torch


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use in python(GB):', memoryUse)

def pytorch_env( fix_seed=None ):
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    # from subprocess import call
    # call(["nvcc", "--version"]) does not work
    # ! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    use_cuda = torch.cuda.is_available()
    print("USE CUDA=" + str(use_cuda))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    Tensor = FloatTensor
    cpuStats()

    if fix_seed is not None:        # fix seed
        seed = fix_seed #17 * 19
        print("!!! __pyTorch FIX SEED={} use_cuda={}!!!".format(seed,use_cuda) )
        random.seed(seed-1)
        np.random.seed(seed)
        torch.manual_seed(seed+1)
        if use_cuda:
            torch.cuda.manual_seed(seed+2)
            torch.cuda.manual_seed_all(seed+3)
            torch.backends.cudnn.deterministic = True

    print("===== torch_init device={}".format(device))
    return device

