import pandas as pd
from scipy import stats
import numpy as np
import os
import csv
from Resnet import _resnet,BasicBlock

import tempfile
import zipfile
from sklearn.model_selection import train_test_split
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.optim.lr_scheduler  import ReduceLROnPlateau 
from torch.distributions import Categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import torch.utils.data
from torch.nn import functional as F
import time
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import random
RANDOM_SEED=2
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)
from torch import Tensor
def valdata(is_val=False):
    data_column = 1
  

    dirname = "/data0401/new_test/"
    data = load_dataset(train_dir, dirname, 'train')
    
    #new_col=['Time','sensor_id','60023','60053','60027','60057','60031','60061','60035','60065','60039','60069','f0']
   
    data.dropna(inplace=True)
    print('{}_X.shape'.format(data.shape)
    x,y=data[ftl2],data['f0']
    windows=3
    new_x=pd.DataFrame()
    new_y=pd.DataFrame()
    for i in range(0,len(x)-windows+1):
        x_1=x.iloc[i]
        for j in range(1,windows):
            if x_1[0]==x.iloc[i+j][0]:
                x_1=pd.concat([x_1[0:5*j+1],x.iloc[i+j][1:6],x_1[5*j+1:],x.iloc[i+j][6:]],axis=0,ignore_index=True)
            else:
                j=0
                break
        if j==windows-1:
            y_1=y.iloc[i+windows-1]
            # x_1.drop(11,inplace=True) #第1个的sensor_id
            # x_1.drop(22,inplace=True) #第2个的sensor_id

            x_1=x_1.reset_index(drop=True) #重新编号
            new_x=new_x.append([x_1.to_list()],ignore_index=True)
            new_y=new_y.append([y_1],ignore_index=True)
    
    #x,y=data[['60053','60031','60023','60065','60057','60035','60027','60069','60061','60039']],data['f0']
    #x,y=data[['60053','60031','60023','60065','60057','60035','60027','60069','60061','60039','1','2']],data['f0']

    #x,y=data[['60053','60065','60057','60069','60061']],data['f0']
    #x,y=data[['60031','60023','60035','60027','60039']],data['f0']
    x=new_x.to_numpy()
    y=new_y.to_numpy()
    channels=1
    data_row=windows*10

  #  y=y.reshape(y.shape[0],1)
    x=x.reshape(x.shape[0],channels,data_row+1 ,1)
    channels=2
    xt=x[:,:,1:,]

    channels=2
    data_row=5*windows
    xt=xt.reshape(xt.shape[0],channels,data_row,1)
    
    labels_test=[]
    for i in range(xt.shape[0]):
        score=0
        book=np.ones(100)*100
        bookv=np.ones(100)*(-1)
        for j in range(5):
            if xt[i][1][10+j]<y[i]*0.5:
                continue
            k=min((y[i]-xt[i][1][10+j]%y[i]),xt[i][1][10+j]%y[i])
            if k==(y[i]-xt[i][1][10+j]%y[i]):
                r=int(xt[i][1][10+j]/y[i])+1
            else:
                r=int(xt[i][1][10+j]/y[i])
            if k<=y[i]*0.25 and k<book[r]:
                # score=score+2**j
                book[r]=k
                bookv[r]=j
        for s in range(100):
            if bookv[s]<0:
                continue
            score=score+2**bookv[s]
        labels_test.append(score)
    
    print("strart")
    input_shape=(channels,data_row,data_column)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = resnet(input_shape, 64, num, dropout).to(device)
    real_val_index=np.array(range(len(y))).reshape(len(y),1)
    yt=torch.Tensor(y)
    labels_test=torch.Tensor(labels_test)
    xt=xt.astype(float)
    rvaldata=MyDataset(xt,labels_test,yt,real_val_index)

    validate_loader = torch.utils.data.DataLoader(rvaldata,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=0)
    if is_val:
        return validate_loader,rvaldata,labels_test
    else:
        return validate_loader,rvaldata


def tocsv(filename,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(datas)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2dx(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride =  _single(stride)
        padding = _single(padding)
        dilation =  _single(dilation)
        super(Conv2dx, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False,  _single(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # padding_cols = max(0, (out_rows - 1) * stride[0] +
    #                     (filter_rows - 1) * dilation[0] + 1 - input_rows)
    # cols_odd = (padding_rows % 2 != 0)

    if rows_odd :
        input = pad(input, [0, 0, 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2,0),
                  dilation=dilation, groups=groups)



class Bottleneck(nn.Module):

    def __init__(self,input_shape, n_feature_maps,  dropout,dilat):
        super().__init__()
        if dilat==2: #下采样
            self.x_total=nn.Conv2d(input_shape,n_feature_maps, (16-dilat+1, 1),stride = (dilat,1), padding=(7,0))
        else:
            self.x_total=Conv2dx(input_shape,n_feature_maps, (16, 1),stride = (1,1), padding=(7,0))
        self.relu=nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(n_feature_maps)  # 853
        self.conv_x1 = Conv2dx(n_feature_maps,n_feature_maps, (16, 1),padding=(15,0))
        self.bn2 = nn.BatchNorm2d(n_feature_maps)
        self.conv_y1=Conv2dx(n_feature_maps,n_feature_maps * 2, (16, 1), padding=(15,0))
        self.bn3 =  nn.BatchNorm2d(n_feature_maps * 2)
        self.drop=nn.Dropout(p=dropout)
        self.conv_z1=Conv2dx(n_feature_maps * 2,n_feature_maps, (8, 1),padding=(7,0))
        self.bn4=nn.BatchNorm2d(n_feature_maps)
        self.is_expand_channels = not (input_shape == n_feature_maps)
        self.shortcut_y1 = Conv2dx(input_shape,n_feature_maps, (1, 1),stride = (dilat,1), padding=(0,0))
        self.shortcut_y2 = nn.BatchNorm2d(n_feature_maps)
        self.shortcut_y3 =nn.BatchNorm2d(n_feature_maps)
    def forward(self, x_input):
       # print('build conv_x')
        #x=self.drop(x)
        x_init=self.x_total(x_input)
        x=self.bn1(x_init)
        x=self.relu(x)
        x=self.conv_x1(x)
        x=self.bn2(x)
        x=self.relu(x)
       # print('build conv_y')
        x=self.conv_y1(x)
        x=self.bn3(x)
        x=self.relu(x)
      #  print('build conv_z')
        x=self.conv_z1(x)
        x=self.bn4(x)

        if self.is_expand_channels:
            x_short=self.shortcut_y1(x_input)
            x_short=self.shortcut_y2(x_short)
        else:
            print(1)
            x_short=self.shortcut_y3(x_input)


        return x,x_short
class SpatialAttentionModule(nn.Module):
    def __init__(self,input_shape):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7,1), stride=1, padding=(3,0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out



class resnet(nn.Module):
    def __init__(
        self, input_shape, n_feature_maps, nb_classes, dropout
    ) -> None:
        super().__init__()
        self. nb_classes= nb_classes
        self.n_feature_maps=n_feature_maps
        self.input_shape=input_shape[0]
        self.dropout=dropout
        self.relu=nn.ReLU(inplace=True)
        self.relu_out=nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.input_shape,self.n_feature_maps,dropout,1)
        self.layer2 = self._make_layer(self.n_feature_maps,self.n_feature_maps*2,dropout,1)
        self.SpatialAttentionModule=SpatialAttentionModule(input_shape)

        self.full=nn.AdaptiveAvgPool2d((1, 1))
        self.drop=nn.Dropout(p=dropout)
        self.linear=nn.Linear(self.n_feature_maps*2, self.nb_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
        self,
        input_shape,
        n_feature_maps,
        dropout,
        dilat,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            Bottleneck(
                input_shape,n_feature_maps, dropout,dilat,
            ))
        

        return nn.Sequential(*layers)

    def _forward_impl(self,x_input):
        x1,x_short1= self.layer1(x_input)     
        x1_1=x_short1+x1
        #sa1
        y_sa=self.SpatialAttentionModule(x1_1)
        x1_1=torch.mul(x1_1,y_sa)   
        x1_1=self.relu(x1_1)
        y=self.drop(x1_1)
        x1,x_short1= self.layer2(y)        
        x1_1=x_short1+x1
        #sa1
        y_sa=self.SpatialAttentionModule(x1_1)
        x1_1=torch.mul(x1_1,y_sa)   
        x1_1=self.relu(x1_1)
        y=self.drop(x1_1)
        y=self.full(y).squeeze()   
        #full=torch.flatten(y,1)
        out=self.linear(y)
        return out
        
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


from torch.utils.data import Dataset    
class MyDataset(Dataset):
    def __init__(self, x_data, label,label_f,data_index):
        super(MyDataset, self).__init__()
        self.x_data =torch.Tensor(x_data)
        self.y_data = torch.Tensor(label)
        self.f_data=torch.Tensor(label_f)
        self.id=torch.IntTensor(data_index)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
       
        return self.x_data[index].to(device), self.y_data[index].to(device),self.f_data[index].to(device),self.id[index].to(device)
# import visdom
import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def softmax(x):
    """Compute the softmax of vector x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

def trainTestSplit(X,y,test_size=0.3):
    X_num=X.shape[0]
    train_index=list(range(X_num))
    test_index=[]
    test_num=int(X_num*test_size)
    last_num=0
    num=1

    for k in range(1,len(y)):
        if y[k]==y[k-1] and k<len(y)-1:  #最后一个元素不比较
            num=num+1
        else:
            #randomIndex=int(np.random.uniform(0,len(train_index)))
            if k==len(y)-1:
                num=num+1
            randomIndex=list(np.arange(int(last_num+int(num*0.7)),int(last_num+num),1))
            last_num=last_num+num
            num=1
            for i in randomIndex:
                if i>=len(X):
                    break
                test_index.append(i)
                train_index.remove(i)

    train=[]
    y_train=[]
    test=[]
    y_test=[]
    for i in train_index:
        train.append(X[i])
        y_train.append(y[i])
    for i in test_index:
        test.append(X[i])
        y_test.append(y[i])
    return np.array(train),np.array(test),np.array(y_train),np.array(y_test)

import pandas as pd
from scipy import stats
import numpy as np
import os
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=0)

    return dataframe
def load_dataset(data_rootdir, dirname, group):
    '''
    该函数实现将训练数据或测试数据文件列表堆叠为三维数组
    '''
    filename_list = []
    filepath_list = []

    for rootdir, dirnames, filenames in os.walk(data_rootdir + dirname):
        for filename in filenames:
            filename_list.append(filename)
            filepath_list.append(os.path.join(rootdir, filename))
        #print(filename_list)
        #print(filepath_list)
    X = load_file(filepath_list[0])
    for filepath in filepath_list[1:]:
        #X.append(load_file(filepath))
        X=X.append(load_file(filepath),ignore_index=True)

    print('{}_X.shape:{}\n'.format(group,X.shape))
    return X
def Denary2Binary(n): 
    '''convert denary integer n to binary string bStr''' 
    bStr = '' 
    if n < 0:  raise ValueError
    if n == 0: return '0' 
    while n > 0: 
        bStr = str(n % 2) + bStr 
        n = n >> 1 
    return bStr 
  
def int2bin(n, count=24): 
    """returns the binary of integer n, using count number of digits""" 
    str2="".join([str((n >> y) & 1) for y in range(count-1, -1, -1)]) 
    ans=[]
    # for i in range(5):
    #     ans.append(int(str2[4-i]))
    for i in range(5):
        ans.append(int(str2[4-i]))
    return ans
class mlp(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(mlp, self).__init__()
        self.flatten=nn.Flatten()
        self.input=input_shape
        self.out=out_shape
        self.linear=nn.Linear(self.input,self.out)
        self.p=torch.nn.Parameter(torch.zeros(out_shape))
        self.relu=nn.ReLU(inplace=True)
    def forward(self,w,out_mid):
        w=self.linear(w)
        out_mid=self.flatten(out_mid)
        out=w*out_mid+self.p
        out=self.relu(out)
        
        
        return out
def count_acc(x,y):
    count=0.0
    for i in range(len(x)):
        count+=np.sum(x[i]==y[i])
    return count/5
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':

    train_dir = "/zhangjunru/zhangjunru_/Freq/data/"
    dirname = "/data0401/new_train/"
    data = load_dataset(train_dir, dirname, 'train')
    
    #new_col=['Time','sensor_id','60023','60053','60027','60057','60031','60061','60035','60065','60039','60069','f0']
   
    data.dropna(inplace=True)
    print('{}_X.shape'.format(data.shape))


    num = 32
    channels = 1
    nb_epochs = 500
    batch_size = 16
    dropout = 0.5
    data_column = 1
    ftl=['60023','60027','60031','60035','60039','60053','60057','60061','60065','60069']
    #ftl=['60053','60057','60061','60065','60069']
    ftl2=['sensor_id','60023','60027','60031','60035','60039','60053','60057','60061','60065','60069']
    x,y=data[ftl2],data['f0']

    ####转换为三个历史数据，每个特征f g 分别是是5*3=15个数字
    windows=3
    new_x=pd.DataFrame()
    new_y=pd.DataFrame()
    for i in range(0,len(x)-windows+1):
        x_1=x.iloc[i]
        
        for j in range(1,windows):
            if x_1[0]==x.iloc[i+j][0]:
                x_1=pd.concat([x_1[0:5*j+1],x.iloc[i+j][1:6],x_1[5*j+1:],x.iloc[i+j][6:]],axis=0,ignore_index=True)
        
                #x_1 = pd.merge(x_1, x.iloc[i-j], left_index=True, right_index=True, how='left')
            else:
                j=0
                break
        
        if j==windows-1:
            y_1=y.iloc[i+windows-1]
            # x_1.drop(11,inplace=True) #第1个的sensor_id
            # x_1.drop(22,inplace=True) #第2个的sensor_id

            x_1=x_1.reset_index(drop=True) #重新编号
            new_x=new_x.append([x_1.to_list()],ignore_index=True)
            new_y=new_y.append([y_1],ignore_index=True)
    data_row=windows*10

   # x,y=data[['60053','60031','60023','60065','60057','60035','60027','60069','60061','60039','1','2']],data['f0']
    x=new_x.to_numpy()
    y=new_y.to_numpy()
    y=y.reshape(y.shape[0],1)
    x=x.reshape(x.shape[0],channels,data_row+1,1)
    x_train11,x_val11,y_train11,y_val11=trainTestSplit(x,y,test_size = 0.3)
    #x_train11,x_val11,y_train11,y_val11=train_test_split(x,y,test_size = 0.3,shuffle=True)
    #x_train11,x_val11,y_train11,y_val11=train_test_split(x,y,test_size = 0.3,stratify=y) #stratify的作用是：保持测试集与整个数据集里y的数据分类比例一致
    x_train1= x_train11[:,:,1:,].astype(float)
    x_val1= x_val11[:,:,1:,].astype(float)
    y_train1= y_train11.astype(float)
    y_val1=y_val11.astype(float)
    x_val1= x_val1.astype(float)
    x_train1= x_train1.astype(float)

    channels=2
    data_row=windows*5
    x_train1=x_train1.reshape(x_train1.shape[0],channels,data_row,1)
    x_val1=x_val1.reshape(x_val1.shape[0],channels,data_row,1)
    print("end")
    input_shape=(channels,data_row,data_column)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from torchsummary import summary
    from tqdm import tqdm
    
    
    dummy_input = torch.rand(batch_size, channels,data_row,data_column).to(device)
    epochs=nb_epochs
    best_acc = 0.0
    best_testloss=100
    repeat=1
    acc_list=[]
    train_index=np.array(range(len(y_train1))).reshape(len(y_train1),1)
    val_index=np.array(range(len(y_val1))).reshape(len(y_val1),1)
    labels_train=[]
    for i in range(x_train1.shape[0]):
        score=0
        book=np.ones(100)*100
        bookv=np.ones(100)*(-1)
        for j in range(5):
            if x_train1[i][1][10+j]<y_train1[i]*0.5:
                continue
            k=min((y_train1[i]-x_train1[i][1][10+j]%y_train1[i]),x_train1[i][1][10+j]%y_train1[i])
            if k==(y_train1[i]-x_train1[i][1][10+j]%y_train1[i]):
                r=int(x_train1[i][1][10+j]/y_train1[i])+1
            else:
                r=int(x_train1[i][1][10+j]/y_train1[i])
            if k<=y_train1[i]*0.25 and k<book[r]:
                #score=score+2**j
                book[r]=k
                bookv[r]=j
        for s in range(100):
            if bookv[s]<0:
                continue
            score=score+2**bookv[s]
        labels_train.append(score)
    labels_train=torch.Tensor(labels_train)
    labels_val=[]
    for i in range(x_val1.shape[0]):
        score=0
        book=np.ones(100)*100
        bookv=np.ones(100)*(-1)
        for j in range(5):
            if x_val1[i][1][10+j]<y_val1[i]*0.5:
                continue
            k=min((y_val1[i]-x_val1[i][1][10+j]%y_val1[i]),x_val1[i][1][10+j]%y_val1[i])
            if k==(y_val1[i]-x_val1[i][1][10+j]%y_val1[i]):
                r=int(x_val1[i][1][10+j]/y_val1[i])+1
            else:
                r=int(x_val1[i][1][10+j]/y_val1[i])
            if k<=y_val1[i]*0.25 and k<book[r]:
                book[r]=k
                bookv[r]=j
        for s in range(100):
            if bookv[s]<0:
                continue
            score=score+2**bookv[s]
        labels_val.append(score)
    labels_val=torch.Tensor(labels_val)
    train_data=MyDataset(x_train1,labels_train,y_train1,train_index)
    validation_data=MyDataset(x_val1,labels_val,y_val1,val_index)

    set_seed(RANDOM_SEED)
    num=32
    input_shape=(channels,data_row,data_column)
    model_in = resnet(input_shape, 64, num, dropout).to(device)
   # model_in=_resnet(BasicBlock, [2, 2, 2,2],None,None).to(device)
    print(summary(model_in,input_shape))

    for t in range(repeat):
        writer = SummaryWriter("runs/224_f")
        optimizer=optim.Adam(model_in.parameters(),lr=0.0001,weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=0.0001)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0)

        validate_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size=batch_size, shuffle=True,num_workers=0)
        rvalidate_loader,rvalidation_data,labels_test=valdata(True)
        rval_num = len(rvalidation_data)
        rval_steps=len(rvalidate_loader)
        train_num=len(train_data)
        val_num = len(validation_data)
        train_steps = len(train_loader) #多少个batch sum(train)/batch
        val_steps=len(validate_loader)
        nametime="new"+str(int(time.time()))
        name1=nametime+"model_fenlei.pth"
        save_path1 =  "/zhangjunru/zhangjunru_/Freq/modellog/"+name1
        print(name1)
        loss_function_mid =torch.nn.CrossEntropyLoss() 
        loss_function_out=torch.nn.L1Loss()
        name=nametime+"yhdqfenlei.log"

        logger = get_logger( "/zhangjunru/zhangjunru_/Freq/modellog/"+name)
        logger.info('start training!')
        record=np.zeros(len(train_data))
        
        for epoch in range(epochs):
            train_y=[]
            val_y=[]
            train_y_f=[]
            val_y_f=[]
            print("学习率：",epoch,optimizer.state_dict()['param_groups'][0]['lr'])
            # train
            # model_out.train()
            model_in.train()
            # model_mid.train()
            running_loss = 0.0
            sumvalloss=0.0
            sumrunloss = 0.0
            train_acc_num=0.0
            train_acc=0.0
            
            train_bar = tqdm(train_loader)
            end2=time.time()
            for step, data in enumerate(train_bar):
                outputs_mid=np.zeros(15)
                x_train,labels,labels_f,train_index = data
                optimizer.zero_grad()
                
                outputs = model_in(x_train)
                #outputs=torch.unsqueeze(outputs,0) #[batch,classes]
                if len(outputs.shape)==1:
                    outputs=torch.unsqueeze(outputs,0)
                loss_mid = loss_function_mid(outputs, labels.long())
                outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
                for i in range(len(outputs)):
                    outputs_bi=int2bin(outputs[i],5)
                    outputs_bi_f=int2bin(int(labels[i].item()),5)
                    train_y.append(outputs[i])
                    train_y_f.append([outputs_bi_f,outputs_bi])
                    train_acc_num+=count_acc(outputs_bi,outputs_bi_f)
                loss=loss_mid
                sumloss=loss.item()*len(labels)
                train_index=train_index.cpu() 
                
                #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                if epoch>=0:
                    if loss.item()>best_testloss:
                        train_index=train_index.cpu()
                        record[train_index]= record[train_index]+1
                loss.backward()
                optimizer.step()
                

                # print statistics
                running_loss += loss.item()
                train_acc+=np.sum(outputs==labels.cpu().numpy())
                
                sumrunloss+=sumloss
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,loss)
                                
            train_accurate1 = train_acc / train_num
            train_accurate_num=train_acc_num/train_num                                                          
            # validate
            model_in.eval()
            acc_mid = 0.0  # accumulate accurate number / epoch
            acc_out=0.0
            acc_mid_num=0.0
            loss_out_sum=0.0
            loss_mid_sum=0.0
            start3=time.time()
            avg_val_loss = 0.
            with torch.no_grad():
                #val_bar = tqdm(validate_loader)
                for val_datam in validate_loader:
                    val_data, val_labels,val_f,val_index = val_datam
                    outputs_mid=np.zeros(15)
                    outputs = model_in(val_data)
                    if len(outputs.shape)==1:
                        outputs=torch.unsqueeze(outputs,0)
                    loss_mid = loss_function_mid(outputs, val_labels.long())
                    outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
                    for i in range(len(outputs)):
                        outputs_bi=int2bin(outputs[i],5)
                        outputs_bi_f=int2bin(int(val_labels[i].item()),5)
                        val_y.append(outputs[i])
                        val_y_f.append([outputs_bi_f,outputs_bi])
                        acc_mid_num+=count_acc(outputs_bi,outputs_bi_f)

                    val_loss=loss_mid
                    avg_val_loss += val_loss.item() 

                    acc_mid +=  np.sum(outputs==val_labels.cpu().numpy())
            val_accurate1 = acc_mid / val_num 
            val_accurate_num=acc_mid_num/val_num
            val_accurate2 = acc_out / val_num
            end3=time.time()
            if scheduler is not None:
                scheduler.step(val_accurate_num)

            logger.info('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_acc: %.3f train_acc_num: %.3f val_accuracy1: %.3f val_acc_num:%.3f ' %
                (epoch + 1, running_loss / train_steps,avg_val_loss/val_steps,train_accurate1,train_accurate_num,val_accurate1,val_accurate_num))
            flag=0
            if avg_val_loss/val_steps<=best_testloss:
                best_testloss = avg_val_loss/val_steps
                best_testacc = val_accurate1
                best_testaccnum=val_accurate_num
                best_trainacc=train_accurate1
                best_trainaccnum=train_accurate_num
                best_trainloss=running_loss / train_steps
                train_ya=train_y
                train_y_fa=train_y_f
                val_ya=val_y
                val_y_fa=val_y_f
                model_in.eval()
                torch.save(model_in.state_dict(), save_path1)
                flag=1
        
            acc_mid = 0.0  # accumulate accurate number / epoch
            acc_out=0.0
            acc_mid_num=0.0
            loss_out_sum=0.0
            loss_mid_sum=0.0
            start3=time.time()
            avg_val_loss = 0.
            model_in.eval()
            with torch.no_grad():
                val_bar = tqdm(rvalidate_loader)
                for val_datam in val_bar:
                    val_data, val_labels,val_f,val_index = val_datam
                    outputs_mid=np.zeros(15)
                    outputs = model_in(val_data)
                    if len(outputs.shape)==1:
                        outputs=torch.unsqueeze(outputs,0)
                    loss_mid = loss_function_mid(outputs, val_labels.long())
                    outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
                    for i in range(len(outputs)):
                        outputs_bi=int2bin(outputs[i],5)
                        outputs_bi_f=int2bin(int(val_labels[i].item()),5)
                        val_y.append(outputs[i])
                        val_y_f.append([outputs_bi_f,outputs_bi])
                        acc_mid_num+=count_acc(outputs_bi,outputs_bi_f)
                    val_loss=loss_mid
                    avg_val_loss += val_loss.item()
                    acc_mid +=  np.sum(outputs==val_labels.cpu().numpy()
            val_accurate1 = acc_mid / rval_num
            val_accurate_num=acc_mid_num/rval_num
            val_accurate2 = acc_out / rval_num 
            end3=time.time()
            if scheduler is not None:
                scheduler.step(val_accurate_num)

            logger.info('Rval_loss: %.3f Rval_accuracy1: %.3f Rval_acc_num:%.3f ' %
                (avg_val_loss/rval_steps,val_accurate1,val_accurate_num))
            if flag:
                best_valloss=avg_val_loss/rval_steps
                best_valacc=val_accurate1
                best_valaccnum=val_accurate_num
                best_epoch=epoch+1
        print('Finished Training') #不加schedular
        logger.info('best:[epoch %d] train_loss: %.3f train_acc:%.3f train_accnum: %.3f test_loss: %.3f test_acc:%.3f  test_accnum: %.3f val_loss:%.3f  val_accuracy: %.3f val_accnum:%.3f ' %
                (best_epoch, best_trainloss, best_trainacc, best_trainaccnum, best_testloss,best_testacc, best_testaccnum, best_valloss,best_valacc, best_valaccnum))

        # 测试
        print("------------------------ 测试中---------------------------")
        #evaluation of the model
        print('Baseline Test Loss: %.2f'%( best_testloss))
        print('Baseline Test ACC: %.2f'%(best_testaccnum))
       # acc_list.append(val_accurate)
   
###验证
    print("——————————预测——————————")

    validate_loader,rvaldata,labels_test=rvalidate_loader,rvalidation_data,labels_test                                      
    num=32
    input_shape=(channels,data_row,data_column)
    model_in = resnet(input_shape, 64, num, dropout).to(device)
   # model_in=_resnet(BasicBlock, [2, 2, 2,2],None,None).to(device)
    val_num = len(rvaldata)
    val_steps=len(validate_loader)
    model_in.load_state_dict(torch.load(save_path1))
    model_in.eval()
    val_y=[]
    val_y_f=[]
    acc_mid = 0.0  # accumulate accurate number / epoch
    acc_out=0.0
    acc_mid_num=0.0
    loss_out_sum=0.0
    loss_mid_sum=0.0
    start3=time.time()
    avg_val_loss = 0.

    with torch.no_grad():
        #val_bar = tqdm(validate_loader)
        for val_datam in validate_loader:
            val_data, val_labels,val_f,val_index = val_datam
            outputs_mid=np.zeros(15)
            outputs = model_in(val_data)
            if len(outputs.shape)==1:
                outputs=torch.unsqueeze(outputs,0)
            loss_mid = loss_function_mid(outputs, val_labels.long())
            outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
            for i in range(len(outputs)):
                outputs_bi=int2bin(outputs[i],5)
                outputs_bi_f=int2bin(int(val_labels[i].item()),5)
                val_y.append(outputs[i])
                val_y_f.append([outputs_bi_f,outputs_bi])
                acc_mid_num+=count_acc(outputs_bi,outputs_bi_f)

            val_loss=loss_mid
            avg_val_loss += val_loss.item() 
            acc_mid +=  np.sum(outputs==val_labels.cpu().numpy())
           
    val_accurate1 = acc_mid / val_num 
    val_accurate2 = acc_out / val_num 
    val_acc_num=acc_mid_num/val_num
# print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))
    logger.info('Final_val:[epoch %d] val_loss: %.3f  val_accuracy: %.3f val_acc_num:.%3f' %
        (epoch + 1, avg_val_loss/val_steps,val_accurate1,val_acc_num))
    name=nametime+"val_class.csv"
    save_path =  '/zhangjunru/zhangjunru_/Freq/modellog/'+name

    #ans=pd.concat([pd.DataFrame(x.squeeze()),pd.DataFrame(y),pd.DataFrame(predict_y),pd.DataFrame(predict_y2)], axis=1)
    ans=pd.concat([pd.DataFrame(x.squeeze()),pd.DataFrame(y),pd.DataFrame(labels_test.cpu()),pd.DataFrame(val_y),pd.DataFrame(val_y_f)], axis=1)
    test_answer = pd.DataFrame(ans).to_csv(save_path, index=None)

#——————保存结果————————
    print("——————————保存结果——————————")
    train_loader = torch.utils.data.DataLoader(train_data,
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=0)

    validate_loader = torch.utils.data.DataLoader(validation_data,
                                    batch_size=batch_size, shuffle=False,num_workers=0)

    print("test_shape:",len(validation_data))
    train_ya,train_y_fa,val_ya,val_y_fa=[],[],[],[]
    with torch.no_grad():
        #val_bar = tqdm(train_loader)
        for val_datam in train_loader:
            val_data, val_labels,val_f,val_index = val_datam
            outputs_mid=np.zeros(15)
            outputs = model_in(val_data)
            if len(outputs.shape)==1:
                outputs=torch.unsqueeze(outputs,0)
            loss_mid = loss_function_mid(outputs, val_labels.long())
            outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
            for i in range(len(outputs)):
                outputs_bi=int2bin(outputs[i],5)
                outputs_bi_f=int2bin(int(val_labels[i].item()),5)
                train_ya.append(outputs[i])
                train_y_fa.append([outputs_bi_f,outputs_bi])
       # val_bar = tqdm(validate_loader)
        for val_datam in validate_loader:
            val_data, val_labels,val_f,val_index = val_datam
            outputs_mid=np.zeros(15)
            outputs = model_in(val_data)
            if len(outputs.shape)==1:
                outputs=torch.unsqueeze(outputs,0)
            loss_mid = loss_function_mid(outputs, val_labels.long())
            outputs=np.argmax(outputs.cpu().detach().numpy(),axis=1)
            for i in range(len(outputs)):
                outputs_bi=int2bin(outputs[i],5)
                outputs_bi_f=int2bin(int(val_labels[i].item()),5)
                val_ya.append(outputs[i])
                val_y_fa.append([outputs_bi_f,outputs_bi])

    name_ans=nametime+"train_test_class.csv"
    save_path = '/zhangjunru/zhangjunru_/Freq/modellog/'+name_ans
    ans=pd.concat([pd.DataFrame(x_train11.squeeze()),pd.DataFrame(y_train1),pd.DataFrame(labels_train),pd.DataFrame(train_ya),pd.DataFrame(train_y_fa),pd.DataFrame(record),pd.DataFrame(x_val11.squeeze()),pd.DataFrame(y_val1),pd.DataFrame(labels_val),pd.DataFrame(val_ya),pd.DataFrame(val_y_fa)], axis=1)

    test_answer = pd.DataFrame(ans).to_csv(save_path, index=None)

    print(name_ans,name)
    print("end")

