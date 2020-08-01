
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
num = 0
test = 0
# hyperparameters
test_rate = 1/8
batch_size = 512
tbatch_size = batch_size
lr = 0.001
seg_label = 1
num_epochs = 200
if test == 1:
    log_dir = './compare32/'+str(num)+'test'
else:
    log_dir = './mae_nospade2'

load_dir = './mae_nospade2'
#load_dir =0
if(load_dir != 0):
    test = 1
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print('torch.version'+str(torch.__version__))

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(
            norm_nc, affine=False)  # norm_nc is input size of channel

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # label_nc is channel of seg
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):  # fin is input channel, fout is output channel
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        # Spade 1:input channel 2:seg channel
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Generator(nn.Module):

    def __init__(self, seg_label):
        super(Generator, self).__init__()
        self.seg_label = seg_label
        # encode

        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.a2 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2, 2)
        # 16*16
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.a3 = nn.ReLU()
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.a4 = nn.ReLU()
        self.m2 = nn.MaxPool2d(2, 2)
        # 8*8

        self.c7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.a7 = nn.ReLU()
        self.c8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.a8 = nn.ReLU()
        self.m4 = nn.MaxPool2d(2, 2)
        # 4*4
        self.c9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.a9 = nn.ReLU()

        # decode

        self.b1 = nn.Conv2d(513, 512, kernel_size=3,
                            stride=1, padding=1)  # >512*4*4

        self.conv_img1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)
        self.conv_img3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)

        self.conv_img4 = nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_img5 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)

        self.conv_img6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_img7 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.conv_img8 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.rel = nn.ReLU()
        self.bn = nn.Sigmoid()  # 勾配消失問題に関わるから不適切

        # initialize_weights(self)

    def forward(self, bf, vol, seg):
        x = torch.cat((bf, seg), dim=1)
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.m1(x)
        x = self.c3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.a4(x)
        x = self.m2(x)
        x = self.c7(x)
        x = self.a7(x)
        x = self.c8(x)
        x = self.a8(x)
        x = self.m4(x)
        x = self.c9(x)
        x = self.a9(x)

        # concat vol
        x2 = vol
        x = torch.cat((x, x2), dim=1)

        x = self.b1(x)
        x = self.up(x)
        x = self.conv_img1(x)
        x = self.rel(x)
        x = self.conv_img3(x)
        x = self.rel(x)

        x = self.up(x)
        x = self.conv_img4(x)
        x = self.rel(x)
        x = self.conv_img5(x)
        x = self.rel(x)

        x = self.up(x)
        x = self.conv_img6(x)
        x = self.rel(x)
        x = self.conv_img7(x)
        x = self.rel(x)

        x = self.conv_img8(x)
        x = self.bn(x)

        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):  # isinstance detect the type same or not
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


G = Generator(seg_label)

if torch.cuda.device_count() > 1:
    if(load_dir):
        PATH = os.path.join(load_dir, 'GoodG.pth')
        G.load_state_dict(torch.load(PATH))
        print('load for multi!')
    print('\nYou can use {} GPU'.format(torch.cuda.device_count()))
    G = nn.DataParallel(G).cuda()
else:
    print('\n you use only one GPU')
    G.cuda()
    if (load_dir):
        PATH = os.path.join(load_dir, 'GoodG.pth')
        G.load_state_dict(torch.load(PATH))
        print('load!')

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

# loss
criterion = nn.BCELoss()  # binarry cross entropy
MAE = nn.L1Loss()  # MAE which is used in Korea one
# for images
rBfx = pd.read_csv(
    filepath_or_buffer="/home/user/Desktop/forconfirm32(42000)/32bodyfx.csv", header=None)
rBfy = pd.read_csv(
    filepath_or_buffer="/home/user/Desktop/forconfirm32(42000)/32bodyfy.csv", header=None)
rEmpty = pd.read_csv(
    filepath_or_buffer="/home/user/Desktop/forconfirm32(42000)/32empty.csv", header=None)
rVol = pd.read_csv(
    filepath_or_buffer="/home/user/Desktop/forconfirm32(42000)/32vol.csv", header=None)
rRho = pd.read_csv(
    filepath_or_buffer="/home/user/Desktop/forconfirm32(42000)/32rho.csv", header=None)

rBfx = np.array(rBfx, dtype=np.float32)
rBfy = np.array(rBfy, dtype=np.float32)
rEmpty = np.array(rEmpty, dtype=np.float32)
rVol = np.array(rVol, dtype=np.float32)
rRho = np.array(rRho, dtype=np.float32)

rBfx = np.reshape(rBfx, (-1, 1, 32, 32))  # サンプル数*チャネル数*縦*横
rBfy = np.reshape(rBfy, (-1, 1, 32, 32))
rBf = np.hstack((rBfx, rBfy))
rEmpty = np.reshape(rEmpty, (-1, 1, 32, 32))
rRho = np.reshape(rRho, (-1, 1, 32, 32))
rVol = np.concatenate([rVol, rVol, rVol, rVol], axis=1)  # 4 elements
rVol = np.concatenate([rVol, rVol, rVol, rVol], axis=1)  # 16 elements
rVol = np.reshape(rVol, (-1, 1, 4, 4))

rBf, rRho, rVol, rEmpty = torch.from_numpy(rBf), torch.from_numpy(
    rRho), torch.from_numpy(rVol), torch.from_numpy(rEmpty)
'''save_image(1-rRho, os.path.join(log_dir, 'ideal_image.png'))
for i in range(rRho.size()[0]):
    save_image(1-rRho[i,:,:,:], os.path.join(log_dir, 'ideal{}.png'.format(i)))'''
if test == 1:
    # test
    Bfx = pd.read_csv(filepath_or_buffer="/home/user/Desktop/alldata(add32)/32bodyfx.csv",
                      header=None, nrows=32*20*batch_size)
    Bfy = pd.read_csv(filepath_or_buffer="/home/user/Desktop/alldata(add32)/32bodyfy.csv",
                      header=None, nrows=32*20*batch_size)
    Empty = pd.read_csv(filepath_or_buffer="/home/user/Desktop/alldata(add32)/32empty.csv",
                        header=None, nrows=32*20*batch_size)
    Vol = pd.read_csv(filepath_or_buffer="/home/user/Desktop/alldata(add32)/32vol.csv",
                      header=None, nrows=20*batch_size)
    Rho = pd.read_csv(filepath_or_buffer="/home/user/Desktop/alldata(add32)/32rho.csv",
                      header=None, nrows=32*20*batch_size)
else:
    # load input dataset
    Bfx = pd.read_csv(
        filepath_or_buffer="/home/user/Desktop/alldata(add32)/32bodyfx.csv", header=None)
    Bfy = pd.read_csv(
        filepath_or_buffer="/home/user/Desktop/alldata(add32)/32bodyfy.csv", header=None)
    Empty = pd.read_csv(
        filepath_or_buffer="/home/user/Desktop/alldata(add32)/32empty.csv", header=None)
    Vol = pd.read_csv(
        filepath_or_buffer="/home/user/Desktop/alldata(add32)/32vol.csv", header=None)
    Rho = pd.read_csv(
        filepath_or_buffer="/home/user/Desktop/alldata(add32)/32rho.csv", header=None)

Bfx = np.array(Bfx, dtype=np.float32)
Bfy = np.array(Bfy, dtype=np.float32)
Empty = np.array(Empty, dtype=np.float32)
Vol = np.array(Vol, dtype=np.float32)
Rho = np.array(Rho, dtype=np.float32)

Bfx = np.reshape(Bfx, (-1, 1, 32, 32))  # サンプル数*チャネル数*縦*横
Bfy = np.reshape(Bfy, (-1, 1, 32, 32))
Bf = np.hstack((Bfx, Bfy))
Empty = np.reshape(Empty, (-1, 1, 32, 32))
Rho = np.reshape(Rho, (-1, 1, 32, 32))
Vol = np.concatenate([Vol, Vol, Vol, Vol], axis=1)  # 4 elements
Vol = np.concatenate([Vol, Vol, Vol, Vol], axis=1)  # 16 elements
Vol = np.reshape(Vol, (-1, 1, 4, 4))

if cuda:
    rBf, rRho, rVol, rEmpty = rBf.cuda(), rRho.cuda(), rVol.cuda(), rEmpty.cuda()
Bf_train, Bf_test, Empty_train, Empty_test, Vol_train, Vol_test, Rho_train, Rho_test = train_test_split(
    Bf, Empty, Vol, Rho, test_size=test_rate, random_state=0)
print(Bf_train.shape)
print(Bf_test.shape)

train = torch.utils.data.TensorDataset(torch.from_numpy(Bf_train), torch.from_numpy(
    Rho_train), torch.from_numpy(Vol_train), torch.from_numpy(Empty_train))
input_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True)
test = torch.utils.data.TensorDataset(torch.from_numpy(Bf_test), torch.from_numpy(
    Rho_test), torch.from_numpy(Vol_test), torch.from_numpy(Empty_test))
test_loader = torch.utils.data.DataLoader(
    test, batch_size=tbatch_size, shuffle=True)


# for the training
def train(G, criterion, MAE, G_optimizer, input_loader):
    # 訓練モードへ

    G.train()

    # 本物のラベルは1
    #y_real = Variable(torch.ones(batch_size, 1))
    y_real = torch.ones(batch_size, 1)
    # 偽物のラベルは0
    #y_fake = Variable(torch.zeros(batch_size, 1))
    y_fake = torch.zeros(batch_size, 1)
    if cuda:
        y_real = y_real.cuda()
        y_fake = y_fake.cuda()

    G_running_loss = 0
    for ibatch_idx, [input_images, real_images, vol, empty] in enumerate(input_loader):
        # 一番最後、バッチサイズに満たない場合は無視する
        if (input_images.size()[0] != batch_size or real_images.size()[0] != batch_size or vol.size()[0] != batch_size or empty.size()[0] != batch_size):
            break

        z = input_images  # 潜在ベクトル
        # 条件入力ベクトルを入れたい

        if cuda:
            real_images, z, vol, empty = real_images.cuda(), z.cuda(), vol.cuda(), empty.cuda()
        #real_images, z = Variable(real_images), Variable(z)

        # DiscriminatorにとってGeneratorが生成した偽物画像の認識結果は0（偽物）に近いほどよい
        # E[log(1 - D(G(z)))]
        # fake_imagesを通じて勾配がGに伝わらないようにdetach()して止める、variableだとこれで止まる
        # Generatorの更新
        G_optimizer.zero_grad()

        # GeneratorにとってGeneratorが生成した画像の認識結果は1（本物）に近いほどよい
        # E[log(D(G(z)))
        fake_images = G(z, vol, empty)

        G_loss = MAE(fake_images, real_images)  # GlossにMAEを加えた
        G_loss.backward()
        G_optimizer.step()
        G_running_loss += G_loss.data

    G_running_loss /= len(input_loader)

    return G_running_loss


history = {}
history['G_loss'] = []
history['test_loss'] = []

best_loss = 1000
if(load_dir == 0):
    for epoch in range(num_epochs):
        start = time.time()
        G_loss = train(G, criterion, MAE, G_optimizer, input_loader)
        elapsed_time = time.time()-start
        print(str(elapsed_time)+"sec")
        with torch.no_grad():
            test_loss = 0
        for tbatch_idx, [tinput_images, treal_images, tvol, tempty] in enumerate(test_loader):
            # 一番最後、バッチサイズに満たない場合は無視する
            if (tinput_images.size()[0] != tbatch_size or treal_images.size()[0] != tbatch_size or tvol.size()[0] != tbatch_size or tempty.size()[0] != tbatch_size):
                break
            tz = tinput_images  # 潜在ベクトル
            # 条件入力ベクトルを入れたい

            if cuda:
                tz, tvol, tempty = tz.cuda(), tvol.cuda(), tempty.cuda()
            start = time.time()
            #test_result=generate(epoch + 1, G, tz,tvol,tempty, log_dir)
            test_result = G(tz, tvol, tempty).data.cpu()
            loss = MAE(treal_images, test_result).cpu()
            test_loss = test_loss+loss.data

        test_loss = (test_loss / len(test_loader))
        if test_loss < best_loss:
            best_loss = test_loss
            '''samples= G(rBf,rVol,rEmpty).data.cpu()
            save_image(1-samples, os.path.join(log_dir, 'results.png'))'''
            if torch.cuda.device_count() > 1:
                torch.save(G.module.state_dict(),
                           os.path.join(log_dir, 'GoodG.pth'))
            else:
                torch.save(G.state_dict(), os.path.join(log_dir, 'GoodG.pth'))
            '''for i in range(samples.size()[0]):
                A=samples[i,0,:,:]
                with open(log_dir+'/'+str(epoch)+'rho'+str(num)+'.csv', 'a') as f_handle:
                    np.savetxt(f_handle, A)'''

        print('epoch %d, G_loss: %.4f test_loss: %.4f' %
              (epoch + 1, G_loss, test_loss))
        history['G_loss'].append(G_loss)
        history['test_loss'].append(test_loss)
        # 学習履歴を保存
        with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        if torch.cuda.device_count() > 1:
            torch.save(G.module.state_dict(),
                       os.path.join(log_dir, 'lastG.pth'))
        else:
            torch.save(G.state_dict(), os.path.join(log_dir, 'lastG.pth'))
else:

    for i in range(int(rBf.size()[0]/batch_size)):

        sBf = rBf[i*batch_size:(i+1)*batch_size, :, :, :]
        sVol = rVol[i*batch_size:(i+1)*batch_size, :, :, :]
        sEmpty = rEmpty[i*batch_size:(i+1)*batch_size, :, :, :]
        samples = G(sBf, sVol, sEmpty).data.cpu()
        '''samples = G(rBf, rVol, rEmpty).data.cpu()
        for i in range(samples.size()[0]):
            save_image(1 - samples[i, :, :, :], os.path.join(log_dir, 'result{}.png'.format(i)))'''
        for j in range(samples.size()[0]):

            A = samples[j, 0, :, :]
            with open(log_dir+'/rho'+str(num)+'.csv', 'a') as f_handle:
                np.savetxt(f_handle, A, delimiter=',')
