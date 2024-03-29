#!/usr/bin/env python3

import os
import os.path as osp
import glob
import re
import datetime
import argparse

import numpy as np
import pandas as pd

import torch as t
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

import torchvision
import torchvision.transforms as transforms

from parser import parser

class ToTensor:

    def __init__(self,):
        pass
    def __call__(self,sample):
        array, label = sample['array'],sample['label']
        array = t.from_numpy(array.astype(np.float32))
        label = t.tensor(label).type(t.LongTensor)

        return {'array':array,
                'label':label}


class RotateK90:

    def __init__(self,):
       pass

    def __call__(self,sample):
        k = np.round( np.random.uniform(0,3) * 10,0).astype(int)
        return {'array':np.rot90(sample['array'],k),
                'label':sample['label']
               }

class FlipBase:

    def __call__(self,sample):

        p = np.random.random()
        if p > 0.5:
            return {'array': self.flipmethod(sample['array']),
                    'label' : sample['label']
                   }
        else:
            return sample


class FlipH(FlipBase):

    def __init__(self,):
        self.flipmethod = np.fliplr

class FlipV(FlipBase):

    def __init__(self,):
        self.flipmethod = np.flipud

class SpotDataset(Dataset):

    def __init__(self,
                 data_pths,
                 transform = None,
                 genes = None,
                 size = 1000,
                 n_classes = 5,
                 encoding = None,
                 arraysize = 8,
                ):

        self.data_pths = data_pths
        self.genelist = pd.Index(genes) if genes is not None else pd.Index([])
        self.genes = genes
        self.samples = []
        self.arraysize = arraysize
        self.size = size
        self.nclasses = n_classes

        if encoding is None and isinstance(encoding,dict):
            self.encoding = encoding
        else:
            self.encoding =  {'her2lum' : 0,
                              'her2nonlum' : 1,
                              'luma' : 2,
                              'lumb' : 3,
                              'tnbc' : 4
                             }

        self.decoding = { v:k for k,v in self.encoding.items() }

        self.transform = transform

        self._init()

    def __len__(self,):
        return len(self.samples)

    def __getitem__(self,idx):
        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return self.samples[idx]



    def _init(self,):

        count_pth = self.data_pths['count_data']
        label_pth = self.data_pths['label_data']


        count_pth.sort()
        label_pth.sort()

        self.nsamples = np.min((len(count_pth),self.size))

        unsrt = np.random.choice(np.arange(0,len(count_pth)),
                                 self.nsamples,
                                 replace = False
                                )

        count_pth = [ count_pth[x] for x in unsrt ]
        label_pth = [ label_pth[x] for x in unsrt ]

        for k,(pcnt, plbl) in enumerate(zip(count_pth,label_pth)):



            c = pd.read_csv(pcnt,
                            sep = '\t',
                            header = 0,
                            index_col = 0,
                            compression = 'gzip',
                           )
            rowSums = c.values.sum(axis = 1).reshape(-1,1)


            if self.genes is not None:
                tc = pd.DataFrame(np.zeros((c.shape[0],
                                            len(self.genelist))
                                           ),
                                 columns = self.genelist
                                 )

                inter = c.columns.intersection(self.genelist)
                tc.loc[:,inter] = c.loc[:,inter].values.astype(np.float32)

                tc = tc.values
                tc = np.divide(tc,
                               rowSums,
                               where = (rowSums != 0)
                              )


                tc = self.frame2tensor(tc)
            else:
                self.genelist = self.genelist.union(c.columns)
                tc = c

            with open(plbl,'r+') as lopen:
                l = lopen.readlines()[0]

            print(f"[{k +1 }/{self.nsamples}] : loaded {pcnt} | subtype {l}",
                  flush = True)

            l = self._one_hot(l)

            self.samples.append([tc,l])

        if self.genes is None:
            for ii in range(len(self.samples)):

                tc = pd.DataFrame(np.zeros((self.samples[ii][0].shape[0],
                                            len(self.genelist))
                                           ),
                                 columns = self.genelist
                                 )

                inter = self.samples[ii][0].columns.intersection(self.genelist)
                tc.loc[:,inter] = self.samples[ii][0].values

                tc = tc.values
                tc = np.divide(tc,
                               rowSums,
                               where = (rowSums != 0)
                              )

                self.samples[ii][0] = self.frame2tensor(tc)

        self.samples = [ {'array':x[0],
                          'label':x[1]} for x in self.samples]

        self.G = len(self.genelist)
        print(f"Assembled dataset of {len(self.samples)} arrays",
              flush = True)


    def _one_hot(self,x : str):

        return self.encoding[x]

    def frame2tensor(self,x):
        if not isinstance(x,np.ndarray):
            x = np.asarray(x)

        x = x.reshape(len(self.genelist),
                      self.arraysize,
                      self.arraysize,
                      ).astype(np.float32)

        return x


class CNN(t.nn.Module):

    def  __init__(self,
                  in_channels):

        super(CNN, self).__init__()

        self.in_channels = in_channels


        # Input 8x8xG --> Output 8x8x1024
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = 1024,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1,
                              )



        # Input 8x8x100 --> Output 6x6x1024
        self.pool = nn.AvgPool2d(kernel_size = 3,
                                 padding = 0,
                                 stride = 1,
                                )

        self.bn1 = nn.BatchNorm2d(num_features = 1024)


        # Input 6x6x1024--> Output 4x4x64

        self.conv2 = nn.Conv2d(in_channels = 1024,
                               out_channels = 64,
                               kernel_size  = 3,
                               padding = 0,
                               stride = 1,
                              )

        self.bn2 = nn.BatchNorm2d(num_features = 64)

        # Input 1024x1 -- > Output 128
        self.fc1 = nn.Linear(in_features = 1024,
                             out_features = 128,
                             bias = True,
                            )

        # Input 64x1 --> Output 5
        self.fc2 = nn.Linear(in_features = 128,
                             out_features = 5,
                             bias = True,
                            )

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.bn1(self.pool(x))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(-1,1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def test(net,
         test_set,
         nlabels = 5,
         num_workers = 1,
         device = None,
         ):

       if device is None:
            device = t.device('cpu')



       decoder = test_set.decoding

       test_loader = DataLoader(test_set,
                                num_workers = num_workers,
                                batch_size = len(test_set))

       class_correct = np.zeros(nlabels)
       class_total = np.zeros(nlabels)
       resmat = np.zeros((nlabels,nlabels))

       with t.no_grad():
            for data in test_loader:
                array, labels = data['array'].to(device), data['label'].to(device)
                outputs = net(array)
                _,pred = t.max(outputs,1)
                print(pred)
                c = (pred == labels).squeeze()

            for i in range(c.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                resmat[label,pred[i].item()] += 1

       resmat = pd.DataFrame(resmat,
                             columns = test_set.encoding.keys(),
                             index = test_set.encoding.keys()
                            )

       class_total[class_total == 0] = np.nan

       for i in range(nlabels):
           txtlabel = decoder[i]
           print(f'Accuracy of label {txtlabel} : {100 * class_correct[i] / class_total[i]}',
                 flush = True)

       return resmat



def train(net,
          train_set,
          val_set,
          batch_size,
          n_epochs,
          output_dir,
          lr = 0.001,
          device = None,
          num_workers = 1,
          ):

    if device is None:
        device = t.device('cpu')

    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              num_workers = num_workers,
                             )

    val_loader = DataLoader(val_set,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle = True,
                            )


    n_batches = len(train_loader)

    loss_fun = nn.CrossEntropyLoss()

    optim = t.optim.Adam(net.parameters(),
                         lr = lr,
                        )

    val_min = np.inf

    try:
        for epoch in range(n_epochs):
            net.train()
            total_loss = 0.0
            total_val_loss = 0.0


            for i, data in enumerate(train_loader):

                inputs, labels = data['array'], data['label']
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                optim.zero_grad()

                outputs = net(inputs)
                loss = loss_fun(outputs, labels)
                loss.backward()
                optim.step()

                total_loss += loss.item()

            print(f"Epoch : {epoch + 1:d} | train_loss : {total_loss}", flush = True)


            net.eval()
            for sample in val_loader:

                 inputs = Variable(sample['array']).to(device)
                 labels = Variable(sample['label']).to(device)
                 val_outputs = net(inputs)
                 val_loss = loss_fun(val_outputs, labels)
                 total_val_loss += val_loss.item()

            if total_val_loss < val_min:
                model_opth = osp.join(output_dir,
                                      '.'.join([TAG,
                                               'best.val.model.pt'
                                               ]
                                              )
                                     )

                t.save(net.state_dict(),
                       model_opth)

                val_min = total_val_loss


            print(f"Validation loss : {total_val_loss / len(val_loader) }", flush = True)

        print("Training Completed",
              flush = True)

    except KeyboardInterrupt:
        print(f"\nEarly Stopping by user")



def main(output_dir,
         device,
         p_train,
         data_pth,
         test_patients,
         samples,
         batch_size,
         epochs,
         learning_rate,
         num_workers,
         ):



    if output_dir is None:
        output_dir = os.getcwd()
    else:
        output_dir = output_dir

    if device.lower() == 'gpu' and t.cuda.is_available():
       device = t.device('cuda')
    else:
       device = t.device('cpu')


    print(f"Will be using Device : {str(device)}",flush = True)

    p_train = p_train

    if test_patients:
        test_patients = [ str(x) for x in test_patients ]
    else:
        test_patients = ["23287","23567","23268","23270","23209"]

    count_data = glob.glob(data_pth + '/count_data/*.tsv.gz')
    label_data = glob.glob(data_pth + '/label_data/*.txt')

    is_test = lambda x: osp.basename(x).split('.')[0] in test_patients
    is_train  = lambda x: is_test(x) == False

    train_label_data = list(filter(is_train,label_data))
    train_count_data = list(filter(is_train,count_data))

    train_pths = dict(count_data = train_count_data,
                      label_data = train_label_data)


    eval_label_data = list(filter(is_test,label_data))
    eval_count_data = list(filter(is_test,count_data))

    eval_pths = dict(count_data = eval_count_data,
                     label_data = eval_label_data)


    if samples is not None:
        nsamples = min(samples,
                       len(train_count_data))
    else:
        nsamples = len(train_count_data)

    print(f"Will be using {nsamples} for training")



    trf = transforms.Compose([RotateK90(),
                              FlipH(),
                              FlipV(),
                              ToTensor(),
                             ]
                            )


    dataset = SpotDataset(train_pths,
                          size = nsamples,
                          transform = trf)



    genefile = osp.join(output_dir,
                        '.'.join([TAG,
                                 'gene_set',
                                 'txt',
                                 ]
                                )
                       )


    with open(genefile,'w+') as gopen:
       gopen.writelines([g + '\n' for g in \
                         dataset.genelist.tolist()])


    len_train = int(len(dataset)*p_train)
    len_val = len(dataset) - len_train

    train_set, val_set = random_split(dataset,
                                      (len_train,len_val),
                                      )


    cnn_net = CNN(dataset.G)

    train(cnn_net,
          train_set = train_set,
          val_set = val_set,
          batch_size = batch_size,
          n_epochs = epochs,
          lr = learning_rate,
          output_dir = output_dir,
         )

    final_model_opth = osp.join(output_dir,
                              '.'.join([TAG,
                                       'final.model.pt'
                                       ]
                                      )
                                )

    t.save(cnn_net.state_dict(),
           final_model_opth)

    eval_dataset = SpotDataset(eval_pths,
                               size = len(eval_count_data),
                               genes = dataset.genelist,
                              )

    resmat = test(cnn_net,
                  eval_dataset,
                  num_workers = num_workers)

    respth = osp.join(output_dir,
                      '.'.join([TAG,
                                "pred.res",
                                "tsv"
                               ]
                              )
                     )

    resmat.to_csv(respth,
                  sep = '\t',
                  header = True,
                  index = True,
                 )


if __name__ == '__main__':

    date = str(datetime.datetime.now())
    TAG = re.sub(':|-|\\.| ','',date)

    prs = parser(date = date)
    args = prs.parse_args()

    main(**vars(args))
