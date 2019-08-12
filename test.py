#!/usr/bin/env python3


import numpy as np
import pandas as pd

import torch as t
from nn import SpotDataset, CNN, test

import argparse as arp
import os.path as osp
import glob

if __name__ == "__main__":

    prs = arp.ArgumentParser()

    prs.add_argument("-m","--model",
                     type = str,
                     required = True,
                     help =  "",
                    )

    prs.add_argument("-g","--gene_list",
                      type = str,
                      required = True,
                      help = "",
                     )

    prs.add_argument("-o", "--output_dir",
                     type = str,
                     required = False,
                     default = None,
                     help = "",
                    )

    prs.add_argument("-nw", "--num_workers",
                     type = int,
                     required = False,
                     default = 1,
                     help = "",
                    )

    prs.add_argument("-p","--patient_list",
                     type = str,
                     nargs = '+',
                     required = False,
                     default = None,
                     help = "",
                    )

    prs.add_argument("-d","--data_path",
                     type = str,
                     required = False,
                     default = None,
                     help = "",
                     )


    args = prs.parse_args()

    if args.patient_list is None:
        patients = ["23287","23567","23268","23270","23209"]
    else:
        patients = args.patient_list


    model_pth = args.model

    with open(args.gene_list,'r+') as gopen:
        genelist = list(map(lambda x: x.strip('\n'),
                            gopen.readlines()))

        genelist = pd.Index(genelist)


    model = t.load(model_pth)

    cnn_net = CNN(model['conv1.weight'].shape[1])

    cnn_net.load_state_dict(model)


    count_data = glob.glob(args.data_path + '/count_data/*.tsv.gz')
    label_data = glob.glob(args.data_path + '/label_data/*.txt')

    is_test = lambda x: osp.basename(x).split('.')[0] in patients

    count_data = list(filter(is_test,count_data))
    label_data = list(filter(is_test,label_data))

    eval_pths = dict(count_data = count_data,
                     label_data = label_data)


    eval_dataset = SpotDataset(eval_pths,
                               size = len(count_data),
                               genes = genelist,
                              )

    test(cnn_net,
         eval_dataset,
         num_workers = args.num_workers)


