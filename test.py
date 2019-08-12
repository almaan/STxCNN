#!/usr/bin/env python3


import numpy as np
import pandas as pd

import torch as t
from nn import SpotDataset, CNN, test

import argparse as arp

if __name__ == "__main__":

    prs = arp.ArgumentParser()

    prs.add_argument("-m","--model",
                     type = str,
                     required = True,
                     help "",
                    )

    prs.add_arguments("-g","--gene_list",
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

    args = prs.parse_args()

    if args.patient_list is None:
        patients = ["23287","23567","23268","23270","23209"]
    else:
        patients = args.patient_list


    model_pth = args.model

    with open(args.gene_list,'w+') as gopen:
        genelist = list(map(lambda x: x.strip('\n'),
                            gopen.readlines()))

        genelist = pd.Index(genelist)

    model = t.load(model_pth)

    cnn_net = CNN(model['conv1.weight'].shape[1])

    cnn_net.load_state_dict(model)

    eval_dataset = SpotDataset(eval_pths,
                               size = len(eval_count_data),
                               genes = genelist,
                              )

    test(cnn_net,
         eval_dataset,
         num_workers = args.num_workers)


