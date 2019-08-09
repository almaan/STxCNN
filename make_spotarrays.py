#!/usr/bin/env python3

import numpy as np
import pandas as pd

import os.path as osp
import glob

import QuickST.utils as ut
from QuickST.data.STsection import STsection

import matplotlib.pyplot as plt

debug = False
cpths = glob.glob("/home/alma/ST-2018/CNNp/DGE/data/mapped_count_data/*/*.tsv")
mpths = glob.glob("/home/alma/ST-2018/CNNp/DGE/data/curated_feature_files/*/*.tsv")

cpths.sort()
mpths.sort()

side = 8

main_pth = "/home/alma/ST-2018/BC_Stanford/data/arrays"

for section in range(len(cpths)):

    tmpST = STsection(cpths[section],
                      mpths[section]
                     )
    sb = str(tmpST.meta['subtype'][0])
    pt = str(tmpST.meta['patient'][0])
    rp = str(tmpST.meta['replicate'][0])

    xcrd =  tmpST.x.round(0).astype(int)
    ycrd = tmpST.y.round(0).astype(int)

    mat = np.zeros((50,50))
    imat = np.ones((50,50)) * -1
    mat[ycrd,xcrd] = 1

    fullarray = []

    #DEBUG
    if debug:
        plt.imshow(mat,
                   aspect = 'auto')
        plt.show()

    for k,(x,y) in enumerate(zip(xcrd,ycrd)):
        imat[y,x] = k
        print(f"Spot : {k} | (x,y) = ({x},{y})")
        fills = mat[y:(y+side),x:(x+side)]
        nn = int(fills.sum())
        print(f"Number of spots: {nn} / {side**2} \n")

        if nn == int(side**2):
           fullarray.append(k)
           print("Has full array")

    for k,arr in enumerate(fullarray):
        idx = imat[ycrd[arr]:(ycrd[arr] + side),xcrd[arr]:(xcrd[arr] + side)]
        idx = idx.ravel()

        ct = tmpST.cnt.iloc[idx,:]
        mt = tmpST.meta.iloc[idx,:]


        uid = '.'.join([pt,rp,
                        str(xcrd[arr]) + 'x' + str(ycrd[arr])]
                      )

        print(f"Saving array >> {uid}")

        cntpth = osp.join(main_pth,
                          'count_data',
                          '.'.join([uid,'tsv'])
                         )
        mtapth = osp.join(main_pth,
                          'meta_data',
                          '.'.join([uid,'tsv'])
                         )

        lblpth = osp.join(main_pth,
                         'label_data',
                         '.'.join([uid,'txt'])
                        )


        ct.to_csv(cntpth,
                  sep = '\t',
                  header = True,
                  index = True,
                  compression = 'gzip',
                 )

        mt.to_csv(mtapth,
                  sep = '\t',
                  header = True,
                  index = True,
                  compression = 'gzip',
                 )

        with open(lblpth,'w+') as lopen:
             lopen.write(sb)










