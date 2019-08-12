#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def makeHeatmap(ax,
                mat,
                x_labels,
                y_labels):


    true_sums = mat.sum(axis = 1).astype(int)

    ax.imshow(mat / true_sums.reshape(-1,1),
             cmap = plt.cm.Greys
             )

    ax.grid(b = True,
            which = 'minor',
            axis = 'both',
            color = 'k',
            linewidth = 2,
           )

    ax.set_xticks(0.5 + np.arange(mat.shape[0]),
                  minor = True,
                 )

    ax.set_yticks(0.5 + np.arange(mat.shape[0]),
                  minor = True,
                 )


    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticks(np.arange(mat.shape[1]))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)




    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = str(mat[i,j].astype(int)) + '/' + str(true_sums[i])
            ax.text(j,i,
                    txt,
                    ha = 'center',
                    va = 'center',
                    color = 'r',
                    fontsize = 20,
                    fontfamily = 'monospace',
                    fontweight = 'bold'
                   )

if __name__ == '__main__':

    resmat = (np.random.random((5,5)) * 10 ).round(0)
    print(resmat)
    labels = ['her2lum','her2nonlum','luma','lumb','tnbc']

    fig, ax = plt.subplots()
    makeHeatmap(ax, resmat, labels,labels)
    plt.show()
