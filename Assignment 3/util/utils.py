import ast
import operator as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import random


def plot_results(paths, lb, ub, x_field, y_field):
    fig, ax = plt.subplots()
    x,y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
    u = eval(x_field)
    v = eval(y_field)
    number_of_colors = len(paths)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    ax.quiver(x,y,u,v)
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    for i, path in enumerate(paths):
        string_path = mpath.Path(path)
        patch = mpatches.PathPatch(string_path, facecolor="none", edgecolor=color[i], lw=2)
        plt.scatter(path[0][0], path[0][1], marker = 'o', color = color[i])
        ax.add_patch(patch)
    plt.savefig('results/plot.png')
    plt.show()