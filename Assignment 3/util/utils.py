import ast
import operator as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import random

import os.path as ops
import time

import loguru


def get_logger(log_file_name_prefix):
    """
    :param log_file_name_prefix: log文件名前缀
    :return:
    """
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file_name = '{:s}_{:s}.log'.format(log_file_name_prefix, start_time)
    log_file_path = ops.join("log_files",log_file_name)

    logger = loguru.logger
    # log_level = 'INFO'
    # if CFG.LOG.LEVEL == "DEBUG":
    #     log_level = 'DEBUG'
    # elif CFG.LOG.LEVEL == "WARNING":
    #     log_level = 'WARNING'
    # elif CFG.LOG.LEVEL == "ERROR":
    #     log_level = 'ERROR'

    logger.add(
        log_file_path,
        format="{time} {level} {message}",
        retention="10 days",
        rotation="1 week"
    )

    return logger

def plot_results(paths, lb, ub, x_field, y_field, result_path):
    fig, ax = plt.subplots()
    x,y = np.meshgrid(np.linspace(lb,ub,25),np.linspace(lb,ub,25))
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
    plt.savefig(result_path + '/plot.png')
    # plt.show()