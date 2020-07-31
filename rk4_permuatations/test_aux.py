import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


def solve_ODE_nn(t, y0, net_func):
    y = []
    y_cur = y0
    for i in t:
        y_cur = net_func(None, y_cur)

    return y


def generate_trajectories(y0, net_func, steps):
    y = np.zeros([y0.shape[0], steps, y0.shape[1]])
    y_cur = y0

    for i in range(steps):
        y[:, i, :] = y_cur
        y_cur = net_func(y_cur)

    return y


def plot_3D_pred(data, data_):
    '''
    A slightly modified version of plot_3D that takes in two data arrays instead of one.

    INPUT:
        data = data position points; 3D array with axes
                - 0 = ith trajectory
                - 1 = point at time t_i
                - 2 = spatial dimension y_i
        data_ = predicted data position points; same format as above

    OUTPUT:
        None
    '''
    assert (data.shape == data_.shape)
    assert (len(data.shape) == 3), 'data must be 3D array.'
    assert (data.shape[2] == 3), 'data must be 3D.'

    plt.close()
    tot_num_traj = data.shape[0]
    fig = plt.figure()
    for traj in range(tot_num_traj):
        ax = fig.gca(projection='3d')
        ax.plot(data[traj, :, 0], data[traj, :, 1], data[traj, :, 2],
                lw=0.5, color='b', label='ground truth')
        ax.plot(data_[traj, :, 0], data_[traj, :, 1], data_[traj, :, 2],
                lw=0.5, color='r', linestyle='--', label='predictions')
        ax.view_init(elev=26, azim=-133)
    ax.set_xlabel('y_1')
    ax.set_ylabel('y_2')
    ax.set_zlabel('y_3')
    ax.set_title('ground truth & predictions')
    plt.savefig('results/test_vs_pred.png')
    plt.show()
    plt.close()


def test_loss(net, criterion, test_y):
    '''
    INPUT:
        net = trained neural network
        criterion = a function that takes in two y arrays and calculates loss

    OUTPUT:
        test loss according to criterion
    '''
    # wrapper function for neural network
    def net_func(y):
        y1 = net.forward(torch.tensor(y, dtype=torch.float32)).detach().numpy()
        return y1

    test_y_ = generate_trajectories(test_y[:, 0, :], net_func, steps=test_y.shape[1])

    if test_y.shape[2] == 3:
        plot_3D_pred(test_y, test_y_)  # visualize solution

    return criterion(test_y, test_y_)
