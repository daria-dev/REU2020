import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from make_data import solve_ODE, spiral
from model_aux import make_directory
from model import MLP


parser = argparse.ArgumentParser('DATA')
parser.add_argument('--T0', type=float, default=0,
                    help='initial time for trajectory')
parser.add_argument('--Tend', type=float, default=5,
                    help='final time for trajectory')
parser.add_argument('--num_point', type=int, default=128,
                    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=1,
                    help='number of training trajectories')
parser.add_argument('--data_dir', type=str, default='data',
                    help='name for data directory')
parser.add_argument('--log_dir', type=str, default='results',
                    help='name for directory in which to save results')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of units per hidden layer')
args = parser.parse_args()


def generate_predictions(t, y0, func, net_func):
    data_y = []
    for _ in range(args.num_traj):
        y = solve_ODE(t, y0, func)  # generates ground truth trajectory
        data_y.append(y)
    data_y = np.stack(data_y, axis=0)

    data_y_ = []
    for _ in range(args.num_traj):
        y_ = solve_ODE(t, y0, net_func)  
        # generates predicted trajectory (where net_func is the neural network)
        data_y_.append(y_)
    data_y_ = np.stack(data_y_, axis=0)

    return data_y, data_y_


# Basically a modified version of plot_3D that takes in two data arrays instead of one
def plot_3D_pred(data, data_, func_name, data_type):
    '''
    NOTES: Plots 3D data and saves plot as png.

    INPUT:
        data = data position points; 3D array with axes
                - 0 = ith trajectory
                - 1 = point at time t_i
                - 2 = spatial dimension y_i
        data_ = predicted data position points; same format as above
        func_name = string with name of ODE
        dir_name = str; name of directory to save plot
        data_type = string with label for data (e.g. training, validation, testing)

    OUTPUT:
        None
    '''
    assert (data.shape == data_.shape)
    assert (len(data.shape) == 3), 'data must be 3D array.'
    assert (data.shape[2] == 3), 'data must be 3D.'
    assert (type(func_name) == str), 'func_name must be string.'
    assert (type(data_type) == str), 'data_type must be string.'

    plt.close()
    tot_num_traj = data.shape[0]
    fig = plt.figure()
    for traj in range(tot_num_traj):
        ax = fig.gca(projection='3d')
        ax.plot(data[traj, :, 0], data[traj, :, 1], data[traj, :, 2], lw=0.5, color='b')
        ax.plot(data_[traj, :, 0], data_[traj, :, 1], data_[traj, :, 2], lw=0.5, color='r', linestyle='--')
        ax.view_init(elev=26, azim=-133)
    ax.set_xlabel('y_1')
    ax.set_ylabel('y_2')
    ax.set_zlabel('y_3')
    ax.set_title(func_name + ': ' + data_type + ' data')
    plt.show()
    plt.close()


def main():
    func = spiral
    func_name = 'Spiral'
    make_directory(args.data_dir)

    test_y = np.load(args.data_dir + '/test_y.npy')

    t = np.linspace(start=args.T0, stop=args.Tend, num=args.num_point)  # define grid on which to solve ODE
    y0 = np.array([4, 3, -2])  # define center of neighborhood for initial starting points
    dim = test_y.shape[2]
    net = MLP(input_dim=dim, hidden_dim=args.hidden_dim, output_dim=dim)
    net.load_state_dict(torch.load(args.log_dir + '/net_state_dict.pt'), strict=False)  # loads trained model

    # wrapper function for neural network
    def net_func(t, y):
        v = net.forward(torch.tensor(y, dtype=torch.float32)).detach().numpy()
        return v

    data_y, data_y_ = generate_predictions(t, y0, func, net_func)

    if data_y.shape[2] == 3:
        plot_3D_pred(data_y, data_y_, func_name, 'predictions & ground truth')  # visualize solution


main()
