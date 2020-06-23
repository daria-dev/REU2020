import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp

from make_data import spiral, initial
from model import MLP


parser = argparse.ArgumentParser('DATA')
parser.add_argument('--T0', type=float, default=0,
                    help='initial time for trajectory')
parser.add_argument('--Tend', type=float, default=5,
                    help='final time for trajectory')
parser.add_argument('--num_point', type=int, default=502,
                    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=10,
                    help='number of testing trajectories')
parser.add_argument('--log_dir', type=str, default='results',
                    help='name for directory in which to save results')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of units per hidden layer')
args = parser.parse_args()


def solve_ODE(t, y0, func):
    '''
    NOTES: A cleaner function for solve_ivp (i.e. adjust the many solve_ivp parameters
            here rather than at every occurance where it is used).

    INPUT:
        t = vector where elements are times at which to store solution (t subset of [args.T0,args.Tend])
        y0 = initial position data

    OUTPUT:
        return #0 = solution of ODE at times 't' starting at 'y0'
    '''
    assert (len(y0.shape) == 1), 'y0 must be a 1D array.'

    sol = solve_ivp(fun=func, t_span=[args.T0, args.Tend],
                    y0=y0, method='RK45', t_eval=t,
                    dense_output=False, vectorized=False,
                    rtol=1e-9, atol=1e-9 * np.ones((len(y0),)))

    return np.transpose(sol.y)


def generate_trajectories(t, y0_l, func, net_func):
    data_y = []
    data_y_ = []

    for y0 in y0_l:
        y = solve_ODE(t, y0, func)  # generates ground truth trajectory
        y_ = solve_ODE(t, y0, net_func)
        # generates predicted trajectory (where net_func is the neural network)
        data_y.append(y)
        data_y_.append(y_)

    data_y = np.stack(data_y, axis=0)
    data_y_ = np.stack(data_y_, axis=0)

    return data_y, data_y_


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
    plt.show()
    plt.close()


def test_loss(net, criterion, y0_l):
    '''
    INPUT:
        net = trained neural network
        criterion = a function that takes in two y arrays and calculates loss

    OUTPUT:
        test loss according to criterion
    '''
    # wrapper function for neural network
    def net_func(t, y):
        v = net.forward(torch.tensor(y, dtype=torch.float32)).detach().numpy()
        return v

    t = np.linspace(start=args.T0, stop=args.Tend, num=args.num_point)  # define grid on which to solve ODE
    func = spiral

    data_y, data_y_ = generate_trajectories(t, y0_l, func, net_func)
    plot_3D_pred(data_y, data_y_)  # visualize solution

    return criterion(data_y, data_y_)


def main():
    dim = 3
    net = MLP(input_dim=dim, hidden_dim=args.hidden_dim, output_dim=dim)
    net.load_state_dict(torch.load(args.log_dir + '/net_state_dict.pt'), strict=False)  # loads trained model
    y0 = np.array([4, 3, -2])  # define center of neighborhood for initial starting points

    y0_l = [initial(y0) for _ in range(args.num_traj)]  # generate random starting points

    def criterion(y, y_):
        return np.mean((y - y_) ** 2)

    print('test loss:', test_loss(net, criterion, y0_l))


main()
