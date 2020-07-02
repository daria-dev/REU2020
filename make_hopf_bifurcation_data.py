import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_aux import make_directory

parser = argparse.ArgumentParser('DATA')
parser.add_argument('--T0', type=float, default=0,
    help='initial time for trajectory')
parser.add_argument('--Tend', type=float, default=10,
    help='final time for trajectory')
parser.add_argument('--Delta', type=float, default=1,
    help='radius of neighborhood around initial point y0')
parser.add_argument('--num_point', type=int, default=302,
    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=50,
    help='number of training trajectories')
parser.add_argument('--data_dir', type=str, default='data_hopf',
        help='name for data directory')
args = parser.parse_args()

# ODE System
def hopf_bifurcation(t, y):
    '''
    Input:
        t = dummy time variable
        mu = mu parameter
        y =  position data
    Output:
        velocity data
    '''
    mu = y[0]
    x = y[1]
    y = y[2]

    v = np.array([0, x*mu - y - x*(x**2+y**2), -x + mu*y - y*(x**2+y**2)])
    
    return v

def initial(y0):
    '''
    NOTES: Creates a random initial starting point close to the given point y0.

    INPUT:
        y0 = initial position data

    OUTPUT:
        return #0 = nearby initial position data.
    '''
    assert (len(y0.shape) == 1),'y0 must be a 1D array.'

    return np.array(y0)+np.random.uniform(low=-args.Delta, high=args.Delta, size=(len(y0),))

def solve_ODE(t,y0,func):
    '''
    NOTES: A cleaner function for solve_ivp (i.e. adjust the many solve_ivp parameters
            here rather than at every occurance where it is used).

    INPUT:
        t = vector where elements are times at which to store solution (t subset of [args.T0,args.Tend])
        y0 = initial position data

    OUTPUT:
        return #0 = solution of ODE at times 't' starting at 'y0'
    '''
    assert (len(y0.shape) == 1),'y0 must be a 1D array.'
    sol = solve_ivp(fun=func, t_span=[args.T0, args.Tend],
        y0=initial(y0), method='RK45', t_eval=t,
        dense_output=False, vectorized=False,
        rtol=1e-9, atol=1e-9*np.ones((len(y0),)))

    return np.transpose(sol.y)

def plot_3D(data,func_name,dir_name):
    '''
    NOTES: Plots 3D data and saves plot as png.

    INPUT:
        data = data position points; 3D array with axes
                - 0 = ith trajectory
                - 1 = point at time t_i
                - 2 = spatial dimension y_i
        func_name = string with name of ODE
        dir_name = str; name of directory to save plot
        data_type = string with label for data (e.g. training, validation, testing)

    OUTPUT:
        None
    '''
    assert (len(data.shape) == 3),'data must be 3D array.'
    assert (data.shape[2] == 3),'data must be 3D.'
    assert (type(func_name) == str),'func_name must be string.'
    assert (type(dir_name) == str),'dir_name must be string.'

    plt.close()
    tot_num_traj = data.shape[0]
    fig = plt.figure()
    for traj in range(tot_num_traj):
        ax = fig.gca(projection='3d')
        ax.plot(data[traj,:,0],data[traj,:,1],data[traj,:,2])
    ax.set_xlabel('mu')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title(func_name+': data')
    plt.autoscale()
    # plt.savefig(args.data_dir+'/spiral_'+('train' if train else 'test')+'_data.svg')
    plt.savefig(dir_name+'/'+func_name+'_data.png')
    plt.show()
    plt.close()


def generate_data(t,y0,func,func_name,num_traj):
    '''
    NOTES: Generates multiple data trajectories (number specified by num_traj) starting near y0.
            Saves this data as 3D array with axes
                - 0 = ith trajectory
                - 1 = point at time t_i
                - 2 = spatial dimension y_i

    INPUT:
        t = vector where elements are times at which to store solution (t subset of [args.T0,args.Tend], containing endpoints)
        y0 = initial position data
        func = function defining ODE used to generate data
        func_name = string with name of ODE
        data_type = string with label for data (e.g. training, validation, testing)
        num_traj = number of trajectories to generate

    OUTPUT:
        None
    '''
    assert (len(t.shape) == 1),'t must be a 1D array.'
    assert ((args.T0 <= min(t)) & (max(t) <= args.Tend)),'t must be subset of [T0,Tend].'
    assert (len(y0.shape) == 1),'y0 must be 1D array.'
    assert (type(func_name) == str),'func_name must be string.'
    assert (type(num_traj) == int),'num_traj must be integer.'

    data_y = []
    for _ in range(num_traj):
        y = solve_ODE(t,y0,func)
        data_y.append(y)
    data_y = np.stack(data_y, axis=0)

    # save data
    np.save(args.data_dir+'/data_y', data_y)
    plot_3D(data_y,func_name,args.data_dir)

if __name__ == "__main__":
    ''' Make directory to store data. '''
    make_directory(args.data_dir) # make directory to store data

    ''' Initialize ODE input. '''
    t = np.linspace(start=args.T0, stop=args.Tend, num=args.num_point) # define grid on which to solve ODE
    y0 = np.array([-0.15,2,0]) # define center of neighborhood for initial starting points

    generate_data(t=t,y0=y0,func=hopf_bifurcation,func_name='Hopf',num_traj=args.num_traj)