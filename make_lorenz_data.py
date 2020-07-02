# Code by: Kayla Bollinger
# Date: 06/17/20
# Email: kbolling@andrew.cmu.edu

import argparse, json
from model_aux import make_directory, plot_3D
import numpy as np
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser('DATA')
parser.add_argument('--T0', type=float, default=0,
    help='initial time for trajectory')
parser.add_argument('--Tend', type=float, default=5,
    help='final time for trajectory')
parser.add_argument('--Delta', type=float, default=2,
    help='radius of neighborhood around initial point y0')
parser.add_argument('--num_point', type=int, default=100,
    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=100,
    help='number of training trajectories')
parser.add_argument('--data_dir', type=str, default='data',
        help='name for data directory')
args = parser.parse_args()

# ODE system
def lorenz(t, y): 
    '''
    NOTES: Defines an ODE system that generates a lorenz system.
            Input y is either a 1D array (used mainly with built in ODE solvers), or 3D array
            (to generate velocity data over multiple trajectories for training).

    INPUT:
        t = dummy variable (system is autonomous, but python solvers require time input)
        y = position data; 1D array, or 3D array with axes
            - 0 = ith trajectory
            - 1 = point at time t_i
            - 2 = spatial dimension y_i

    OUTPUT:
        return #0 = velocity data
    '''
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'


    # Original
    if len(y.shape) == 1:
        v = np.array([10*(y[1]-y[0]), y[0]*(28-y[2])-y[1], y[0]*y[1]-8/3.*y[2]])
    else:
        v = np.zeros(y.shape)
        v[:,:,0] = 10*(y[:,:,1] - y[:,:,0])
        v[:,:,1] = y[:,:,0]*(28 - y[:,:,2]) - y[:,:,1]
        v[:,:,2] = y[:,:,0]*y[:,:,1] - 8/3. * y[:,:,2]
    return v

def fwd_euler(t, y): 
    '''
    NOTES: Calculates velocity using the Forward Euler Approximation
            Input y is either a 1D array (used mainly with built in ODE solvers), or 3D array
            (to generate velocity data over multiple trajectories for training).

    INPUT:
        t = dummy variable (system is autonomous, but python solvers require time input)
        y = position data; 1D array, or 3D array with axes
            - 0 = ith trajectory
            - 1 = point at time t_i
            - 2 = spatial dimension y_i

    OUTPUT:
        return #0 = velocity data
    '''

    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        dt = (args.Tend - args.T0)/len(y)
        end = len(y)
        v = np.array([ (y[2:end]-y[1:end-1])/dt ])
    else:
        dt = (args.Tend - args.T0)/len(y[0,:])
        end = y.shape[1]
        v = np.zeros((y.shape[0],y.shape[1]-2,y.shape[2]))
        v[:,:,:] = (y[:,2:end,:]-y[:,1:end-1,:])/dt
    return v

def central_diff(t,y):
    '''
    NOTES: Calculates velocity using the Central Difference Method
            Input y is either a 1D array (used mainly with built in ODE solvers), or 3D array
            (to generate velocity data over multiple trajectories for training).

    INPUT:
        t = dummy variable (system is autonomous, but python solvers require time input)
        y = position data; 1D array, or 3D array with axes
            - 0 = ith trajectory
            - 1 = point at time t_i
            - 2 = spatial dimension y_i

    OUTPUT:
        return #0 = velocity data
    '''

    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        dt = (args.Tend - args.T0)/len(y)
        end = len(y)
        v = np.array([ (y[2:end]-y[0:end-2])/(2*dt) ])
    else:
        dt = (args.Tend - args.T0)/len(y[0,:])
        end = y.shape[1]
        v = np.zeros((y.shape[0],y.shape[1]-2,y.shape[2]))
        v[:,:,:] = (y[:,2:end,:]-y[:,0:end-2,:])/(2*dt)
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
        rtol=1e-12, atol=1e-12*np.ones((len(y0),)))

    return np.transpose(sol.y)


def generate_data(t,y0,func,func_name,data_type,num_traj):
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
    assert (type(data_type) == str),'data_type must be string.'
    assert (type(num_traj) == int),'num_traj must be integer.'

    data_y = []
    for _ in range(num_traj):
        y = solve_ODE(t,y0,func)
        data_y.append(y)
    data_y = np.stack(data_y, axis=0)
    data_v = func(t=None,y=data_y) # exact velocity calculated
    #data_v = fwd_euler(t=None,y=data_y) # velocity calculated using forward euler
    #data_v = central_diff(t=None,y=data_y) #velocity calculated using central difference

    # uncomment this part if using fwd_euler or central_diff
    # if len(data_y.shape) == 1:
    #     data_y = data_y[1:-1]
    # else:
    #     data_y = data_y[:, 1:-1, :]

    if data_y.shape[2] == 3 :
        plot_3D(data_y,func_name,args.data_dir,data_type) # visualize solution

    # save data
    np.save(args.data_dir+'/'+data_type+'_y', data_y)
    np.save(args.data_dir+'/'+data_type+'_v', data_v)

if __name__ == "__main__":
    ''' Make directory to store data. '''
    make_directory(args.data_dir) # make directory to store data

    ''' Initialize ODE input. '''
    t = np.linspace(start=args.T0, stop=args.Tend, num=args.num_point) # define grid on which to solve ODE
    y0 = np.array([-8, 7, 27]) # define center of neighborhood for initial starting points

    ''' Save parameters for reference. '''
    # save args
    with open(args.data_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    np.save(args.data_dir+'/grid',t) # save grid
    np.save(args.data_dir+'/ipcenter',y0) # save center

    ''' Generate data. '''
    generate_data(t=t,y0=y0,func=lorenz,func_name='Lorenz',data_type='train',num_traj=args.num_traj)
    generate_data(t=t,y0=y0,func=lorenz,func_name='Lorenz',data_type='val',num_traj=int(args.num_traj*0.2))
    generate_data(t=t,y0=y0,func=lorenz,func_name='Lorenz',data_type='test',num_traj=int(args.num_traj*0.2))
