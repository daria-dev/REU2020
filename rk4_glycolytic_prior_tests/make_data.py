import argparse, json
from model_aux import make_directory, plot_3D
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('DATA')
parser.add_argument('--T0', type=float, default=0,
    help='initial time for trajectory')
parser.add_argument('--Tend', type=float, default=5,
    help='final time for trajectory')
parser.add_argument('--Delta', type=float, default=2,
    help='radius of neighborhood around initial point y0')
parser.add_argument('--num_point', type=int, default=502,
    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=100,
    help='number of training trajectories')
parser.add_argument('--data_dir', type=str, default='data',
        help='name for data directory')

parser.add_argument('--ODE_system', type=str, default='Glycolytic',
    help='ODE System used to generate data e.g. Spiral, Lorenz, Hopf, Glycolytic')
parser.add_argument('--split_method', type=int, default=2,
    help='method to split data into train/val/test, either 1 or 2')
parser.add_argument('--T_ss', type=int, default = 250,
    help='number of points in training set')
parser.add_argument('--V_ss', type=int,default = 75,
    help='number of trajectories in validation set')
parser.add_argument('--noise',type=float,default=None,
        help='percent of noise to add as a float, else None')
args = parser.parse_args()

# Spiral ODE system
def spiral(t, y): 
    '''
    NOTES: Defines an ODE system that generates a spiral/corkscrew shape.
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
        v = np.array([y[1]**3, -( (y[0]+y[2])/2)**3 -0.2*y[1], y[2]/10.])
    else:
        v = np.zeros(y.shape)
        v[:,:,0] = y[:,:,1]**3
        v[:,:,1] = -((y[:,:,0]+y[:,:,2])/2.)**3 - 0.2*y[:,:,1]
        v[:,:,2] = y[:,:,2]/10.
    return v

# Lorenz ODE
def lorenz(t, y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        v = np.array([10*(y[1]-y[0]), y[0]*(28-y[2])-y[1], y[0]*y[1]-8/3.*y[2]])
    else:
        v = np.zeros(y.shape)
        v[:,:,0] = 10*(y[:,:,1] - y[:,:,0])
        v[:,:,1] = y[:,:,0]*(28 - y[:,:,2]) - y[:,:,1]
        v[:,:,2] = y[:,:,0]*y[:,:,1] - 8/3. * y[:,:,2]

    return v

# Hopf Bifurication ODE System
def hopf_bifurcation(t, y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        v = np.array([0, y[0]*y[1] + y[2] - y[1]*(y[1]**2 + y[2]**2), 
                        y[0]*y[2] - y[1] - y[2]*(y[1]**2 + y[2]**2)])
    else:
        v = np.zeros(y.shape)
        v[:,:,0] = np.zeros((v.shape[0], v.shape[1]))
        v[:,:,1] = y[:,:,0]*y[:,:,1] + y[:,:,2] - y[:,:,1]*(y[:,:,1]**2 + y[:,:,2]**2)
        v[:,:,2] = y[:,:,0]*y[:,:,2] - y[:,:,1] - y[:,:,2]*(y[:,:,1]**2 + y[:,:,2]**2)
    return v
    
# Glycolytic ODE System
def glycolytic_oscillator(t, y):
    
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)), 'y must be a 1D or 3D array.'

    # code partially taken from https://github.com/maziarraissi/MultistepNNs/blob/master/Glycolytic.py

    J0 = 2.5
    k1 = 100.0
    k2 = 6.0
    k3 = 16.0
    k4 = 100.0
    k5 = 1.28
    k6 = 12.0
    k = 1.8
    kappa = 13.0
    q = 4
    K1 = 0.52
    psi = 0.1
    N = 1.0
    A = 4.0

    if len(y.shape) == 1:
        v1 = J0 - (k1 * y[0] * y[5]) / (1 + (y[5] / K1) ** q)
        v2 = 2 * (k1 * y[0] * y[5]) / (1 + (y[5] / K1) ** q) - k2 * y[1] * (N - y[4]) - k6 * y[1] * y[4]
        v3 = k2 * y[1] * (N - y[4]) - k3 * y[2] * (A - y[5])
        v4 = k3 * y[2] * (A - y[5]) - k4 * y[3] * y[4] - kappa * (y[3] - y[6])
        v5 = k2 * y[1] * (N - y[4]) - k4 * y[3] * y[4] - k6 * y[1] * y[4]
        v6 = -2 * (k1 * y[0] * y[5]) / (1 + (y[5] / K1) ** q) + 2 * k3 * y[2] * (A - y[5]) - k5 * y[5]
        v7 = psi * kappa * (y[3] - y[6]) - k * y[6]
        v = np.array([v1, v2, v3, v4, v5, v6, v7])
    else:
        v = np.zeros(y.shape)
        v[:, :, 0] = J0 - (k1 * y[:, :, 0] * y[:, :, 5]) / (1 + (y[:, :, 5] / K1) ** q)
        v[:, :, 1] = (k1 * y[:, :, 0] * y[:, :, 5]) / (1 + (y[:, :, 5] / K1) ** q) - k2 * y[:, :, 1] * \
                     (N - y[:, :, 4]) - k6 * y[:, :, 1] * y[:, :, 4]
        v[:, :, 2] = k2 * y[:, :, 1] * (N - y[:, :, 4]) - k3 * y[:, :, 2] * (A - y[:, :, 5])
        v[:, :, 3] = k3 * y[:, :, 2] * (A - y[:, :, 5]) - k4 * y[:, :, 3] * y[:, :, 4] - kappa * \
                     (y[:, :, 3] - y[:, :, 6])
        v[:, :, 4] = k2 * y[:, :, 1] * (N - y[:, :, 4]) - k4 * y[:, :, 3] * y[:, :, 4] - k6 * y[:, :, 1] * y[:, :, 4]
        v[:, :, 5] = -2 * (k1 * y[:, :, 0] * y[:, :, 5]) / (1 + (y[:, :, 5] / K1) ** q) + 2 * k3 * y[:, :, 2] * \
                     (A - y[:, :, 5]) - k5 * y[:, :, 5]
        v[:, :, 6] = psi * kappa * (y[:, :, 3] - y[:, :, 6]) - k * y[:, :, 6]

    return v

def initial(y0):
    '''
    NOTES: Creates a random initial starting point close to the given point y0.

    INPUT:
        y0 = initial position data

    OUTPUT:
        return #0 = nearby initial position data.
    '''
    
    if args.ODE_system == 'Glycolytic':
        # code partially taken from https://github.com/maziarraissi/MultistepNNs/blob/master/Glycolytic.py
        S1 = np.random.uniform(0.15, 1.60, 1)
        S2 = np.random.uniform(0.19, 2.16, 1)
        S3 = np.random.uniform(0.04, 0.20, 1)
        S4 = np.random.uniform(0.10, 0.35, 1)
        S5 = np.random.uniform(0.08, 0.30, 1)
        S6 = np.random.uniform(0.14, 2.67, 1)
        S7 = np.random.uniform(0.05, 0.10, 1)

        # initial condition
        y0 = np.array([S1, S2, S3, S4, S5, S6, S7]).flatten()
        return y0

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
        num_traj = number of trajectories to generate

    OUTPUT:
        None
    '''
    assert (len(t.shape) == 1),'t must be a 1D array.'
    assert ((args.T0 <= min(t)) & (max(t) <= args.Tend)),'t must be subset of [T0,Tend].'
    assert (type(func_name) == str),'func_name must be string.'
    assert (type(num_traj) == int),'num_traj must be integer.'

    data_y = []
    for _ in range(num_traj):
        if args.ODE_system == 'Glycolytic': y0 = initial(y0)
        y = solve_ODE(t,y0,func)
        data_y.append(y)
    data_y = np.stack(data_y, axis=0)
    
    # add noise
    if args.noise != None:
        tot_num_traj = data_y.shape[0]
        for traj in range(tot_num_traj):
            y_vals = data_y[traj,:,2]
            noiseSigma = args.noise * y_vals;
            mu,sigma = 0,1
            noise  = noiseSigma*np.random.normal(mu, sigma, args.num_point)
            data_y[traj,:,2] += noise

    train_y = data_y[:,:args.T_ss,:]
    val_y = data_y[:args.V_ss,args.T_ss:,:]
    test_y = data_y[args.V_ss:, args.T_ss:, :]

    reg_v = fwd_euler(t=None,y=data_y[:,:args.T_ss,:])
    reg_y = train_y
    if len(reg_y.shape) == 1:
        reg_y = reg_y[1:-1]
    else:
        reg_y = reg_y[:, 1:-1, :]

    np.save(args.data_dir+'/'+'train'+'_y', train_y)
    np.save(args.data_dir+'/'+'val'+'_y', val_y)
    np.save(args.data_dir+'/'+'test'+'_y', test_y)
    np.save(args.data_dir+'/'+'reg'+'_y', reg_y)
    np.save(args.data_dir+'/'+'reg'+'_v', reg_v)

    if data_y.shape[2] == 3 :
        plot_3D(train_y,func_name,args.data_dir,'training') # visualize solution
        plot_3D(val_y,func_name,args.data_dir,'validation')
        plot_3D(test_y,func_name,args.data_dir,'testing')
    

if __name__ == "__main__":
    ''' Make directory to store data. '''
    make_directory(args.data_dir) # make directory to store data

    ''' Initialize ODE input. '''
    t = np.linspace(start=args.T0, stop=args.Tend, num=args.num_point) # define grid on which to solve ODE

    func_name = args.ODE_system
    if func_name == 'Spiral': 
        func = spiral
        y0 = np.array([4, 3, -2]) # define center of neighborhood for initial starting points
    elif func_name == 'Hopf': 
        func = hopf_bifurcation
        y0 = np.array([-0.15, 0, 2])
    elif func_name == 'Lorenz': 
        func = lorenz
        y0 = np.array([-8, 7, 27])
    elif func_name == 'Glycolytic': 
        func = glycolytic_oscillator
        y0 = None # dummy variable 
                  # does not take in y0, samples random y0 instead
        
    ''' Save parameters for reference. '''
    # save args
    with open(args.data_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    np.save(args.data_dir+'/grid',t) # save grid
    np.save(args.data_dir+'/ipcenter',y0) # save center

    generate_data(t=t,y0=y0,func=func,func_name=func_name,data_type=None,num_traj=args.num_traj)
