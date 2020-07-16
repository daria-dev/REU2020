import argparse, json
from model_aux import make_directory, plot_3D
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('DIFF')
parser.add_argument('--ODE_system', type=float, default='Lorenz',
    help='system for which to generate prior derivative')
parser.add_argument('--domain', type=float, default='C',
    help='domain on which to generate prior derivative')
parser.add_argument('--num_point', type=int, default=100,
    help='number of points per trajectory')
parser.add_argument('--num_traj', type=int, default=100,
    help='number of training trajectories')
parser.add_argument('--data_dir', type=str, default='diff_data',
        help='name for data directory')
parser.add_argument('--noise',type=float,default=None,
        help='percent of noise to add as a float, else None')
args = parser.parse_args()

# Spiral ODE Jacobian
def spiral(t, y): 
    '''
    NOTES: Defines Jacobian for ODE system that generates a spiral/corkscrew shape.
            Input y is either a 1D array (used mainly with built in ODE solvers), or 3D array
            (to generate velocity data over multiple trajectories for training).

    INPUT:
        t = dummy variable (system is autonomous, but python solvers require time input)
        y = position data; 1D array, or 3D array with axes
            - 0 = ith trajectory
            - 1 = point at time t_i
            - 2 = spatial dimension y_i

    OUTPUT:
        return #0 = Jacobian matrix
    '''
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'


    # Original
    if len(y.shape) == 1:
        J10 = -1.5*y[0]**2 - 3.*y[0]*y[2]-1.5*y[2]**2
        J = np.array([[0., 3.*y[1]**2, 0.],
                      [J10, -0.2, J10],
                      [0., 0., 0.1]
                     ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[2],y.shape[2]))
        J[:,:,0,0] = 0.
        J[:,:,0,1] = 3.*y[:,:,1]**2
        J[:,:,0,2] = 0.
        J[:,:,1,0] = -1.5*y[:,:,0]**2 - 3.*y[:,:,0]*y[:,:,2]-1.5*y[:,:,2]**2
        J[:,:,1,1] = -0.2
        J[:,:,1,2] = -1.5*y[:,:,0]**2 - 3.*y[:,:,0]*y[:,:,2]-1.5*y[:,:,2]**2
        J[:,:,2,0] = 0.
        J[:,:,2,1] = 0.
        J[:,:,2,2] = 0.1
    return J

# Lorenz ODE Jacobian
def lorenz(t, y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        J = np.array([
            [-10., 10., 0.],
            [28.-y[2], -1., -1*y[0]],
            [y[1], y[0], -8./3.]
        ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[2],y.shape[2]))
        J[:,:,0,0] = -10.
        J[:,:,0,1] = 10.
        J[:,:,0,2] = 0.
        J[:,:,1,0] = 28.-y[:,:,2]
        J[:,:,1,1] -1.
        J[:,:,1,2] = -1*y[:,:,0]
        J[:,;,2,0] = y[:,:,1]
        J[:,:,2,1] = y[:,:,0]
        J[:,:,2,2] = -8./3.

    return J

# Hopf Bifurication ODE System Jacobian
def hopf_bifurcation(t, y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        J = np.array([
            [0., 0., 0.],
            [y[1], y[0]-3.*y[1]**2-y[2]**2, 1-2.*y[1]*y[2]],
            [y[2], -1-2.*y[2]*y[1], y[0]-3*y[2]**2-y[1]**2]
        ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[2],y.shape[2]))
        J[:,:,0,:] = 0.
        J[:,:,1,0] = y[:,:,1]
        J[:,:,1,1] = y[:,:,0]-3.*y[:,:,1]**2-y[:,:,2]**2
        J[:,:,1,2] = 1-2.*y[:,:,1]*y[:,:,2]
        J[:,:,2,0] = y[:,:,2]
        J[:,:,2,1] = -1-2.*y[:,:,2]*y[:,:,1]
        J[:,:,2,2] = y[:,:,0]-3*y[:,:,2]**2-y[:,:,1]**2
    return J
    
# Glycolytic ODE System Jacobian
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
        J00 = -(k1*y[5])/(1+(y[5]/K1)**q)
        J05 = -1*( ((1+(y[5]/K1)**q)k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )

        J10 = 2*(k1 *y[5]) / (1 +(y[5] / K1) ** q)
        J11 = - k2*(N - y[4]) - k6*y[4]
        J14 = - k2*y[1] - k6*y[1]
        J15 = 2*( ((1+(y[5]/K1)**q)k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )
        
        J21 = k2 * (N - y[4])
        J22 = -k3 * (A - y[5])
        J24 = k2 * y[1]
        J25 = - k3 * y[2]

        J32 = k3 * (A - y[5])
        J33 = - k4 * y[3] - kappa
        J34 = - k4 * y[3]
        J35 = - k3 * y[2]
        J36 = kappa

        J41 = k2 * (N - y[4])- k6 * y[4]
        J43 = - k4 * y[4]
        J44 = -k2 * y[1] - k4 * y[3] - k6 * y[1]

        J50 = -2 * (k1 * y[5]) / (1 + (y[5] / K1) ** q)
        J52 = 2 * k3 *(A - y[5])
        J55 = -2*( ((1+(y[5]/K1)**q)k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )
               - 2 * k3 * y[2] - k5

        J63 = psi * kappa
        J66 = -psi * kappa - k

        J = np.array([
            [J00, 0., 0., 0., 0., J05, 0.],
            [J10, J11, 0., 0., J14, J15, 0.],
            [0., J21, J22, 0., J24, J25, 0.],
            [0., 0., J32, J33, J34, J35, J36],
            [0., J41, 0., J43, J44, 0., 0.],
            [J50, 0., J52, 0., 0., J55, 0.],
            [0., 0., 0., J63, 0., 0., J66]

        ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[2],y.shape[2]))
        J[:,:,0,0] = -(k1*y[:,:,5])/(1+(y[:,:,5]/K1)**q)
        J[:,:,0,5] = -1*( ((1+(y[:,:,5]/K1)**q)k1*y[:,:,0]-k1y[:,:,0]*y[:,:,5]*q*(y[:,:,5]/K1)**(q-1)) 
                    / ((1 + (y[:,:,5] / K1) ** q)**2) )

        J[:,:,1,0] = 2*(k1 *y[:,:,5]) / (1 +(y[:,:,5] / K1) ** q)
        J[:,:,1,1] = - k2*(N - y[4]) - k6*y[4]
        J14 = - k2*y[1] - k6*y[1]
        J15 = 2*( ((1+(y[5]/K1)**q)k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )
        
        J21 = k2 * (N - y[4])
        J22 = -k3 * (A - y[5])
        J24 = k2 * y[1]
        J25 = - k3 * y[2]

        J32 = k3 * (A - y[5])
        J33 = - k4 * y[3] - kappa
        J34 = - k4 * y[3]
        J35 = - k3 * y[2]
        J36 = kappa

        J41 = k2 * (N - y[4])- k6 * y[4]
        J43 = - k4 * y[4]
        J44 = -k2 * y[1] - k4 * y[3] - k6 * y[1]

        J50 = -2 * (k1 * y[5]) / (1 + (y[5] / K1) ** q)
        J52 = 2 * k3 *(A - y[5])
        J55 = -2*( ((1+(y[5]/K1)**q)k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )
               - 2 * k3 * y[2] - k5

        J63 = psi * kappa
        J66 = -psi * kappa - k

    return J

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
    
    # save data for model 1
    if args.split_method == 1: 
        np.save(args.data_dir+'/data_y', data_y)
        if data_y.shape[2] == 3 :
            plot_3D(data_y,func_name,args.data_dir,data_type) # visualize solution
    
    # split and save train/val/test based on model 2
    elif args.split_method == 2:
        train_y = data_y[:,:args.T_ss,:]
        val_y = data_y[:args.V_ss,args.T_ss:,:]
        test_y = data_y[args.V_ss:, args.T_ss:, :]
        np.save(args.data_dir+'/'+'train'+'_y', train_y)
        np.save(args.data_dir+'/'+'val'+'_y', val_y)
        np.save(args.data_dir+'/'+'test'+'_y', test_y)
        

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
    
    # Generate data based on model 1
    if args.split_method == 1:
        generate_data(t=t,y0=y0,func=func,func_name=func_name,data_type='training',num_traj=args.num_traj)
        generate_data(t=t,y0=y0,func=func,func_name=func_name,data_type='validation',num_traj=int(args.num_traj*0.2))
        generate_data(t=t,y0=y0,func=func,func_name=func_name,data_type='testing',num_traj=int(args.num_traj*0.2))
    
    # Generate data based on model 2
    else:
        generate_data(t=t,y0=y0,func=func,func_name=func_name,data_type=None,num_traj=args.num_traj)
