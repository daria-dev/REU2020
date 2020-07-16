## [in] - ODE System
#       - domain (file with points)
#       - directory where to save jacobian 
#
# [out] - file with Jacobian on domain points


import argparse, json
from model_aux import make_directory, plot_3D
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('DIFF')
parser.add_argument('--ODE_system', type=float, default='Lorenz',
    help='system for which to generate prior derivative')
parser.add_argument('--domain', type=float,
    help='domain on which to generate prior derivative')
parser.add_argument('--log_dir', type=str, default='diff_data',
        help='name for data directory in which to save derivative')
parser.add_argument('--noise',type=float,default=None,
        help='percent of noise to add as a float, else None')
args = parser.parse_args()

# Spiral ODE Jacobian
def spiral(y): 
    '''
    NOTES: Defines Jacobian for ODE system that generates a spiral/corkscrew shape.
            Input y is either a 1D array , or 3D array
            (to generate jacobian data over multiple trajectories).

    INPUT:
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
def lorenz(y): 
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
def hopf_bifurcation(y): 
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
def glycolytic_oscillator(y):
    
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
        J[:,:,1,1] = - k2*(N - y[:,:,4]) - k6*y[:,:,4]
        J[:,:,1,4] = - k2*y[:,:,1] - k6*y[:,:,1]
        J[:,:,1,5] = 2*( ((1+(y[:,:,5]/K1)**q)k1*y[:,:,0]-k1y[:,:,0]*y[:,:,5]*q*(y[:,:,5]/K1)**(q-1)) 
                / ((1 + (y[:,:,5] / K1) ** q)**2) )
        
        J[:,:,2,1] = k2 * (N - y[:,:,4])
        J[:,:,2,2] = -k3 * (A - y[:,:,5])
        J[:,:,2,4] = k2 * y[:,:,1]
        J[:,:,2,5] = - k3 * y[:,:,2]

        J[:,:,3,2] = k3 * (A - y[:,:,5])
        J[:,:,3,3] = - k4 * y[:,:,3] - kappa
        J[:,:,3,4] = - k4 * y[:,:,3]
        J[:,:,3,5] = - k3 * y[:,:,2]
        J[:,:,3,6] = kappa

        J[:,:,4,1] = k2 * (N - y[:,:,4])- k6 * y[:,:,4]
        J[:,:,4,3] = - k4 * y[:,:,4]
        J[:,:,4,4] = -k2 * y[:,:,1] - k4 * y[:,:,3] - k6 * y[:,:,1]

        J[:,:,5,0] = -2 * (k1 * y[:,:,5]) / (1 + (y[:,:,5] / K1) ** q)
        J[:,:,5,2] = 2 * k3 *(A - y[:,:,5])
        J[:,:,5,5] = -2*( ((1+(y[:,:,5]/K1)**q)k1*y[:,:,0]-k1*y[:,:,0]*y[:,:,5]*q*(y[:,:,5]/K1)**(q-1)) / 
                    ((1 + (y[:,:,5] / K1) ** q)**2) ) - 2 * k3 * y[:,:,2] - k5

        J[:,:,6,3] = psi * kappa
        J[:,:,6,6] = -psi * kappa - k
    return J

# TODO: finish generate data
def generate_data(func, domain):
    '''
    NOTES: Generates jacobian specified by func on domain points

    INPUT:
        func = function defining Jacobian
        domain = points on which to caclulate

    OUTPUT:
        saves jacobian in file
    '''
    assert (type(func_name) == str),'func_name must be string.'

    return
    
    

if __name__ == "__main__":
    ''' Make directory to store data. '''
    make_directory(args.log_dir) # make directory to store Jacobian


    func_name = args.ODE_system
    if func_name == 'Spiral': 
        func = spiral
    elif func_name == 'Hopf': 
        func = hopf_bifurcation
    elif func_name == 'Lorenz': 
        func = lorenz
    elif func_name == 'Glycolytic': 
        func = glycolytic_oscillator
        
    ''' Save parameters for reference. '''
    # save args
    with open(args.data_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    np.save(args.data_dir+'/grid',t) # save grid
    np.save(args.data_dir+'/ipcenter',y0) # save center
    
    # Generate data
    generate_data(func, args.domain)
