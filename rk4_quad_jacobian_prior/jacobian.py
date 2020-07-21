import argparse, json
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import math

def spiral(y): 
    '''
    NOTES: Defines Jacobian for ODE system that generates a spiral/corkscrew shape.
            Input y is either a 1D array , or 2D array
            (to generate jacobian data over a 2D boundary).

    INPUT:
        y = position data; 1D array, or 2D array with axes
            - 0 = ith set of points
            - 1 = spatial dimension y_i

    OUTPUT:
        return #0 = Jacobian matrix
    '''
    assert ((len(y.shape) == 1) or (len(y.shape) == 2)),'y must be a 1D or 2D array.'


    # Original
    if len(y.shape) == 1:
        J10 = -1.5*y[0]**2 - 3.*y[0]*y[2]-1.5*y[2]**2
        J = np.array([[0., 3.*y[1]**2, 0.],
                      [J10, -0.2, J10],
                      [0., 0., 0.1]
                     ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[1]))
        J[:,0,0] = 0.
        J[:,0,1] = 3.*y[:,1]**2
        J[:,0,2] = 0.
        J[:,1,0] = -1.5*y[:,0]**2 - 3.*y[:,0]*y[:,2]-1.5*y[:,2]**2
        J[:,1,1] = -0.2
        J[:,1,2] = -1.5*y[:,0]**2 - 3.*y[:,0]*y[:,2]-1.5*y[:,2]**2
        J[:,2,0] = 0.
        J[:,2,1] = 0.
        J[:,2,2] = 0.1
    return J

# Lorenz ODE Jacobian
def lorenz(y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 2)),'y must be a 1D or 2D array.'

    if len(y.shape) == 1:
        J = np.array([
            [-10., 10., 0.],
            [28.-y[2], -1., -1*y[0]],
            [y[1], y[0], -8./3.]
        ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[1]))
        J[:,0,0] = -10.
        J[:,0,1] = 10.
        J[:,0,2] = 0.
        J[:,1,0] = 28.-y[:,2]
        J[:,1,1] -1.
        J[:,1,2] = -1*y[:,0]
        J[:,2,0] = y[:,1]
        J[:,2,1] = y[:,0]
        J[:,2,2] = -8./3.

    return J

# Hopf Bifurication ODE System Jacobian
def hopf_bifurcation(y): 
    assert ((len(y.shape) == 1) or (len(y.shape) == 2)),'y must be a 1D or 2D array.'

    if len(y.shape) == 1:
        J = np.array([
            [0., 0., 0.],
            [y[1], y[0]-3.*y[1]**2-y[2]**2, 1-2.*y[1]*y[2]],
            [y[2], -1-2.*y[2]*y[1], y[0]-3*y[2]**2-y[1]**2]
        ])
    else:
        J = np.zeros((y.shape[0],y.shape[1],y.shape[1]))
        J[:,0,:] = 0.
        J[:,1,0] = y[:,1]
        J[:,1,1] = y[:,0]-3.*y[:,1]**2-y[:,2]**2
        J[:,1,2] = 1-2.*y[:,1]*y[:,2]
        J[:,2,0] = y[:,2]
        J[:,2,1] = -1-2.*y[:,2]*y[:,1]
        J[:,2,2] = y[:,0]-3*y[:,2]**2-y[:,1]**2
    return J
    
# Glycolytic ODE System Jacobian
def glycolytic_oscillator(y):
    
    assert ((len(y.shape) == 1) or (len(y.shape) == 2)), 'y must be a 1D or 2D array.'

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
        J05 = -1*( ((1+(y[5]/K1)**q)*k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )

        J10 = 2*(k1 *y[5]) / (1 +(y[5] / K1) ** q)
        J11 = - k2*(N - y[4]) - k6*y[4]
        J14 = - k2*y[1] - k6*y[1]
        J15 = 2*( ((1+(y[5]/K1)**q)*k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )
        
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
        J55 = -2*( ((1+(y[5]/K1)**q)*k1*y[0]-k1y[0]*y[5]*q*(y[5]/K1)**(q-1)) / ((1 + (y[5] / K1) ** q)**2) )- 2 * k3 * y[2] - k5

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
        J = np.zeros((y.shape[0],y.shape[1],y.shape[1]))
        J[:,0,0] = -(k1*y[:,5])/(1+(y[:,5]/K1)**q)
        J[:,0,5] = -1*( ((1+(y[:,5]/K1)**q)*k1*y[:,0]-k1y[:,0]*y[:,5]*q*(y[:,5]/K1)**(q-1)) 
                    / ((1 + (y[:,5] / K1) ** q)**2) )

        J[:,1,0] = 2*(k1 *y[:,5]) / (1 +(y[:,5] / K1) ** q)
        J[:,1,1] = - k2*(N - y[:,4]) - k6*y[:,4]
        J[:,1,4] = - k2*y[:,1] - k6*y[:,1]
        J[:,1,5] = 2*( ((1+(y[:,5]/K1)**q)*k1*y[:,0]-k1y[:,0]*y[:,5]*q*(y[:,5]/K1)**(q-1)) 
                / ((1 + (y[:,5] / K1) ** q)**2) )
        
        J[:,2,1] = k2 * (N - y[:,4])
        J[:,2,2] = -k3 * (A - y[:,5])
        J[:,2,4] = k2 * y[:,1]
        J[:,2,5] = - k3 * y[:,2]

        J[:,3,2] = k3 * (A - y[:,5])
        J[:,3,3] = - k4 * y[:,3] - kappa
        J[:,3,4] = - k4 * y[:,3]
        J[:,3,5] = - k3 * y[:,2]
        J[:,3,6] = kappa

        J[:,4,1] = k2 * (N - y[:,4])- k6 * y[:,4]
        J[:,4,3] = - k4 * y[:,4]
        J[:,4,4] = -k2 * y[:,1] - k4 * y[:,3] - k6 * y[:,1]

        J[:,5,0] = -2 * (k1 * y[:,5]) / (1 + (y[:,5] / K1) ** q)
        J[:,5,2] = 2 * k3 *(A - y[:,5])
        J[:,5,5] = -2*( ((1+(y[:,5]/K1)**q)*k1*y[:,0]-k1*y[:,0]*y[:,5]*q*(y[:,5]/K1)**(q-1)) / 
                    ((1 + (y[:,5] / K1) ** q)**2) ) - 2 * k3 * y[:,2] - k5

        J[:,6,3] = psi * kappa
        J[:,6,6] = -psi * kappa - k
    return J
    
def generate_jacobian(y,func,num_boundary_points):
    # finds Jacobian over a boundary for a function 
    # y = all generated data i.e. np.stack(y_train,y_val_y,y_test)
    # func = function name
    # num_boundary_points = 0.7*args.num_traj * args.num_point 
    #           = number of points on the boundary
    
    # impose boundary on left-most boundary wall
    inf_norm = np.amax(np.abs(y))
    
    boundary_top = np.amax(y[:,:,2]) + inf_norm*1.05
    boundary_bot = np.amin(y[:,:,2]) - inf_norm*1.05
    
    boundary_lef = np.amax(y[:,:,1]) + inf_norm*1.05
    boundary_rig = np.amin(y[:,:,1]) - inf_norm*1.05
    
    boundary_x = np.amin(y[:,:,0]) # left most side where boundary is imposed
    boundary_center_v = (boundary_top - boundary_bot)/2
    boundary_len = (boundary_top - boundary_bot)
    
    boundary_center_h = (boundary_lef - boundary_rig)/2
    boundary_wid = (boundary_lef - boundary_rig)
    
    # Domain consists of normally distributed points around boundary wall
    x = boundary_x*np.ones(num_boundary_points)
    y = np.random.uniform(boundary_center_h,boundary_len,num_boundary_points)
    z = np.random.uniform(boundary_center_v,boundary_wid,num_boundary_points)
    
    D = np.stack((x,y,z),axis=-1)
    
    if func == 'Spiral':
        jacobian = spiral(D)
    elif func == 'Hopf':
        jacobian = hopf_bifurcation(D)
    elif func == 'Glycolytic':
        jacobian = glycolytic_oscillator(D)
    elif func == 'Lorenz':
        jacobian = lorenz(D)
    
    return jacobian, D