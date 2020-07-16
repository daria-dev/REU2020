from __future__ import print_function
import os, time, torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def make_directory(dir_name):
    '''
    NOTES: Makes directory (if it doesn't already exist).

    INPUT: 
        dir_name = string; name of directory

    OUTPUT:
        None
    '''
    assert (type(dir_name) == str),'dir_name must be string.'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# NOTE: time parameters set for Lorenz system
def fwd_euler(y): 
    '''
    NOTES: Calculates velocity using the Forward Euler Approximation
            Input y is either a 1D array (used mainly with built in ODE solvers), or 3D array
            (to generate velocity data over multiple trajectories for training).

    INPUT:
        y = position data; 1D array, or 3D array with axes
    OUTPUT:
        return #0 = velocity data
    '''

    assert ((len(y.shape) == 1) or (len(y.shape) == 3)),'y must be a 1D or 3D array.'

    if len(y.shape) == 1:
        dt = (25 - 0)/len(y)
        end = len(y)
        v = np.array([ (y[2:end]-y[1:end-1])/dt ])
    else:
        dt = (25 - 0)/len(y[0,:])
        end = y.shape[1]
        v = np.zeros((y.shape[0],y.shape[1]-2,y.shape[2]))
        v[:,:,:] = (y[:,2:end,:]-y[:,1:end-1,:])/dt
    return v

###############################
''' Graphing and output Utils '''
################################

def plot_3D(data,func_name,dir_name,data_type):
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
    assert (type(data_type) == str),'data_type must be string.'

    plt.close()
    tot_num_traj = data.shape[0]
    fig = plt.figure()
    for traj in range(tot_num_traj):
        ax = fig.gca(projection='3d')
        ax.plot(data[traj,:,0],data[traj,:,1],data[traj,:,2], lw=0.5)
        ax.view_init(elev=26,azim=-133)
    ax.set_xlabel('y_1')
    ax.set_ylabel('y_2')
    ax.set_zlabel('y_3')
    ax.set_title(func_name+': '+data_type+' data')
    # plt.savefig(args.data_dir+'/spiral_'+('train' if train else 'test')+'_data.svg')
    plt.savefig(dir_name+'/'+func_name+'_'+data_type+'_data.png')
    plt.show()
    plt.close()

def print_epoch(epoch,num_epoch,loss_train,loss_val,overwrite):
    '''
    NOTES: Structure for nice printing of train/validation loss for given epoch.

    INPUT: 
        epoch = int; current epoch number
        num_epoch = int; total number of epochs that will be executed
        loss_train = float; error for training data
        loss_val = float; error for validation data
        overwrite = bool; choice to overwrite printed line

    OUTPUT:
        None
    '''
    assert (type(epoch) == int),'epoch must be int.'
    assert (type(num_epoch) == int),'num_epoch must be int.'
    assert (type(loss_train) == float),'loss_train must be float.'
    assert (type(loss_val) == float),'loss_val must be float.'
    assert (type(overwrite) == bool),'overwrite must be bool.'

    line = 'Epoch {}/{}'.format(epoch+1, num_epoch)
    line += ' | ' + 'Train Loss: {:.8f}'.format(loss_train)
    line += ' | ' + 'Validation Loss: {:.8f}'.format(loss_val)
    if overwrite:
        print(line, end='\r')
    else:
        print(line)

def plot_loss(train_loss, val_loss, epoch_list, dir_name):
    '''
    NOTES: Plots 2D data and saves plot as png.

    INPUT:
        epoch_list = list of epochs in which loss data collected
        train_loss = list of train losses
        val_loss = list of val losses
        dir_name = str; name of directory to save plot

    OUTPUT:
        None
    '''

    assert (type(dir_name) == str),'dir_name must be string.'
    assert (len(train_loss) == len(val_loss)),'same number of datapoints'
    assert (len(train_loss) == len(epoch_list)),'same number of datapoints'
    
    plt.close()
    fig = plt.figure()
    plt.plot(epoch_list, train_loss, label = 'Train')
    plt.plot(epoch_list, val_loss, label = 'Validation')
    fig.suptitle('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper left")
    plt.yscale('log')
    plt.savefig(dir_name+'/'+ 'Loss.png')
    plt.show()
    plt.close()

###########################################
'''Helper functions for generating prior'''
##########################################

def lorenz_prior(x0):
    y = torch.zeros(x0.shape[0], x0.shape[1], requires_grad=False)

    for n in range(x0.shape[0]):
        # fill with relevant products
        y[n,:] = torch.tensor([0, x0[n,0]*x0[n,2], x0[n,0]*x0[n,1]])
        
    return y

# helper function for make product
# [in] x0 = N x M matrix
# [out] y = N x M(M-1) matrix where each row n contains
#       elements x0_n,i*x0_n,j for j,i <= M. No row contains
#       duplicates, i.e only one of x0_n,i*x0_n,j and x0_n,j*x0_n,i
#       is present for each row n.
def product(x0):
    # dimensions of x0
    N = x0.shape[0]
    M = x0.shape[1]
        
    # creates N x (M + (M choose 2)) matrix
    y = torch.zeros(N, M*(M+1)//2)

    # fills y with products, discarding duplicates
    for n in range(N):
        x = torch.reshape(x0[n,:], (1,M))
        prod = x.T @ x

        # get upper triangular part of prod + diagonal
        y[n,:] = prod[torch.triu_indices(M,M)[0], torch.triu_indices(M,M)[1]]
        
    return y

# caclulates prior for each trajectory
# [in] x0 = input to neural network
# [out] y = tensor of products
def make_product(x0):
    if len(x0.shape) == 3:
        y = torch.zeros(x0.shape[0], x0.shape[1], x0.shape[2], requires_grad=False)

        for k in range(x0.shape[0]):
            y[k,:,:] = lorenz_prior(x0[k,:,:])
    else:
        y = lorenz_prior(x0)
        
    return y;

#######################################
'''Neural Network training function'''
########################################

def train_nn(train_y,val_y,net,criterion,optimizer,args):
    '''
    NOTES: Trains neural network and checks against validation data, and saves network state_dict. 
            All data has the following structure: 3D array with axes
                - 0 = ith trajectory/sample
                - 1 = point at time t_i
                - 2 = spatial dimension y_i

    INPUT: 
        train_y = 3D array; training data input
        train_v = 3D array; training data output
        val_y = 3D array; validation data input
        val_v = 3D array; validation data output
        net = network function
        criterion = loss function
        optimizer = optimization algorithm
        args = user input/parameters

    OUTPUT:
        None
    '''
    assert (len(train_y.shape) == 3),'training data should be 3D array'
    assert (len(val_y.shape) == 3),'validation data should be 3D array'

    start = time.time()

    # convert data to torch tensor
    # uses y to predict y at next time step instead of predicting v
    train_y_tensor = torch.from_numpy(train_y[:,:-1]).float()
    train_y1_tensor = torch.from_numpy(train_y[:,1:]).float()
    val_y_tensor = torch.from_numpy(val_y[:,:-1]).float()
    val_y1_tensor = torch.from_numpy(val_y[:,1:]).float()

    # create batches for training data
    train_dataset = torch.utils.data.TensorDataset(train_y_tensor, train_y1_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # holds loss data
    train_list,val_list,epoch_list = [],[],[]

    # run training
    current_step = 0
    for epoch in range(args.num_epoch):
        for i, (train_y_batch, train_y1_batch) in enumerate(train_loader):
            current_step += 1

            def closure():
                optimizer.zero_grad()
                loss = criterion(net(train_y_batch), train_y1_batch)

                # add regularization if necessary
                if args.Reg == 'L2':
                        l2 = torch.tensor(0.0)
                        Lambda = torch.tensor(args.Lambda)
                        for n,w in net.named_parameters():
                            if (n != "f.priorLayer.weight"):
                                l2 += w.norm()
                        loss += l2*Lambda

                if args.Reg == 'L1':
                        l1 = torch.tensor(0.0)
                        Lambda = torch.tensor(args.Lambda)
                        for n,w in net.named_parameters():
                            if (n != "f.priorLayer.weight"):
                                l1 += w.norm(1)
                        loss += l1*Lambda

                loss.backward()
                return loss
            optimizer.step(closure)

        if (epoch+1) % 100 == 0:
            with torch.no_grad():
                loss_train = criterion(net(train_y_tensor), train_y1_tensor)
                loss_val = criterion(net(val_y_tensor), val_y1_tensor)
                train_list.append(loss_train)
                val_list.append(loss_val)
                epoch_list.append(epoch + 1)

            print_epoch(epoch, args.num_epoch, loss_train.item(), loss_val.item(), overwrite=False) # print at each batch

    end = time.time()
    print('\n=====> Running time: {}'.format(end-start))

    torch.save(net.state_dict(),args.log_dir+'/net_state_dict.pt')

    plot_loss(train_list, val_list, epoch_list, args.data_dir)

