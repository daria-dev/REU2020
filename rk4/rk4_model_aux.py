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
    train_y_tensor = torch.from_numpy(train_y[:-1]).float()
    train_y1_tensor = torch.from_numpy(train_y[1:]).float()
    val_y_tensor = torch.from_numpy(val_y[:-1]).float()
    val_y1_tensor = torch.from_numpy(val_y[1:]).float()

    # create batches for training data
    train_dataset = torch.utils.data.TensorDataset(train_y_tensor, train_y1_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

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
                        for w in net.parameters():
                                l2 += w.norm()
                        loss += l2*Lambda

                if args.Reg == 'L1':
                        l1 = torch.tensor(0.0)
                        Lambda = torch.tensor(args.Lambda)
                        for w in net.parameters():
                                l1 += w.norm(1)
                        loss += l1*Lambda

                loss.backward()
                return loss
            optimizer.step(closure)

        if (epoch+1) % 500 == 0:
            with torch.no_grad():
                loss_train = criterion(net(train_y_tensor), train_y1_tensor)
                loss_val = criterion(net(val_y_tensor), val_y1_tensor)
            print_epoch(epoch, args.num_epoch, loss_train.item(), loss_val.item(), overwrite=False) # print at each batch

    end = time.time()
    print('\n=====> Running time: {}'.format(end-start))

    torch.save(net.state_dict(),args.log_dir+'/net_state_dict.pt')

# redefining some function from make_data to avoid command line arg problems
# maybe make a new class for this?
def initial(y0):
    assert (len(y0.shape) == 1),'y0 must be a 1D array.'

    return np.array(y0)+np.random.uniform(low=-2, high=2, size=(len(y0),))

def solve_ODE(t,y0,func):

    assert (len(y0.shape) == 1),'y0 must be a 1D array.'
    sol = solve_ivp(fun=func, t_span=[0, 5],
        y0=initial(y0), method='RK45', t_eval=t,
        dense_output=False, vectorized=False,
        rtol=1e-9, atol=1e-9*np.ones((len(y0),)))

    return np.transpose(sol.y)

def spiral(t, y):
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


# generate test trajectories with exact velocity
def generate_test_traj(t,y0,func):
    data_y = []
    for _ in range(5):
        y = solve_ODE(t,y0,func)
        data_y.append(y)
    data_y = np.stack(data_y, axis=0)
    data_v = func(t=None,y=data_y) # exact velocity calculated

    return (data_v, data_y)

# compute test loss for net using criterion and randomly generated trajectories
def test_loss (net, criterion):
    t = np.linspace(start=0, stop=5, num=100) # define grid on which to solve ODE
    y0 = np.array([4, 3, -2]) # define center of neighborhood for initial starting points


    v = []
    y = []
    y0_l = [initial(y0) for _ in range(5)]
    loss = 0

    for y0 in y0_l:
        v, y = generate_test_traj(t, y0, spiral)
        v_hat = net(torch.from_numpy(y).float())
        loss += criterion(v_hat, torch.from_numpy(v).float())
    
    loss /= 5

    return loss.item()


