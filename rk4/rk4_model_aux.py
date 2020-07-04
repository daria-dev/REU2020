ó
ëúø^c           @  s´   d  d l  m Z d  d l Z d  d l Z d  d l Z d  d l j Z d  d l Z	 d  d l
 m Z d   Z d   Z d   Z d   Z d   Z d	   Z d
   Z d   Z d   Z d S(   iÿÿÿÿ(   t   print_functionN(   t	   solve_ivpc         C  sD   t  |   t k s t d   t j j |   s@ t j |   n  d S(   s   
    NOTES: Makes directory (if it doesn't already exist).

    INPUT: 
        dir_name = string; name of directory

    OUTPUT:
        None
    s   dir_name must be string.N(   t   typet   strt   AssertionErrort   ost   patht   existst   makedirs(   t   dir_name(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   make_directory   s    
c         C  sÇ  t  |  j  d k s! t d   |  j d d k s@ t d   t |  t k s^ t d   t |  t k s| t d   t |  t k s t d   t j   |  j d } t j   } x t |  D] } | j	 d	 d
  } | j
 |  | d d  d f |  | d d  d f |  | d d  d f d d | j d d d d  qÊ W| j d  | j d  | j d  | j | d | d  t j | d | d | d  t j   t j   d S(   sÃ  
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
    i   s   data must be 3D array.i   s   data must be 3D.s   func_name must be string.s   dir_name must be string.s   data_type must be string.i    t
   projectiont   3dNi   t   lwg      à?t   elevi   t   azimi{ÿÿÿt   y_1t   y_2t   y_3s   : s    datat   /t   _s	   _data.png(   t   lent   shapeR   R   R   t   pltt   closet   figuret   ranget   gcat   plott	   view_initt
   set_xlabelt
   set_ylabelt
   set_zlabelt	   set_titlet   savefigt   show(   t   datat	   func_nameR	   t	   data_typet   tot_num_trajt   figt   trajt   ax(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   plot_3D   s&    !
R!
c         C  s  t  |   t k s t d   t  |  t k s< t d   t  |  t k sZ t d   t  |  t k sx t d   t  |  t k s t d   d j |  d |  } | d d	 j |  7} | d d
 j |  7} | ró t | d d n
 t |  d S(   s  
    NOTES: Structure for nice printing of train/validation loss for given epoch.

    INPUT: 
        epoch = int; current epoch number
        num_epoch = int; total number of epochs that will be executed
        loss_train = float; error for training data
        loss_val = float; error for validation data
        overwrite = bool; choice to overwrite printed line

    OUTPUT:
        None
    s   epoch must be int.s   num_epoch must be int.s   loss_train must be float.s   loss_val must be float.s   overwrite must be bool.s   Epoch {}/{}i   s    | s   Train Loss: {:.8f}s   Validation Loss: {:.8f}t   ends   N(   R   t   intR   t   floatt   boolt   formatt   print(   t   epocht	   num_epocht
   loss_traint   loss_valt	   overwritet   line(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   print_epoch<   s    c      	     s)  t  |  j  d k s! t d   t  | j  d k sB t d   t j   } t j |  d   j   } t j |  d  j   } t j | d   j   }	 t j | d  j   }
 t j j j	 | |  } t j j j
 d | d   j d t  } d	 } xë t   j  D]Ú } xU t |  D]G \ } \   | d 7}        f d
   }  j |  qW| d d d	 k rt j   /   |  |  }   |	  |
  } Wd QXt |   j | j   | j   d t qqWt j   } t d j | |   t j  j     j d  d S(   s¨  
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
    i   s    training data should be 3D arrays"   validation data should be 3D arrayiÿÿÿÿi   t   datasett
   batch_sizet   shufflei    c            s    j          }    j d k r t j d  } t j   j  } x$  j   D] } | | j   7} q\ W|  | | 7}  n    j d k rò t j d  } t j   j  } x'  j   D] } | | j d  7} qÄ W|  | | 7}  n  |  j   |  S(   Nt   L2g        t   L1i   (   t	   zero_gradt   Regt   torcht   tensort   Lambdat
   parameterst   normt   backward(   t   losst   l2RB   t   wt   l1(   t   argst	   criteriont   nett	   optimizert   train_y1_batcht   train_y_batch(    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   closure   s     

iô  NR6   s   
=====> Running time: {}s   /net_state_dict.pt(   R   R   R   t   timeR@   t
   from_numpyR.   t   utilsR$   t   TensorDatasett
   DataLoaderR:   t   TrueR   R3   t	   enumeratet   stept   no_gradR8   t   itemt   FalseR1   R0   t   savet
   state_dictt   log_dir(   t   train_yt   val_yRL   RK   RM   RJ   t   startt   train_y_tensort   train_y1_tensort   val_y_tensort   val_y1_tensort   train_datasett   train_loadert   current_stepR2   t   iRP   R4   R5   R,   (    (   RJ   RK   RL   RM   RN   RO   s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   train_nnX   s0    !!
/c      	   C  sV   t  |  j  d k s! t d   t j |   t j j d d d d d t  |   f  S(   Ni   s   y0 must be a 1D array.t   lowiþÿÿÿt   highi   t   size(   R   R   R   t   npt   arrayt   randomt   uniform(   t   y0(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   initial§   s    !c         C  s   t  | j  d k s! t d   t d | d d d g d t |  d d	 d
 |  d t d t d d d d t j t  |  f   	} t j | j	  S(   Ni   s   y0 must be a 1D array.t   funt   t_spani    i   Rr   t   methodt   RK45t   t_evalt   dense_outputt
   vectorizedt   rtolgÖ&è.>t   atol(
   R   R   R   R   Rs   R[   Rn   t   onest	   transposet   y(   t   tRr   t   funct   sol(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt	   solve_ODE¬   s    !%c         C  s¤  t  | j  d k s6 t  | j  d k s6 t d   t  | j  d k r t j | d d | d | d d d d | d | d d g  } n	t j | j  } | d  d   d  d   d f d | d  d   d  d   d f <| d  d   d  d   d f | d  d   d  d   d f d d d | d  d   d  d   d f | d  d   d  d   d f <| d  d   d  d   d f d | d  d   d  d   d f <| S(	   Ni   i   s   y must be a 1D or 3D array.i    i   gÉ?g      $@g       @(   R   R   R   Rn   Ro   t   zeros(   R   R   t   v(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   spiral¶   s    6L<<c         C  sp   g  } x3 t  d  D]% } t |  | |  } | j |  q Wt j | d d } | d d  d |  } | | f S(   Ni   t   axisi    R   R   (   R   R   t   appendRn   t   stackt   None(   R   Rr   R   t   data_yR   R   t   data_v(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   generate_test_trajÆ   s    c   
      C  sç   t  j d d d d d d  } t  j d d d	 g  } g  } g  } g  t d  D] } t |  ^ qO } d } xc | D][ } t | | t  \ } } |  t j |  j	    }	 | | |	 t j |  j	    7} qt W| d :} | j
   S(
   NRa   i    t   stopi   t   numid   i   i   iþÿÿÿ(   Rn   t   linspaceRo   R   Rs   R   R   R@   RR   R.   RZ   (
   RL   RK   R   Rr   R   R   R   t   y0_lRF   t   v_hat(    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt	   test_lossÑ   s    %&
(   t
   __future__R    R   RQ   R@   t   matplotlib.pyplott   pyplotR   t   numpyRn   t   scipy.integrateR   R
   R+   R8   Rj   Rs   R   R   R   R   (    (    (    s9   /mnt/c/Users/the_d/Documents/REU2020/rk4/rk4_model_aux.pyt   <module>   s   $		&		O		
		