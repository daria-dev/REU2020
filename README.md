# REU2020

files:
  - model.py: defines and trains nn on generated data
  - make_data.py: generates spiral data
  - model_aux.py: helper functions
  
  - changes/model.py: supports multiple layers
  - changes/model_aux.py: supports L1 and L2 regularization
  - changes/make_data.py: generates data using forward Euler or central instead of using exact velocity
  
  - rk4/* : same functionality as the files in changes, loss function uses RK4 
