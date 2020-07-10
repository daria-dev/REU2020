# REU2020

**make_data.py**
- General script to produce data. Supports four ODE systems: Spiral, Lorenz, Hopf, Glycolytic and two ways to split for training/test/validation.

**/rk4**:
- make_data.py: Generates data for model using rk4 using any of the systems
- model.py: Model using rk4 to learn velocity
- model_aux.py: Helper functions
- test_aux.py: Calculates test loss

**/rk4_quadratic_prior**
- model_quadratic_prior.py: Uses rk4 to learn velocity. Adds a linear combination of the product of input features to the neural network output.
- model_quadratic_prior_aux.py: Helper functions
- test_aux_quadratic_prior.py: Calculates test loss

**/changes**:
- make_data.py: Generates spiral data for training, validation, and testing. Computes velocity using either exact, forward Euler, or central difference.
- make_data_sets.py: Splits generated data into training, testing, and validation. Validation and testing trajectories start after the last point of training trajectories.
- make_glycolytic_data.py: Generates glycolitic data
- make_hopf_bifurcation_data.py: Generates hopf bifurcation data
- make_lorenz_data.py: Generates lorenz data
- model.py: Original model with layer depth as argument
- model_aux.py: Original model helper functions with regularization
- test_aux.py: Generates trajectories based on trained model and compares with test data

**/orginial**:
- *.py: Orginial code
