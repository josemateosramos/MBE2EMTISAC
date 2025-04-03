# Information regarding the files in the lib/ folder

This document contains important information to read regarding the files in the lib/ folder.

## Inputs to many of the functions in ```functions.py```
Some of the functions in the 'functions.py' file have the same input variables. To avoid a lenghty description of those functions, here's the description of the common input variables, which can also be found at the end of the 'functions.py' file:

- ```network```: network with learnable parameters to optimize.
- ```train_it```: number of training iterations. Integer.
- ```maxTargets```: maximum number of targets in the radar channel. Integer.
- ```maxPaths``` or ```maxPaths_comm```: maximum number of scatterers for communications. Integer.
- ```P_power```: transmitted power. Float.
- ```mean_rcs```: mean value of the radar cross section of the targets. Float
- ```Rcp```: equivalent range of the cyclic prefix time. Real tensor.
- ```theta_mean_min_sens_train``` or ```theta_mean_min_comm_train```: minimum value of the mean of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- ```theta_mean_max_sens_train``` or ```theta_mean_max_comm_train```: maximum value of the mean of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- ```theta_span_min_sens_train``` or ```theta_span_min_comm_train```: minimum value of the span of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- ```theta_span_max_sens_train``` or ```theta_span_max_comm_train```: maximum value of the span of the a priori angular
sector of the targets. Real tensor of shape (batch_size, 1).
- ```range_mean_min_sens_train``` or ```range_mean_min_comm_train```: minimum value of the mean of the a priori range sector of the targets. Real tensor of shape (batch_size, 1).
- ```range_mean_max_sens_train``` or ```range_mean_max_comm_train```: maximum value of the mean of the a priori range sector of the targets. Real tensor of shape (batch_size, 1).
- ```range_span_min_sens_train``` or ```range_span_min_comm_train```: minimum value of the span of the a priori range sector of the targets. Real tensor of shape (batch_size, 1).
- ```range_span_max_sens_train``` or ```range_span_max_comm_train```: maximum value of the span of the a priori range sector of the targets. Real tensor of shape (batch_size, 1).
- (The theta_ and range_ variables before also have a _test counterpart for inference)
- ```numAntennas``` or ```K```: number of antennas of the ISAC transceiver. Integer.
- ```numSubcarriers``` or ```S```: number of subcarriers of the OFDM signal. Integer.
- ```Delta_f```: subcarrier spacing. Float.
- ```lamb```: wavelength. Float.
- ```Ngrid_angle```: number of candidate angles to try in beamforming and at the Rx.
Integer.
- ```Ngrid_range```: number of candidate ranges to try in beamforming and at the Rx.
Integer.
- ```angle_grid```: grid of angles to try. Real tensor of shape (Ngrid_angle, )
- ```range_grid```: grid of ranges to try. Real tensor of shape (Ngrid_range, )
- ```pixels_angle```: number of angular elements to consider in the angle-delay map around
the maximum value (see OMP functions for more details). Integers.
- ```pixels_range```: number of range elements to consider in the angle-delay map around
the maximum value (see OMP functions for more details). Integers.
- ```msg_card```: cardinality of the constellation of messages for communication. Integer.
- ```refConst```: constellation to use for communications. Complex tensor of shape
(msg_card,).
- ```varNoise``` or ```N0```: variance of the additive noise at the Rx. Real value.
- ```ant_pos```: true (impaired) antenna positions. Real tensor of shape (numAntennas, 1).
- ```batch_size```: number of samples to use in one batch. Integer.
- ```optimizer```: optimizer to update the parameters to learn.
- ```scheduler```: scheduler to change the learning rate over training iterations. It may
be activated or deactivated depending on the binary flag sch_flag.
- ```loss_comm_fn```: loss function to use for communication training.
- ```eta_weight```: weight to combine sensing and communication loss functions. Real 
value between 0 and 1.
- ```epoch_test_list```: list of epochs at which the network performance should be 
tested during training. List of integers.
- ```target_pfa```: target value of the false alarm probability. Real value.
- ```delta_pfa```: maximum allowable error w.r.t target_pfa. Real value.
- ```thresholds_pfa```: thresholds to distinguish if a measurement is actually a
target. List of real values.
- ```gamma_gospa_train```: value of the gamma parameter in the GOSPA loss. Float
- ```mu_gospa```: value of the mu parameter in the GOSPA loss. Float between 0 (exclusive) and 2.
- ```p_gospa```: value of the p parameter in the GOSPA loss. Integer greater than or equal to 1.
- ```nTestSamples```: number of samples to use during testing.
- ```sch_flag```: flag indicating if a scheduler should be used. Defaul: False.
- ```imp_flag```: flag indicating if impairment learning or dictionary learning are
being considered. True means that impairment learning would be considered.
Defaul: True.
- ```initial_pos```: initial position of the first (reference) antenna element.
- ```device```: device where to perform simulations (```cpu``` or ```cuda```). Default ```cuda```.
- ```rx_feature```: It can be either a matrix of steering vectors or the assumed positions. This depends on the ```dyn_flag```. 
If ```dyn_flag=True```, rx_feature represents the assumed antenna positions and the matrix of steering vectors is
dynamically computed for each batch sample in each iteration. Otherwise, rx_feature is directly such matrix.
- ```dyn_flag```: binary flag that indicates if ```rx_feature``` corresponds to the assumed antenna positions and we 
compute a matrix of steering vectors from those positions or if ```rx_feature``` direcly corresponds to a matrix of
steering vectors for the receiver side.
