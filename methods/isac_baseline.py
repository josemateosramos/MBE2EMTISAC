# -*- coding: utf-8 -*-
'''
File that tests the ISAC trade-off performance of the baseline.
In this file, the matrix of steering vectors to use at the receiver side is dynamically
computed for each new angular sector [theta_min, theta_max].
'''
from ..lib.simulation_parameters import *

file_name = 'isac_baseline'
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)

thresholds_pfa = torch.linspace(5.958e-4, 5.970e-4, 3, device=device)  #Detection threshold for radar receiver
#Update dictionary matrix for the baseline
A_tx, _ = steeringMatrix(angle_grid, angle_grid, assumed_pos, lamb)

#Test sensing performance
ser_isac, pd_isac, pfa_isac, gospa_pos_isac, avg_dist_isac = \
    testISAC(maxTargets_r, maxPaths_c, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
             theta_span_min_sens_test, theta_span_max_sens_test, theta_mean_min_comm_test, theta_mean_max_comm_test, 
             theta_span_min_comm_test, theta_span_max_comm_test, range_mean_min_sens_test, range_mean_max_sens_test, 
             range_span_min_sens_test, range_span_max_sens_test, range_mean_min_comm_test, range_mean_max_comm_test, 
             range_span_min_comm_test, range_span_max_comm_test,
             Ngrid_angle, Ngrid_range, K, S, Rcp, noiseVariance, Delta_f, lamb, ant_pos, 
             A_tx, assumed_pos, refConst, target_pfa, delta_pfa, thresholds_pfa, eta, phi, gamma_gospa_test, mu_gospa, p_gospa, 
             batch_size, nTestSamples, dyn_flag=True, device=device)

### Save results ###
np.savez('results/' + file_name, \
         ser_isac = ser_isac, pd_isac = pd_isac, pfa_isac = pfa_isac, \
         gospa_pos_isac = gospa_pos_isac, avg_dist_isac = avg_dist_isac)