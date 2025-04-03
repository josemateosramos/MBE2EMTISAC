# -*- coding: utf-8 -*-
'''
File that tests the ISAC trade-off performance of a pretrained model according to the 
impairment learning approach.
In this file, the matrix of steering vectors to use at the receiver side is dynamically
computed for each new angular sector [theta_min, theta_max].
'''
from ..lib.simulation_parameters import *

load_name = 'impairment_maxTargets_' + str(maxTargets_r) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    load_name += '_hwi'
    if scheduler_flag:
        load_name += '_sch'
else:
    load_name += '_ideal'
load_name += '_lr_' + str(lr_impairment) + '_weight_' + str(weight_loss.item()) + '_fixed_eta'
### Load network ###
path_load = 'models/' + load_name + '_model'
checkpoint = torch.load(path_load)
network_impairment.load_state_dict(checkpoint['model'])
print(load_name, flush=True)

#Update dictionary matrix for the baseline
A_tx, _ = steeringMatrix(angle_grid, angle_grid, network_impairment.pos, lamb)
A_rx = torch.clone(A_tx)

#Test sensing performance
ser_isac, pd_isac, pfa_isac, gospa_pos_isac, avg_dist_isac = \
    testISAC(maxTargets_r, maxPaths_c, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
             theta_span_min_sens_test, theta_span_max_sens_test, theta_mean_min_comm_test, theta_mean_max_comm_test, 
             theta_span_min_comm_test, theta_span_max_comm_test, range_mean_min_sens_test, range_mean_max_sens_test, 
             range_span_min_sens_test, range_span_max_sens_test, range_mean_min_comm_test, range_mean_max_comm_test, 
             range_span_min_comm_test, range_span_max_comm_test,
             Ngrid_angle, Ngrid_range, K, S, Rcp, noiseVariance, Delta_f, lamb, ant_pos, 
             A_tx, A_rx, refConst, target_pfa, delta_pfa, thresholds_pfa, eta, phi, gamma_gospa_test, mu_gospa, p_gospa, 
             batch_size, nTestSamples, device, dyn_flag=True)

### Save results ###
file_name = 'isac_impairment_learning'
np.savez('results/' + file_name, \
         ser_isac = ser_isac, pd_isac = pd_isac, pfa_isac = pfa_isac, \
            gospa_pos_isac = gospa_pos_isac, avg_dist_isac = avg_dist_isac)