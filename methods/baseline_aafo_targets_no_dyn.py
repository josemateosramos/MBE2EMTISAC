# -*- coding: utf-8 -*-
'''
File that tests the sensing performance of the baseline as a 
function of targets. The steering matrix used to estimate the
number of targets and their position always assumes the 
angular sector [-pi/2, pi/2] like dictionary learning.
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_no_dyn_aafo_targets_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)

#Vectors to save results
num_targets, pd_target, pfa_target, gospa_target, avg_dist_target = [], [], [], [], []

### Testing ###
A_tx, _ = steeringMatrix(angle_grid, angle_grid, assumed_pos, lamb)
A_rx = torch.clone(A_tx)
    
for targets in range(1,maxTargets_r+1):
    num_targets.append(targets)
    pd_temp, pfa_temp, gospa_temp, avg_dist_temp = \
        testSensingFixedPfa(targets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                   theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                   range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                       K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, A_rx, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=False, device=device)

    #Update vectors to save
    pd_target.append(pd_temp)
    pfa_target.append(pfa_temp)
    gospa_target.append(gospa_temp)
    avg_dist_target.append(avg_dist_temp)

### Save results ###
np.savez('results/' + file_name, \
         num_targets = num_targets, pd_target = pd_target, \
         pfa_target = pfa_target, gospa_target = gospa_target, \
         avg_dist_target = avg_dist_target)