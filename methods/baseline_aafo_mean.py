# -*- coding: utf-8 -*-
'''
File that tests the sensing performance of the baseline as
a function of the mean of the angular sector. In this file,
the matrix of steering vecotrs is dynamically computed for
each [theta_min, theta_max] angular sector as in impairment
learning.
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_aafo_mean' 
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)

#Vectors to save results
pd_target, pfa_target, gospa_target, avg_dist_target = [], [], [], []

### Testing ###
A_tx, _ = steeringMatrix(angle_grid, angle_grid, assumed_pos, lamb)
#Fix span of the angular sector
theta_span_min_sens_test = torch.tensor(20*np.pi/180,dtype=torch.float32, device=device)
theta_span_max_sens_test = torch.tensor(20*np.pi/180,dtype=torch.float32, device=device)
    
num_means = 9
means_try = 0 + 10*np.pi/180*torch.arange(num_means, device=device)
for k in range(num_means):
    theta_mean_min_sens_test = theta_mean_max_sens_test = means_try[k]
    pd_temp, pfa_temp, gospa_temp, avg_dist_temp = \
        testSensingFixedPfa(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                   theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                   range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                       K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, assumed_pos, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=True, device=device)

    #Update vectors to save
    pd_target.append(pd_temp)
    pfa_target.append(pfa_temp)
    gospa_target.append(gospa_temp)
    avg_dist_target.append(avg_dist_temp)

### Save results ###
np.savez('results/' + file_name, \
         means_try = means_try.cpu(), pd_target = pd_target, \
         pfa_target = pfa_target, gospa_target = gospa_target, \
         avg_dist_target = avg_dist_target)