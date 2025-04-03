# -*- coding: utf-8 -*-
'''File to test the sensing performance of the baseline'''
from ..lib.simulation_parameters import *

file_name = 'baseline_maxTargets_' + str(maxTargets_r) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)

### Testing ###
# Change thresholds depending on the maximum number of targets
if maxTargets_r == 1:
    thresholds_roc_baseline   = torch.logspace(np.log10(2.50e-4), np.log10(1e-3), numThresholds, device=device)
if maxTargets_r == 2:
    thresholds_roc_baseline   = torch.logspace(np.log10(2.50e-4), np.log10(3e-3), numThresholds, device=device)
if maxTargets_r == 3:
    thresholds_roc_baseline   = torch.logspace(np.log10(2.50e-4), np.log10(7e-3), numThresholds, device=device)
if maxTargets_r == 4:
    thresholds_roc_baseline   = torch.logspace(np.log10(2.50e-4), np.log10(1e-2), numThresholds, device=device)
if maxTargets_r == 5:
    thresholds_roc_baseline   = torch.logspace(np.log10(2.50e-4), np.log10(4.7e-2), numThresholds, device=device)
# Update matrix (K x Ngrid_angle) of steering vectors that will be used later for learning
A_tx, _ = steeringMatrix(angle_grid, angle_grid, assumed_pos, lamb)
pd_roc, pfa_roc, gospa_roc, avg_dist_roc = \
    testSensingROC(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                   theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                   range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, 
                   Ngrid_angle, Ngrid_range,
                   K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, assumed_pos, refConst,
                   thresholds_roc_baseline, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples, 
                   dyn_flag=True, device=device)

### Save results ###
np.savez('results/' + file_name, \
         pd_roc = pd_roc, pfa_roc = pfa_roc, gospa_roc = gospa_roc, avg_dist_roc = avg_dist_roc)
