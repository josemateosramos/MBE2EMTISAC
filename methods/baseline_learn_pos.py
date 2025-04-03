# -*- coding: utf-8 -*-
'''
File to perform model-based greedy calibration of the antenna array and sensing testing.
'''
from ..lib.simulation_parameters import *

num_iterations = 10

file_name = 'baseline_learn_pos_maxTargets_' + str(maxTargets_r) + '_iter_' + str(num_iterations) + '_P_' + str(P_power.item())
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)

print(f'The considered antenna impairments are: \n\t{ant_pos}')
#Set the grid of spacings to try with each antenna element
Ngrid_spacing = 100 
spacings_try = torch.linspace(lamb/4.0, 3*lamb/4.0, Ngrid_spacing)

#Initialize estimted positions the same as the assumed positions
est_ant_pos = torch.clone(assumed_pos)

### Training ###
est_ant_pos, gospa_loss = \
    baselineLearningSeveralTargets(num_iterations, maxTargets_r, P_power, mean_rcs, K, S, Delta_f, angle_grid, 
                    Ngrid_range, pixels_angle, pixels_range, 
                    assumed_pos, ant_pos, est_ant_pos, spacings_try, noiseVariance, 
                    lamb, msg_card, refConst,
                    theta_mean_min, theta_mean_max,
                    theta_span_min, theta_span_max,
                    range_mean_min, range_mean_max,
                    range_span_min, range_span_max, batch_size, 
                    gamma_gospa_train, mu_gospa, p_gospa, device=device)

### Testing ###
#Update thresholds depending on the maximum number of targets in the scenario
if maxTargets_r == 1:
    thresholds_roc_impairment = torch.logspace(np.log10(3.2e-4), np.log10(6.8e-4), numThresholds, device=device)
if maxTargets_r == 2:
    thresholds_roc_impairment = torch.logspace(np.log10(3.05e-4), np.log10(3e-3), numThresholds, device=device)
if maxTargets_r == 3:
    thresholds_roc_impairment = torch.logspace(np.log10(2.7e-4), np.log10(4.8e-3), numThresholds, device=device)
if maxTargets_r == 4:
    thresholds_roc_impairment = torch.logspace(np.log10(2.7e-4), np.log10(6.7e-3), numThresholds, device=device)
if maxTargets_r == 5:
    thresholds_roc_impairment = torch.logspace(np.log10(2.7e-4), np.log10(7e-3), numThresholds, device=device)
# Update matrix (K x Ngrid_angle) of steering vectors that will be used later for learning
A_tx, _ = steeringMatrix(angle_grid, angle_grid, est_ant_pos, lamb)

P_power      = torch.tensor(1.0, dtype=torch.float32, device=device)                  #Transmitter power [W]
nTestSamples = int(1e5)
numTestIt            = nTestSamples // batch_size     
nTestSamples         = numTestIt*batch_size
pd_roc, pfa_roc, gospa_roc, avg_dist_roc = \
    testSensingROC(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                   K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, est_ant_pos, refConst,
                   thresholds_roc_impairment, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples, 
                   dyn_flag=True, device=device)

### Save results ###
np.savez('results/' + file_name, \
        est_ant_pos = est_ant_pos.cpu().detach().numpy(), gospa_loss = gospa_loss, \
        pd_roc = pd_roc, pfa_roc = pfa_roc, gospa_roc = gospa_roc, avg_dist_roc = avg_dist_roc
)