# -*- coding: utf-8 -*-
'''
File to perform training and testing following the impairment learning approach.
'''
from ..lib.simulation_parameters import *

file_name = 'impairment_maxTargets_' + str(maxTargets_r) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
    if scheduler_flag:
        file_name += '_sch'
else:
    file_name += '_ideal'
file_name += '_lr_' + str(lr_impairment) + '_weight_' + str(weight_loss.item()) + '_fixed_eta'
print(file_name, flush=True)

### Training ###
loss_gospa_np, loss_comm_np, loss_avg_np, num_iterations, pd_inter, pfa_inter, gospa_pos_inter, avg_dist_inter, snr_inter, ser_inter = \
    trainNetworkFixedEta(network_impairment, train_it, maxTargets_r, maxPaths_c, P_power, mean_rcs, Rcp,
                    theta_mean_min, theta_mean_max,
                    theta_span_min, theta_span_max, 
                    range_mean_min, range_mean_max,
                    range_span_min, range_span_max, 
                    theta_mean_min, theta_mean_max,
                    theta_span_min, theta_span_max, 
                    range_mean_min_comm, range_mean_max_comm,
                    range_span_min_comm, range_span_max_comm, 
                    K, S, Delta_f, lamb, Ngrid_angle, Ngrid_range, 
                    angle_grid, pixels_angle, pixels_range, msg_card, 
                    refConst, noiseVariance, ant_pos, batch_size, optimizerImpairment, 
                    scheduler_impairment, loss_comm_fn, weight_loss, test_iterations_list, 
                    theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test, 
                    range_mean_min_sens_test, range_mean_max_sens_test, 
                    range_span_min_sens_test, range_span_max_sens_test,
                    theta_mean_min_comm_test, theta_mean_max_comm_test, 
                    theta_span_min_comm_test, theta_span_max_comm_test, 
                    range_mean_min_comm_test, range_mean_max_comm_test, 
                    range_span_min_comm_test, range_span_max_comm_test,
                    target_pfa, delta_pfa, 
                    thresholds_pfa, gamma_gospa_train, mu_gospa, p_gospa, nTestSamples, gamma_gospa_test, 
                    sch_flag=scheduler_flag, imp_flag = True, initial_pos = initial_pos_ula, device=device)

### Save network ###
saveNetwork('models/' + file_name + '_model', network_impairment, optimizerImpairment)

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
A_tx, _ = steeringMatrix(angle_grid, angle_grid, network_impairment.pos, lamb)

pd_roc, pfa_roc, gospa_roc, avg_dist_roc = \
    testSensingROC(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                   K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, network_impairment.pos, refConst,
                   thresholds_roc_impairment, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples, 
                   dyn_flag=True, device=device)

# Test comm. performance as a function of the SNR
snr_comm, ser_comm = [], []
num_ranges_try = 10
ranges_try = torch.logspace(np.log10(100), np.log10(1e3), num_ranges_try)
for k in range(num_ranges_try):
    print(f'Testing range {k} out of {num_ranges_try}')
    range_max = ranges_try[k]
    range_min_mean_test = range_max_mean_test = (range_min_glob_comm+range_max)/2.0
    range_min_span_test = range_max_span_test = range_max-range_min_glob_comm
    snr_temp, ser_temp = testCommunication(maxPaths_c, P_power, mean_rcs, 
                                            theta_mean_min_comm_test, theta_mean_max_comm_test, 
                                            theta_span_min_comm_test, theta_span_max_comm_test, 
                                            range_min_mean_test, range_max_mean_test, 
                                            range_min_span_test, range_max_span_test,
                                            Ngrid_angle, K, S, noiseVariance, 
                                            Delta_f, Rcp, lamb, ant_pos, A_tx, refConst,
                                            batch_size, nTestSamples_comm, device)
    snr_comm.append(snr_temp.item())
    ser_comm.append(ser_temp.item())

# Test ISAC performance
ser_isac, pd_isac, pfa_isac, gospa_pos_isac, avg_dist_isac = \
    testISAC(maxTargets_r, maxPaths_c, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
             theta_span_min_sens_test, theta_span_max_sens_test, theta_mean_min_comm_test, theta_mean_max_comm_test, 
             theta_span_min_comm_test, theta_span_max_comm_test, range_mean_min_sens_test, range_mean_max_sens_test, 
             range_span_min_sens_test, range_span_max_sens_test, range_mean_min_comm_test, range_mean_max_comm_test, 
             range_span_min_comm_test, range_span_max_comm_test,
             Ngrid_angle, Ngrid_range, K, S, Rcp, noiseVariance, Delta_f, lamb, ant_pos, 
             A_tx, network_impairment.pos, refConst, target_pfa, delta_pfa, thresholds_pfa, eta, phi, gamma_gospa_test, mu_gospa, p_gospa, 
             batch_size, nTestSamples, dyn_flag=True, device=device)

### Save results ###
np.savez('results/' + file_name, \
         loss_pos = loss_gospa_np, loss_avg_dist = loss_avg_np, num_iterations = num_iterations, \
         loss_comm = loss_comm_np, snr_inter = snr_inter, ser_inter = ser_inter, \
         pd_inter = pd_inter, pfa_inter = pfa_inter, gospa_pos_inter = gospa_pos_inter, avg_dist_inter = avg_dist_inter, \
         pd_roc = pd_roc, pfa_roc = pfa_roc, gospa_roc = gospa_roc, avg_dist_roc = avg_dist_roc, \
         pd_isac = pd_isac, pfa_isac = pfa_isac, ser_isac = ser_isac, gospa_pos_isac = gospa_pos_isac, \
         avg_dist_isac = avg_dist_isac, snr_comm = snr_comm, ser_comm = ser_comm)