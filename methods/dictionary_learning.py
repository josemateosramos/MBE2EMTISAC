# -*- coding: utf-8 -*-
'''
File to perform training and testing of the dictionary learning approach.
'''
from ..lib.simulation_parameters import *

file_name = 'dictionary_maxTargets_' + str(maxTargets_r) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
    if scheduler_flag:
        file_name += '_sch'
else:
    file_name += '_ideal'
file_name += '_lr_' + str(lr_dictionary) + '_weight_' + str(weight_loss.item())
print(file_name, flush=True)

### Training ###
loss_gospa_np, loss_comm_np, loss_avg_np, num_iterations, pd_inter, pfa_inter, gospa_pos_inter, avg_dist_inter, snr_inter, ser_inter = \
    trainNetworkFixedEta(network_dictionary, train_it, maxTargets_r, maxPaths_c, P_power, mean_rcs, Rcp,
                    theta_mean_min, theta_mean_max, theta_span_min, theta_span_max, 
                    range_mean_min, range_mean_max, range_span_min, range_span_max, 
                    theta_mean_min, theta_mean_max, theta_span_min, theta_span_max, 
                    range_mean_min_comm, range_mean_max_comm,
                    range_span_min_comm, range_span_max_comm, 
                    K, S, Delta_f, lamb, Ngrid_angle, Ngrid_range, 
                    angle_grid, pixels_angle, pixels_range, msg_card, 
                    refConst, noiseVariance, ant_pos, batch_size, optimizerDictionary, 
                    scheduler_dictionary, loss_comm_fn, weight_loss, test_iterations_list, 
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
                    sch_flag=scheduler_flag, imp_flag = False, initial_pos = None, device=device)

### Save network ###
saveNetwork('models/' + file_name + '_model', network_dictionary, optimizerDictionary)

### Testing ###
if maxTargets_r == 1:
    thresholds_roc_dictionary = torch.logspace(np.log10(1e-2), np.log10(1e-1), numThresholds, device=device)
if maxTargets_r == 2:
    thresholds_roc_dictionary = torch.logspace(np.log10(4e-2), np.log10(6e-1), numThresholds, device=device)
if maxTargets_r == 3:
    thresholds_roc_dictionary = torch.logspace(np.log10(1e-2), np.log10(1.6e-1), numThresholds, device=device)
if maxTargets_r == 4:
    thresholds_roc_dictionary = torch.logspace(np.log10(1e-2), np.log10(2e-1), numThresholds, device=device)
if maxTargets_r == 5:
    thresholds_roc_dictionary = torch.logspace(np.log10(1e-2), np.log10(1), numThresholds, device=device)
pd_roc, pfa_roc, gospa_roc, avg_dist_roc = \
    testSensingROC(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                   K, S, noiseVariance, Delta_f, lamb, ant_pos, network_dictionary.A, network_dictionary.A, refConst,
                   thresholds_roc_dictionary, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples, 
                   dyn_flag=False, device=device)

# ISAC testing
ser_isac, pd_isac, pfa_isac, gospa_pos_isac, avg_dist_isac = \
    testISAC(maxTargets_r, maxPaths_c, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
             theta_span_min_sens_test, theta_span_max_sens_test, theta_mean_min_comm_test, theta_mean_max_comm_test, 
             theta_span_min_comm_test, theta_span_max_comm_test, range_mean_min_sens_test, range_mean_max_sens_test, 
             range_span_min_sens_test, range_span_max_sens_test, range_mean_min_comm_test, range_mean_max_comm_test, 
             range_span_min_comm_test, range_span_max_comm_test,
             Ngrid_angle, Ngrid_range, K, S, Rcp, noiseVariance, Delta_f, lamb, ant_pos, 
             network_dictionary.A, network_dictionary.A, refConst, target_pfa, delta_pfa, 
             thresholds_pfa, eta, phi, gamma_gospa_test, mu_gospa, p_gospa, 
             batch_size, nTestSamples, dyn_flag=False, device=device)

### Save results ###
np.savez('results/' + file_name, \
         loss_pos = loss_gospa_np, loss_comm = loss_comm_np, loss_avg_dist = loss_avg_np, num_iterations = num_iterations, \
         pd_inter = pd_inter, pfa_inter = pfa_inter, gospa_pos_inter = gospa_pos_inter, avg_dist_inter = avg_dist_inter, \
         snr_inter = snr_inter, ser_inter = ser_inter, \
         pd_roc = pd_roc, pfa_roc = pfa_roc, gospa_roc = gospa_roc, avg_dist_roc = avg_dist_roc, \
         pd_isac = pd_isac, pfa_isac = pfa_isac, ser_isac = ser_isac, gospa_pos_isac = gospa_pos_isac, \
         avg_dist_isac = avg_dist_isac)