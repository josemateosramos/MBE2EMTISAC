# -*- coding: utf-8 -*-
'''
File to load a pretrained model following the dictionary learning approach and
test it as a function of the number of targets.
Important: the name to load the model follows the same name of the dictionary_learning.py
file, it is important that you run that script before this one.
'''
from ..lib.simulation_parameters import *

#Vectors to save results
num_targets, pd_target, pfa_target, gospa_target, avg_dist_target = [], [], [], [], []

for targets in range(1,maxTargets_r+1):
    num_targets.append(targets)
    ### Load network based on the file name given in dictionary_learning.py ###
    file_name = 'dictionary_maxTargets_' + str(targets) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
    if impaired_flag:
        file_name += '_hwi'
        if scheduler_flag:
            file_name += '_sch'
    else:
        file_name += '_ideal'
    file_name += '_lr_' + str(lr_dictionary) + '_weight_' + str(weight_loss.item())
    path_load = 'models/' + file_name + '_model'
    checkpoint = torch.load(path_load)
    network_dictionary.load_state_dict(checkpoint['model'])

    ### Testing ###
    pd_temp, pfa_temp, gospa_temp, avg_dist_temp = \
        testSensingFixedPfa(targets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                       K, S, noiseVariance, Delta_f, lamb, ant_pos, network_dictionary.A, network_dictionary.A, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=False, device=device)

    #Update vectors to save
    pd_target.append(pd_temp)
    pfa_target.append(pfa_temp)
    gospa_target.append(gospa_temp)
    avg_dist_target.append(avg_dist_temp)

    print(f'**Finished testing iteration {targets}/{maxTargets_r}**')

### Save results ###
file_name = 'dictionary_aafo_targets_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
np.savez('results/' + file_name, \
         num_targets = num_targets, pd_target = pd_target, \
         pfa_target = pfa_target, gospa_target = gospa_target, \
         avg_dist_target = avg_dist_target)