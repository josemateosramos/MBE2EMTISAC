# -*- coding: utf-8 -*-
'''
File to load a pretrained model according to the impairment learning approach and test
it as a function of the number of targets.
In this file, the matrix of steering vectors to use at the receiver side always assumes
an angular sector [-pi/2,pi/2] to have a fair comparison with dictionary learning.
Note: this file uses the same filename to load the model as it was used to save the 
model in impairment_learning.py. It is required to run that script first before
this one.
'''
from ..lib.simulation_parameters import *

#Vectors to save results
num_targets, pd_target, pfa_target, gospa_target, avg_dist_target = [], [], [], [], []

for targets in range(1,maxTargets_r+1):
    num_targets.append(targets)
    ### Load network ###
    file_name = 'impairment_maxTargets_' + str(targets) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
    if impaired_flag:
        file_name += '_hwi'
        if scheduler_flag:
            file_name += '_sch'
    else:
        file_name += '_ideal'
    file_name += '_lr_' + str(lr_impairment) + '_weight_' + str(weight_loss.item()) + '_fixed_eta'
    path_load = 'models/' + file_name + '_model'
    checkpoint = torch.load(path_load)
    network_impairment.load_state_dict(checkpoint['model'])

    ### Testing ###
    A_tx, _ = steeringMatrix(angle_grid, angle_grid, network_impairment.pos, lamb)
    A_rx = torch.clone(A_tx)
    pd_temp, pfa_temp, gospa_temp, avg_dist_temp = \
        testSensingFixedPfa(targets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                       K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, A_rx, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=False, device=device)

    #Update vectors to save
    pd_target.append(pd_temp)
    pfa_target.append(pfa_temp)
    gospa_target.append(gospa_temp)
    avg_dist_target.append(avg_dist_temp)

    print(f'**Finished testing iteration {targets}/{maxTargets_r}**')
    
### Save results ###
file_name = 'impairment_no_dyn_aafo_targets_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    file_name += '_hwi'
else:
    file_name += '_ideal'
print(file_name, flush=True)
np.savez('results/' + file_name, \
         num_targets = num_targets, pd_target = pd_target, \
         pfa_target = pfa_target, gospa_target = gospa_target, \
         avg_dist_target = avg_dist_target)