# -*- coding: utf-8 -*-
'''
File to load a pretrained model according to the impairment learning approach and test
it as a function of the mean of the angular sector of the targets.
In this file, the matrix of steering vectors to use at the receiver side is dynamically
computed for each new angular sector [theta_min, theta_max].
Note: this file uses the same filename to load the model as it was used to save the 
model in impairment_learning.py. It is required to run that script first before
this one.
'''
from ..lib.simulation_parameters import *

### Load network ###
load_name = 'impairment_maxTargets_' + str(maxTargets_r) + '_maxRange_' + str(np.round(range_max_glob.item(),3))
if impaired_flag:
    load_name += '_hwi'
    if scheduler_flag:
        load_name += '_sch'
else:
    load_name += '_ideal'
load_name += '_lr_' + str(lr_impairment) + '_weight_' + str(weight_loss.item()) + '_fixed_eta'
path_load = 'models/' + load_name + '_model'
checkpoint = torch.load(path_load)
network_impairment.load_state_dict(checkpoint['model'])

#Vectors to save results
pd_target, pfa_target, gospa_target, avg_dist_target = [], [], [], []

#Fix span of the angular sector
theta_span_min_sens_test = torch.tensor(20*np.pi/180,dtype=torch.float32, device=device)
theta_span_max_sens_test = torch.tensor(20*np.pi/180,dtype=torch.float32, device=device)
    
num_means = 9
means_try = 0 + 10*np.pi/180*torch.arange(num_means, device=device)
for k in range(num_means):
    theta_mean_min_sens_test = theta_mean_max_sens_test = means_try[k]

    ### Testing ###
    A_tx, _ = steeringMatrix(angle_grid, angle_grid, network_impairment.pos, lamb)
    pd_temp, pfa_temp, gospa_temp, avg_dist_temp = \
        testSensingFixedPfa(maxTargets_r, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                 theta_span_min_sens_test, theta_span_max_sens_test,
                 range_mean_min_sens_test, range_mean_max_sens_test,
                 range_span_min_sens_test, range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                       K, S, noiseVariance, Delta_f, lamb, ant_pos, A_tx, network_impairment.pos, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=True, device=device)

    #Update vectors to save
    pd_target.append(pd_temp)
    pfa_target.append(pfa_temp)
    gospa_target.append(gospa_temp)
    avg_dist_target.append(avg_dist_temp)

    print(f'**Finished testing iteration {k}/{num_means}**')
    
### Save results ###
file_name = 'impairment_aafo_mean_weight_' + str(weight_loss.item())
np.savez('results/' + file_name, \
         means_try = means_try.cpu(), pd_target = pd_target, \
         pfa_target = pfa_target, gospa_target = gospa_target, \
         avg_dist_target = avg_dist_target)