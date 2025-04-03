# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import argparse

from ..lib.functions import *
from ..lib.learnable_parameters import *

#Fix seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)
#Try to make results reproducible
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
#To use torch.use_deterministic_algorithms(True), we need the following sentences
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" #This will increase memory, more info: https://docs.nvidia.com/cuda/cublas/#results-reproducibility

#Computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######## Simulation Parameters selected by user ########
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--maxTargets", type=int, default=5,
                    help="Integer controlling the maximum number of targets in the scene.")
parser.add_argument("-i", "--impaired", type=int, default=1,
                    help="Binary flag controlling whether we consider impairments (0 for no impairments and 1 otherwise).")
parser.add_argument("-n", "--number", type=int, default=10,
                    help="Number controling the seed of the simulation.")
parser.add_argument("-w", "--weight", type=int, default=0,
                    help="Integer controlling the weight of sensing and communication losses.")
args = parser.parse_args()

######## Simulation Parameters ########
save_path      = 'results/'                                                             #Directory to save results
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
impaired_flag  = args.impaired                                                          #True to include impairments, False otherwise
maxTargets_r   = args.maxTargets                                                        #Maximum number of sensing targets in the scenario
maxPaths_c     = 6                                                                      #Maximum number of paths in the communication channel (including LOS)
P_power        = torch.tensor(1.0, dtype=torch.float32, device=device)                  #Transmitter power [W]
mean_rcs       = torch.tensor(1.0, dtype=torch.float32, device=device)                  #Mean target radar cross section (RCS) [sq. m]
K              = 64                                                                     #Number of antennas
S              = 256                                                                    #Number of subcarriers
Delta_f        = torch.tensor(240e3, dtype=torch.float32, device=device)                #Spacing between subcarriers
fc             = torch.tensor(60e9, dtype=torch.float32, device=device)                 #Carrier frequency [Hz]
lamb           = 3e8 / fc                                                               #Wavelength [m]
boltzmann      = torch.tensor(1.38e-23, device=device)                                  #Boltzmann constant [J/K]
sys_temp       = torch.tensor(290, device=device)                                       #System temperature [K]
N0             = 10*torch.log10(boltzmann*sys_temp/1e-3)                                #Noise PSD [dBm/Hz]
noiseFigure    = torch.tensor(8, device=device)                                         #Noise Figure [dB]
noiseVariance  = 10**((N0+noiseFigure)/10)*1e-3*S*Delta_f                               #Receiver noise variance [W]
msg_card       = 4                                                                      #Comm. constellation size
target_pfa     = 1e-2                                                                   #Target false alarm prob. for ISAC
delta_pfa      = 1e-4                                                                   #Max deviation from target_pfa
range_min_glob = torch.tensor(10, dtype=torch.float32, device=device)                   #Minimum possible considered range of the targets
range_max_glob = torch.tensor(43.75, dtype=torch.float32, device=device)                #Maximum possible considered range of the targets
range_max_glob_comm = torch.tensor(200, dtype=torch.float32, device=device)
range_min_glob_comm = torch.tensor(10, dtype=torch.float32, device=device)
angle_min_glob = torch.tensor(-np.pi/2, dtype=torch.float32, device=device)             #Minimum possible considered angle of the targets
angle_max_glob = torch.tensor(np.pi/2, dtype=torch.float32, device=device)              #Maximum possible considered angle of the targets
Ngrid_angle    = 720                                                                    #Number of points in the oversampled grid of angles
Ngrid_range    = 200                                                                    #Number of points in the oversampled grid of ranges
refConst       = MPSK(msg_card, np.pi/4, device=device)                                 #Refence constellation (PSK)
angle_res      = 2/K                                                                    #Angle resolution (roughly) [rad]
range_res      = 3e8 / (2*S*Delta_f)                                                    #Range resolution (roughly) [m]
pixels_angle   = int(angle_res / (np.pi/Ngrid_angle))                                   #Pixels corresponding to the angle resolution
pixels_range   = int(range_res / ((range_max_glob - range_min_glob)/Ngrid_range))       #Pixels corresponding to the range resolution
Tcp            = 0.07/Delta_f                                                           #Cyclic prefix time [s]
Rcp            = Tcp*3e8                                                                #Equivalent range corresponding to Tcp [m]
numThresholds  = 20                                                                     #Number of thresholds to test when computing the ROC curve
#Antenna positions
initial_pos_ula = -((K-1)/2) * (lamb/2)     #Array approx. centered around 0
'''
# Generate ordered and impaired antenna positions
std_d = lamb/15                         #Antenna displacement standard deviation
ant_pos = generateImpairedPos(std_d, lamb, K, initial_pos_ula)
#We should always check that the final positions are ordered
'''
#Here is an example of impairments that produce ordered positions
ant_pos = torch.tensor([[-0.0787499994, -0.0765224472, -0.0742076337, -0.0719831958,
        -0.0699155182, -0.0674813092, -0.0653027520, -0.0629738495,
        -0.0595979244, -0.0573467575, -0.0545117259, -0.0520973392,
        -0.0494385064, -0.0471602455, -0.0447811335, -0.0427645817,
        -0.0403477773, -0.0375711881, -0.0346975401, -0.0318642333,
        -0.0289920028, -0.0261332411, -0.0236553177, -0.0211114753,
        -0.0185554437, -0.0160367228, -0.0134548638, -0.0106370309,
        -0.0080185849, -0.0054478869, -0.0030604769, -0.0006783467,
         0.0017306840,  0.0046736933,  0.0075728190,  0.0096513135,
         0.0120065007,  0.0137796346,  0.0159098431,  0.0177294966,
         0.0202406179,  0.0223804526,  0.0245860349,  0.0274107363,
         0.0300984588,  0.0322267525,  0.0347764231,  0.0369123220,
         0.0393733270,  0.0420691930,  0.0441824421,  0.0470172837,
         0.0493256450,  0.0520207472,  0.0540654734,  0.0561210141,
         0.0591074117,  0.0620236322,  0.0637644902,  0.0666820034,
         0.0693642274,  0.0718709156,  0.0738426670,  0.0771570578]], dtype=torch.float32, device = device).T
#We allow for training of random [theta_min, theta_max] and [range_min, range_max]
theta_mean_min = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
theta_span_min = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
theta_span_max = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min = (range_min_glob + range_max_glob)/2.0           #This fixes the target range sector to [range_min_glob, range_max_glob]
range_mean_max = (range_min_glob + range_max_glob)/2.0
range_span_min = range_max_glob - range_min_glob
range_span_max = range_max_glob - range_min_glob
range_mean_min_comm = (range_max_glob_comm + range_min_glob_comm)/2.0
range_mean_max_comm = (range_max_glob_comm + range_min_glob_comm)/2.0
range_span_min_comm = range_max_glob_comm - range_min_glob_comm
range_span_max_comm = range_max_glob_comm - range_min_glob_comm
#Create an oversampled dictionary of possible target ranges and angles
angle_grid = torch.linspace(angle_min_glob, angle_max_glob, Ngrid_angle, device=device)
range_grid = torch.linspace(range_min_glob, range_max_glob, Ngrid_range, device=device)
#Testing parameters after training
batch_size           = 800
nTestSamples         = int(1e6)                      #This value will be slightly changed in the next lines to be a multiple of batch_size
nTestSamples_comm    = int(1e6)                      
numTestIt            = nTestSamples // batch_size     
numTestIt_comm       = nTestSamples_comm // batch_size   
nTestSamples         = numTestIt*batch_size       
nTestSamples_comm    = numTestIt_comm*batch_size       
theta_mean_min_sens_test = torch.tensor(-30*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max_sens_test = torch.tensor(-30*np.pi/180, dtype=torch.float32, device=device)
theta_span_min_sens_test = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
theta_span_max_sens_test = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min_sens_test = (range_min_glob + range_max_glob)/2.0
range_mean_max_sens_test = (range_min_glob + range_max_glob)/2.0
range_span_min_sens_test = range_max_glob - range_min_glob
range_span_max_sens_test = range_max_glob - range_min_glob
theta_mean_min_comm_test = torch.tensor(50*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max_comm_test = torch.tensor(50*np.pi/180, dtype=torch.float32, device=device)
theta_span_min_comm_test = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
theta_span_max_comm_test = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min_comm_test = (range_max_glob_comm + range_min_glob_comm)/2.0
range_mean_max_comm_test = (range_max_glob_comm + range_min_glob_comm)/2.0
range_span_min_comm_test = range_max_glob_comm - range_min_glob_comm
range_span_max_comm_test = range_max_glob_comm - range_min_glob_comm
#ISAC Trade-off beam parameters
ISAC_grid_pts  = 8                                                          #Number of points for the ISAC trade-off beam
eta = torch.logspace(-3,0, ISAC_grid_pts, device=device)
phi = torch.tensor([0, np.pi], device=device)

######## Other Parameters computed from simulation parameters ########
#Assumed inter-antenna spacing
if impaired_flag:
    lambda_2_spacing = torch.cat((torch.tensor([initial_pos_ula], device=device), lamb/2 * torch.ones((K-1,), device=device)))
    assumed_pos = torch.cumsum(lambda_2_spacing, dim=0)
else: #Known impairments
    assumed_pos = torch.clone(ant_pos)                 
#Thresholds for ISAC plots
thresholds_pfa = torch.linspace(9e-4, 1.1e-3, 3, device=device)  #Detection threshold for radar receiver

######## NN-related parameters ########
initial_pos_impairment = torch.clone(assumed_pos)       #These already include the impairments depending on the input flag -i
initial_pos_dictionary = torch.clone(assumed_pos)

network_impairment = PertNet(initial_pos_impairment.cpu()).to(device)
initial_matrix, _  = steeringMatrix(angle_grid, angle_grid, initial_pos_dictionary, lamb)
network_dictionary = MatrixNet(initial_matrix.cpu()).to(device)
#Define different learning rates and optimizers for the case of sequential training
lr_impairment        = 1e-5
lr_dictionary        = 2e-1
optimizerImpairment  = torch.optim.Adam(list(network_impairment.parameters()), lr = lr_impairment)
optimizerDictionary  = torch.optim.Adam(list(network_dictionary.parameters()), lr = lr_dictionary)
scheduler_flag       = False          #Flag to use scheduler if True or skip it if False. This was False in the paper results.
scheduler_impairment = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerImpairment, factor=0.5,
                                                                    patience=1000, threshold=0.0001)
scheduler_dictionary = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerDictionary, factor=0.5,
                                                                    patience=1000, threshold=0.0001)

train_it             = int(1.5e4)                              #Number of training iterations
test_iterations_list = np.round(np.logspace(0,np.log10(train_it), 10)).astype(int) #Iterations where to test the network performance during training   
weight_list          = np.linspace(0,1,5)
weight_loss          = torch.tensor(weight_list[args.weight], dtype=torch.float32, device=device)
loss_comm_fn         = nn.CrossEntropyLoss()

#GOSPA-related parameters
gamma_gospa_train = torch.tensor(1e5, dtype=torch.float32, device=device)                       #Not using inf to avoid issues
gamma_gospa_test  = range_mean_max_sens_test-range_mean_min_sens_test+range_span_max_sens_test  #Range difference in the broader case of [theta_mean_min-span_max, theta_mean_max+span_max]
mu_gospa          = torch.tensor(2, dtype=torch.float32, device=device)
p_gospa           = torch.tensor(2, dtype=torch.float32, device=device)