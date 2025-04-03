# -*- coding: utf-8 -*-
'''
File to define the functions that will be used in the methods.
Disclaimer: some of the inputs of the functions have the same name and function.
These are commented at the end of the file and in the README_LIB.md file in the lib/ folder.
If the input parameter is not commented below the function definition, it is very likely 
that it is defined in the two aforementioned places.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import math

#####################
# Generic functions #
#####################

def generateImpairedPos(std_d, lamb, K, initial_pos, device='cuda'):
    '''
    Function that generates impaired antenna positions. It generates random antenna
    spacing as d~N(lamb/2,std_d**2), and then positions as the cumulative sum of 
    spacings starting at initial_pos, i.e., p=cumsum(cat(initial_pos,d))
    Inputs:
        - std_d: standard deviation of the random perturbation.
        - lamb: carrier wavelength.
        - K: number of antennas in the ULA.
        - initial_pos: position of the first element of the antenna array.
    Output:
        - random_pos: tensor that contains the perturbed inter-antenna positions.
        Shape: (K, 1)
    '''
    d_random = torch.maximum(lamb/2 + std_d*torch.randn(K-1,1, device=device), torch.tensor(0,device=device))       
    temp = torch.cat((torch.tensor([[initial_pos]], device=device), d_random), dim=0)
    random_pos = torch.cumsum(temp, dim=0)
    
    return random_pos

def generateUniform(min_val, max_val, out_shape, device='cuda'):
    '''Function that generates a tensor of shape out_shape with uniformly 
    distributed values between min_val and max_val.
    Inputs:
        - min_val: minimum value of the uniform distribution. 
        - max_val: maximum value of the uniform distribution. It should have the
        same shape as min_val.
        - out_shape: output shape. List of integers.
    Output
        - output: tensor whose entries are uniformly distributed between min_val 
        and max_val.
    '''
    return torch.rand(out_shape, device=device) * (max_val - min_val) + min_val
    

def generateInterval(mean_min, mean_max, span_min, span_max, batch_size=1, device='cuda'):
    '''
    Function that creates an interval [minimum, maximum] as
    [minimum, maximum] = mean + [span/2, -span/2], where
    mean~U(mean_min, mean_max), and span~U(span_min, span_max).
    Inputs:
        - mean_min: minimum value of mean. Float.
        - mean_max: maximum value of mean. Float.
        - span min: minimum value of span. Float.
        - span_max: maximum value of span. Float.
        - batch_size: number of intervals we want to randomly draw. Integer.
    Outputs:
        - minimum: minimum value of the intervals. Shape: (batch_size,1).
        - maximum: maximum value of the intervals. Shape: (batch_size,1).
    '''
    mean = generateUniform(mean_min, mean_max, (batch_size, 1), device=device)
    span = generateUniform(span_min, span_max, (batch_size, 1), device=device)
    minimum = mean - span / 2.0
    maximum = mean + span / 2.0

    return minimum, maximum

def noise(var, dims, device='cuda'):
    '''Function that returns a complex vector that follows a complex Gaussian of 
    zero mean and covariance matrix var*eye(dims[1]).
    Inputs:
        - var - variance of each component of the multivariate Gaussian
        - dims - dimensions of the output noise
            - dims[0] - #samples        - dims[1] - #dimensions
    Output:
        - noise_realization: realization of the complex Gaussian random variable
    '''

    return torch.sqrt(var/2.0) * (torch.randn(dims, device=device) + 1j * torch.randn(dims, device=device))

def steeringMatrix(theta_min, theta_max, pos, lamb):
    '''Function that returns a matrix whose columns are steering vectors of the
    form e^(-j*2*pi/lamb*pos*sin(theta)), where theta~U(theta_min, theta_max).
    Inputs:
         - theta_min: minimum of the angle uncertainty region. Real tensor with
        batch_size elements.
        - theta_max: maximum of the angle uncertainty region. It should have the
        same shape as theta_min.
        - pos: antenna positions. Real tensor containing as many elements as 
        the number of Tx/Rx antennas.
        - lamb: carrier wavelength. Float.
    Outputs:
        - matrix: matrix whose columns are steering vectors.
        - theta: realization of the random variable of the angle.
    '''
    theta = generateUniform(theta_min, theta_max, theta_min.shape, device=theta_min.device) #Uniform in [theta_min, theta_max)
    theta = theta.view(1, -1)   #Row vector to have steering vectors in columns at the output

    matrix = torch.exp(-1j * 2 * np.pi / lamb * pos.view(-1,1).type(torch.cfloat) @ torch.sin(theta.type(torch.cfloat)))
    return matrix, theta.view(-1,1)

def delayMatrix(r_min, r_max, delta_f, S):
    '''Function that returns a matrix whose columns are complex vectors 
    of the form e^(-j*2*pi*s*delta_f*2*R/c), where s=0,...,S-1 and 
    R~U(r_min, r_max). 
    Inputs:
        - r_min: minimum value of the expected target range. Real tensor.
        - r_max: maximum value of the expected target range. It should have the 
        same size as r_min
        - delta_f: spacing between different subcarriers.
        - S: number of subcarriers.
        - numColumns: number of columns of the output matrix.
    Outputs:
        - matrix: matrix whose columns are complex vectors.
        - range_tgt: realization of the random variable of the target range.
    '''
    device = r_min.device
    range_tgt = generateUniform(r_min, r_max, r_min.shape, device=device)   #Uniform in [r_min, r_max)
    range_tgt = range_tgt.view(1, -1)   #Row vector to have steering vectors in columns at the output

    matrix = torch.exp(-1j*2*np.pi*delta_f * 2/3e8 * torch.arange(0,S, device=device).view(-1,1).type(torch.cfloat) @ range_tgt.type(torch.cfloat))
    return matrix, range_tgt.view(-1,1)


def createPrecoder(theta_min, theta_max, A_matrix, Ngrid = 256, P=None, device='cuda'):
    '''
    Function that creates a precoder based on a LS solution 
    x_out = min_x ||b-A^Tx||^2. Considering Ngrid angles between [-pi/2, pi/2], 
    b is a vector such that [b]_i=K if theta_i is in [theta_min, theta_max] and
    0 otherwise. 
    Input:
        - theta_min: minimum angle from the known a priori information. 
        Real tensor of shape (batch_size, 1)
        - theta_max: maximum angle of the a priori information. It should have the
        same shape as theta_min.
        - A_matrix: matrix of steering vectors covering from -pi to +pi. 
        Complex tensor of shape (K,Ngrid) or shape (batch_size, K, Ngrid)
        - Ngrid: number of grid angles to consider between [-pi/2, pi/2]. Default: 256
        - P: transmitter power, defined as the squared norm of the precoder. 
        Default: 1W.
    Output:
        - precoder_out: solution to the LS beamformer. Complex tensor of shape 
        (batch_size, K, 1)
        - b_vec: matrix whose rows are binary vectors indicating 1 for the desired 
        direction to be illuminated by the antenna. Shape: (batch_size, Ngrid)
    '''
    #Set default value for P
    if P==None:
        P = torch.tensor(1.0, dtype=torch.float32, device=device)
    #Extract parameters from data
    batch_size = theta_min.shape[0] if theta_min.dim()>0 else 1
    K = A_matrix.shape[0]
    #Create b vector
    angle_vector = torch.linspace(-np.pi/2, np.pi/2, Ngrid, device = device).view(1,-1)
    b_vec = ((angle_vector >= theta_min) & (angle_vector <= theta_max))*1.0*K          #Shape: (batch_size, Ngrid)  
    b_vec = b_vec.type(torch.cfloat).transpose(-1,-2)                                  #A_matrix is of complex type, so should b
    #Compute solution
    precoder = torch.linalg.lstsq(A_matrix.transpose(-1,-2),b_vec).solution.transpose(-1,-2)     #Shape: (batch_size, K)
    #Normalize precoder to have unit norm per batch sample (per row)
    precoder_norm = precoder * torch.sqrt(P) / torch.norm(precoder, dim=-1, keepdim=True)
    precoder_out = precoder_norm.view((batch_size, K, 1))
    return precoder_out, b_vec.transpose(-1,-2)


def createCommonPrecoder(f_radar, f_comm, eta, phi):
    '''Function that creates a precoder for JRC using the technique of
    [A. Zhang et all]. We suppose that the precoder is unit-energy.
    Inputs:
        - f_radar: radar precoder. Shape: (batch_size, K, 1)
        - f_comm: communication precoder. Shape: (batch_size, K, 1)
        - eta: trade-off parameter. Real tensor
        - phi: trade-off parameter. Real tensor
    Output:
        - precoder: combined precoder. Shape: (batch_size, K, 1)
    '''

    temp = torch.sqrt(eta) * f_radar + torch.sqrt(1-eta) * torch.exp(1j * phi) * f_comm    
    precoder = temp / torch.norm(temp, dim=-2, keepdim=True)

    return precoder

#####################
# Sensing functions #
#####################
def radarCH(mean_rcs, theta_min, theta_max, r_min, r_max, delta_f, K, pos, lamb, S, f_precoder, x_msg, numTargets, noiseVariance_r):
    '''
    Function that simulates the radar channel under multiple targets
    Inputs:
        - mean_rcs: mean value of the radar cross section (RCS) of the targets. Real tensor.
        - theta_min: minimum value of the targets' angle. Real tensor of size (batch_size, 1)
        - theta_max: maximum value of the targets' angle. It should have the same shape as theta_min
        - r_min: minimum value of the targets' range. Real tensor of size (batch_size, 1)
        - r_max: maximum value of the targets' range. It should have the same shape as r_min
        - delta_f: subcarrier spacing. Real tensor.
        - K: number of transmitted antenna elements. Integer.
        - pos: actual position of the antenna elements in the array (true impaired values). Real
        tensor of K elements.
        - S: number of subcarriers. Integer.
        - f_precoder: precoder to steer the antenna energy into a particular direction. Complex
        tensor of shape (batch_size, K, 1)
        - x_msg: complex messages to transmit. Complex tensor of shape (batch_size, S, 1)
        - numTargets: number of targets involved in the radar channel. Shape:
        (batch_size, ) 
        - noiseVariance_r: variance of the noise to be added at the receiver side. Real tensor.
    Output:
        - Y: output of the radar channel. Shape: (batch_size, K, S)
        - theta: realization of the targets' angle. Shape: (batch_size, 1)
        - range_tgt: realization of the targets' range. Shape: (batch_size, 1)
    '''
    device = f_precoder.device

    '''Idea to avoid loops: use a batch size equal to the sum of the number of targets
    in all batches, then sum those matrices that correspond to the same batch'''
    batch_size_new = numTargets.sum()
    #We need to repeat each precoder, angle information, and range information according to numTargets
    f_precoder = torch.repeat_interleave(f_precoder, numTargets, dim=0)     #Shape: (batch_size_new, K, 1)
    x_msg      = torch.repeat_interleave(x_msg, numTargets, dim=0)          #Shape: (batch_size_new, S, 1)
    theta_min  = torch.repeat_interleave(theta_min, numTargets, dim=0)      #Shape: (batch_size_new, 1)
    theta_max  = torch.repeat_interleave(theta_max, numTargets, dim=0)      #Shape: (batch_size_new, 1)
    r_min      = torch.repeat_interleave(r_min, numTargets, dim=0)          #Shape: (batch_size_new, 1)
    r_max      = torch.repeat_interleave(r_max, numTargets, dim=0)          #Shape: (batch_size_new, 1)

    # Obtain delay vector and target range per batch sample
    phase_shift, range_tgt = delayMatrix(r_min, r_max, delta_f, S)      #phase_shift.shape = (S, batch_size_new); range_tgt.shape = (batch_size_new,1)
    phase_shift = phase_shift.transpose(-1,-2).view(batch_size_new, S, 1)
    range_tgt = range_tgt.view(batch_size_new, 1, 1)

    # Random complex gain, constant across all subcarriers, changes for each batch
    rcs = torch.empty((batch_size_new,1,1), device=device).exponential_(lambd=1/mean_rcs)       #Swerling model 3, exp distri. with mean 1/lambd
    mag_gain = torch.sqrt(rcs*lamb**2/((4*np.pi)**3*range_tgt**4))
    phase_gain = generateUniform(0, 2*np.pi, (batch_size_new,1,1), device=device)               #Uniformly distributed within [0, 2pi)
    psi = mag_gain*torch.exp(1j*phase_gain)            #Shape: (batch_size_new, 1, 1)

    # Steering vector that accounts for the antenna array
    vector, theta = steeringMatrix(theta_min, theta_max, pos, lamb)          #vector.shape = (K, batch_size_new); theta.shape = (batch_size_new,1)
    # Reshape it to have a column steering vector in each batch
    vector = vector.transpose(-1,-2).reshape(batch_size_new,K,1)

    #Calculate the reflection using the steering vector
    reflection = psi * vector @ vector.permute(0,2,1) @ f_precoder @ (x_msg * phase_shift).permute(0,2,1)   #Shape: (batch_size_new, K, S)
    #sum those reflections corresponding to the same batch
    batch_size = len(numTargets)
    ind = torch.arange(batch_size, device=device).repeat_interleave(numTargets)
    reflection_sum = torch.index_add(torch.zeros((batch_size,K,S), dtype=torch.cfloat, device=device),0,ind, reflection) #Shape: (batch_size, K, S)
    
    #Create noise to add to the reflected signal
    N = noise(noiseVariance_r, reflection_sum.shape, device=device)
    Y = reflection_sum + N
    
    return Y, theta.view(-1,1), range_tgt.view(-1,1)


def OMPReceiver(Y_radar, theta_min, theta_max, r_min, r_max, f_spacing, maxTargets, A_matrix, numSamplesAngle = 360, numSamplesRange = 100):
    '''Function that implements the orthogonal matching pursuit (OMP) algorithm 
    for the received radar signal, to estimate the number of targets, angles and
    ranges of each of them. We assume a MIMO OFDM radar monostatic transceiver
    with K antennas and S subcarriers.
    Inputs:
        - Y_radar: received radar signal. Shape: (batch_size, K, S).
        - theta_min: minimum value of the a priori target angle information.
        Real tensor
        - theta_max: maximum value of the a priori target angle information.
        Real tensor
        - r_min: minimum of the a priori target range information. Real tensor
        - r_max: maximum of the a priori target range information. Real tensor
        - f_spacing: spacing between OFDM subcarriers. Real tensor
        - maxTargets: considered maximum number of targets in the environment.
        Shape: (batch_size, )
        - A_matrix: matrix of steering vectors. Complex tensor of shape 
        (batch_size, numAntennas, Ngrid) or (numAntennas, Ngrid)
        - numSamplesAngle: number of samples to consider in the grid search for 
        angle. Integer. Default: 360
        - numSamplesRange: number of samples to consider in the grid search for 
        range. Integer. Default: 100
    Outputs:
        - metric_result: estimated metric for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets). The metric is the value that 
        distinguishes if the measurement is really a target.
        - theta_result: estimated angle for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets)
        - range_result: estimated range for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets)
    '''
    device = Y_radar.device
    batch_size, K, S = Y_radar.shape

    #Retrieve number of angular and range sectors that are being considered (either 1 or batch_size)
    num_angle_sectors = 1 if ((not torch.is_tensor(theta_min)) or (theta_min.dim()==0)) else len(theta_min)
    num_angle_sectors_arange = torch.arange(num_angle_sectors, device=device)
    num_range_sectors = 1 if ((not torch.is_tensor(r_min)) or (r_min.dim()==0)) else len(r_min)
    num_range_sectors_arange = torch.arange(num_range_sectors, device=device)
    #Force A_matrix to have shape (num_sectors, K, Ngrid_angle). Useful for dictionary learning.
    if A_matrix.dim() == 2:
        A_matrix = A_matrix.unsqueeze(0).repeat_interleave(num_angle_sectors,dim=0)

    # Create matrix of steering vectors and phase shifts to perform 2D grid search
    #Grids of angle and range
    delta_angle = (theta_max-theta_min)/(numSamplesAngle-1)
    theta_grid_matrix = theta_min + delta_angle*torch.arange(numSamplesAngle, device=device).reshape(1,-1)  #Shape: (num_angle_sectors, numSamplesAngle)
    delta_range = (r_max-r_min)/(numSamplesRange-1)
    range_grid_matrix = r_min + delta_range*torch.arange(numSamplesRange, device=device).reshape(1,-1)      #Shape: (num_range_sectors, numSamplesRange)
    # Matrix whose columns are phase shifts for 1 tau
    phaseGrid, _ = delayMatrix(range_grid_matrix, range_grid_matrix, f_spacing, S)                          #Shape: (S, num_range_sectors*numSamplesRange)
    phaseGrid = phaseGrid.transpose(-1,-2).reshape(num_range_sectors, numSamplesRange, S).permute(0,2,1)

    #Vectors to save
    theta_result, range_result = torch.empty(batch_size, maxTargets, device=device), torch.empty(batch_size, maxTargets, device=device)
    metric_result = torch.empty(batch_size, maxTargets, device=device)
    #Matrix to update in each iteration by appending new matrices
    M_vec = torch.tensor([], device=device, dtype=torch.cfloat)
    #Initialize residual as the received signal
    residual = Y_radar
    for i in range(maxTargets):
        #Function to maximize (size: batch_size x numSamplesAngle x numSamplesRange)
        metric = torch.abs( torch.conj(A_matrix.transpose(-1,-2)) @ residual @ torch.conj(phaseGrid) )

        #Estimate of AoA and range
        maximum = metric.view(batch_size, -1).argmax(dim=-1)   
        max_theta = torch.div(maximum, numSamplesRange, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
        max_range = maximum % numSamplesRange                                   #Shape: (batch_size)
        #We can directly use previous tensors since they are integers
        est_theta = theta_grid_matrix[num_angle_sectors_arange, max_theta].view(batch_size, 1)
        est_range = range_grid_matrix[num_range_sectors_arange, max_range].view(batch_size, 1)
        # Compute maximum value of metric to calculate later probability of detection
        metric_max, _ = torch.max(metric.view(batch_size, -1), dim=1)

        #Compute matrix M from estimated angles and ranges
        est_steringGrid = A_matrix[num_angle_sectors_arange.unsqueeze(1), :, max_theta.unsqueeze(1)].reshape((batch_size, K, 1))
        est_phaseGrid, _ = delayMatrix(est_range, est_range, f_spacing, S)      #Since estimated theta is an on-grid estimate, this amounts to taking columns of P_matrix
        est_phaseGrid = est_phaseGrid.transpose(0,1).view((batch_size, S, 1))
        M = est_steringGrid @ est_phaseGrid.permute(0,2,1)   #(batch_size, K, S)

        #Append M matrix to calculate the LS solution
        M_vec = torch.cat((M_vec, M.view(batch_size,K*S,1)), dim=-1)
        #Compute LS solution for alpha
        Y_radar_vec = Y_radar.view(batch_size, K*S, 1)
        alpha_ls = torch.linalg.lstsq(M_vec, Y_radar_vec).solution

        #Update residual
        residual = (Y_radar_vec - M_vec@alpha_ls).view(batch_size, K, S)

        #Save estimated angle, range and metric
        theta_result[:,i] = est_theta.flatten()
        range_result[:,i] = est_range.flatten()
        metric_result[:,i] = metric_max.flatten()

    return metric_result, theta_result, range_result


def OMPReceiverFixed(Y_radar, theta_min, theta_max, r_min, r_max, f_spacing, maxTargets, A_matrix, numSamplesAngle = 360, numSamplesRange = 100):
    '''
    
    '''
    device = Y_radar.device
    batch_size, K, S = Y_radar.shape

    #Retrieve number of angular and range sectors that are being considered (either 1 or batch_size)
    num_angle_sectors = 1 if ((not torch.is_tensor(theta_min)) or (theta_min.dim()==0)) else len(theta_min)
    num_angle_sectors_arange = torch.arange(num_angle_sectors, device=device)
    num_range_sectors = 1 if ((not torch.is_tensor(r_min)) or (r_min.dim()==0)) else len(r_min)
    num_range_sectors_arange = torch.arange(num_range_sectors, device=device)
    #Force A_matrix to have shape (num_sectors, K, Ngrid_angle). Useful for dictionary learning.
    if A_matrix.dim() == 2:
        A_matrix = A_matrix.unsqueeze(0).repeat_interleave(num_angle_sectors,dim=0)

    #Create binary vector to mask the angle-delay map later on
    angle_vector = torch.linspace(-np.pi/2, np.pi/2, numSamplesAngle, device = device).view(1,-1)
    mask_angle = ((angle_vector >= theta_min) & (angle_vector <= theta_max))*1.0          #Shape: (num_angle_sectors, Ngrid)  
    mask_angle = mask_angle.reshape(num_angle_sectors, numSamplesAngle, 1)    

    # Create matrix of steering vectors and phase shifts to perform 2D grid search
    #Grids of angle and range
    delta_angle = (theta_max-theta_min)/(numSamplesAngle-1)
    theta_grid_matrix = theta_min + delta_angle*torch.arange(numSamplesAngle, device=device).reshape(1,-1)  #Shape: (num_angle_sectors, numSamplesAngle)
    delta_range = (r_max-r_min)/(numSamplesRange-1)
    range_grid_matrix = r_min + delta_range*torch.arange(numSamplesRange, device=device).reshape(1,-1)      #Shape: (num_range_sectors, numSamplesRange)
    # Matrix whose columns are phase shifts for 1 tau
    phaseGrid, _ = delayMatrix(range_grid_matrix, range_grid_matrix, f_spacing, S) #Shape: (S, num_range_sectors*numSamplesRange)
    phaseGrid = phaseGrid.transpose(-1,-2).reshape(num_range_sectors, numSamplesRange, S).permute(0,2,1)

    #Vectors to save
    theta_result, range_result = torch.empty(batch_size, maxTargets, device=device), torch.empty(batch_size, maxTargets, device=device)
    metric_result = torch.empty(batch_size, maxTargets, device=device)
    #Matrix to update in each iteration by appending new matrices
    M_vec = torch.tensor([], device=device, dtype=torch.cfloat)
    #Initialize residual as the received signal
    residual = Y_radar
    for i in range(maxTargets):
        #Function to maximize (size: batch_size x numSamplesAngle x numSamplesRange)
        metric = torch.abs( torch.conj(A_matrix.transpose(-1,-2)) @ residual @ torch.conj(phaseGrid) )
        #Mask the angle-delay map to the considered angular sector
        metric *= mask_angle

        #Estimate of AoA and range
        maximum = metric.view(batch_size, -1).argmax(dim=-1)   
        max_theta = torch.div(maximum, numSamplesRange, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
        max_range = maximum % numSamplesRange                                   #Shape: (batch_size)
        #We can directly use previous tensors since they are integers
        est_theta = theta_grid_matrix[num_angle_sectors_arange, max_theta].view(batch_size, 1)
        est_range = range_grid_matrix[num_range_sectors_arange, max_range].view(batch_size, 1)
        # Compute maximum value of metric to calculate later probability of detection
        metric_max, _ = torch.max(metric.view(batch_size, -1), dim=1)

        #Compute matrix M from estimated angles and ranges
        est_steringGrid = A_matrix[num_angle_sectors_arange.unsqueeze(1), :, max_theta.unsqueeze(1)].reshape((batch_size, K, 1))
        est_phaseGrid, _ = delayMatrix(est_range, est_range, f_spacing, S)      #Since estimated theta is an on-grid estimate, this amounts to taking columns of P_matrix
        est_phaseGrid = est_phaseGrid.transpose(0,1).view((batch_size, S, 1))
        M = est_steringGrid @ est_phaseGrid.permute(0,2,1)   #(batch_size, K, S)

        #Append M matrix to calculate the LS solution
        M_vec = torch.cat((M_vec, M.view(batch_size,K*S,1)), dim=-1)
        #Compute LS solution for alpha
        Y_radar_vec = Y_radar.view(batch_size, K*S, 1)
        alpha_ls = torch.linalg.lstsq(M_vec, Y_radar_vec).solution

        #Update residual
        residual = (Y_radar_vec - M_vec@alpha_ls).view(batch_size, K, S)

        #Save estimated angle, range and metric
        theta_result[:,i] = est_theta.flatten()
        range_result[:,i] = est_range.flatten()
        metric_result[:,i] = metric_max.flatten()

    return metric_result, theta_result, range_result


def diffOMPReceiver(Y_radar, A_matrix, b_matrix_angle, angle_grid, r_min, r_max, numSamplesRange, f_spacing, pixels_angle, pixels_range, maxTargets):
    '''Function that implements a differentiable version of the Orthogonal 
    matching pursuit (OMP) algorithm for the received radar signal, to 
    estimate the number of targets, angles and ranges of each of them. We 
    assume a MIMO OFDM radar monostatic transceiver with K antennas and S 
    subcarriers.
    Inputs:
        - Y_radar: received radar signal. Shape: (batch_size, K, S).
        - A_matrix: matrix whose columns are steering vectors for a given angle.
        Shape: (K, numSamplesAngle)
        - b_matrix_angle: binary matrix whose rows are 1 for those angles 
        inside the prior information and 0 otherwise. 
        Shape: (batch_size, numSamplesAngle)
        - angle_grid: grid of potential angles to consider. Real tensor with
        Ngrid_angle elements.
        - r_min: minimum range of the a priori information for the targets'
        range. Real tensor of shape (batch_size, 1)
        - r_max: maximum range of the a priori information for the targets'
        range. Real tensor of shape (batch_size, 1)
        - numSamplesRange: number of candidate ranges to consider between
        r_min and r_max. Integer.
        - f_spacing: subcarrier spacing. Float.
        - pixels_angle: number of rows to look around the maximum in the 
        angle-delay map. Integer tensor.
        - pixels_range: number of columns to look around the maximum in the 
        angle-delay map. Integer tensor.
        - maxTargets: considered maximum number of targets in the environment.
        Shape: (batch_size, )
    Outputs:
        - metric_result: estimated metric for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets). The metric is the value that 
        distinguishes if the measurement is really a target.
        - theta_result: estimated angle for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets)
        - range_result: estimated range for each possible target in each batch 
        sample. Shape: (batch_size, maxTargets)
    '''
    device = Y_radar.device
    batch_size, K, S = Y_radar.shape
    numSamplesAngle = A_matrix.shape[1]

    #Useful tensors to use later 
    batch_size_arange = torch.arange(batch_size, device=device)
    temp_batch_indices =  batch_size_arange.unsqueeze(1)
    temp_row_indices = temp_batch_indices.repeat(1, 2 * pixels_range + 1)

    # Constrain matrix A to known angle information (each batch sample could have a different angle sector)
    b_matrix_angle_rsh = b_matrix_angle.reshape((batch_size, 1, numSamplesAngle))
    A_matrix_con = A_matrix * b_matrix_angle_rsh #Shape: (b_size, K, numSamplesAngle)
    
    #Create matrix of delays to perform grid search
    delta_range = (r_max-r_min)/(numSamplesRange-1)
    range_grid_matrix = r_min + delta_range*torch.arange(numSamplesRange, device=device).reshape(1,-1)      #Shape: (num_range_sectors, numSamplesRange)
    phaseGrid, _ = delayMatrix(range_grid_matrix, range_grid_matrix, f_spacing, S) #Shape: (S, num_range_sectors*numSamplesRange)
    P_matrix_con = phaseGrid.transpose(-1,-2).reshape(batch_size, numSamplesRange, S).permute(0,2,1)

    #Vectors to save
    theta_result, range_result = torch.empty(batch_size, maxTargets, device=device), torch.empty(batch_size, maxTargets, device=device)
    metric_result = torch.empty(batch_size, maxTargets, device=device)
    #Matrix to update in each iteration by appending new matrices
    M_vec = torch.tensor([], device=device, dtype=torch.cfloat)
    #Initialize residual as the received signal
    residual = Y_radar
    for i in range(maxTargets):
        #Function to maximize (size: batch_size x numSamplesAngle x numSamplesRange)
        metric = torch.abs(A_matrix_con.conj().permute(0,2,1) @ residual @ P_matrix_con.conj())

        '''Estimate angle and range'''
        with torch.no_grad():
            #Compute the index of the highest element of each matrix in the batch, by flattening the matrices first
            top_index = metric.view(batch_size, -1).argmax(dim=-1).reshape(batch_size,1) 
            #Compute the rows and columns of the maximum element
            max_row = torch.div(top_index, numSamplesRange, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
            max_col = top_index % numSamplesRange                                   #Shape: (batch_size,1)
            #Create vectors with the corresponding indexes that we want to threshold around the maximum
            max_row_indexes = max_row-pixels_angle + torch.arange(2*pixels_angle+1, dtype=torch.int32, device=device).reshape(1,-1)     #Shape:(batch_size, 2*pixels_angle+1)
            max_col_indexes = max_col-pixels_range + torch.arange(2*pixels_range+1, dtype=torch.int32, device=device).reshape(1,-1)     #Shape:(batch_size, 2*pixels_range+1)
            #Get the rows and columns of the previous indexes that correspond to values outside the angle-delay map
            temp_mask_rows = torch.where((numSamplesAngle-1 < max_row_indexes) | (max_row_indexes < 0), 1.0, 0.0)
            inf_idx_rows = torch.nonzero(temp_mask_rows, as_tuple=True)
            temp_mask_cols = torch.where((numSamplesRange-1 < max_col_indexes) | (max_col_indexes < 0), 1.0, 0.0)
            inf_idx_cols = torch.nonzero(temp_mask_cols, as_tuple=True)
            #Restrict the possible rows and columns not to exceed the limits of the map
            max_row_restr = torch.maximum(torch.tensor(0, device=device), max_row_indexes)
            max_row_restr = torch.minimum(torch.tensor(numSamplesAngle-1, device=device), max_row_restr)     #Shape:(batch_size, 2*pixels_angle+1)
            max_col_restr = torch.maximum(torch.tensor(0, device=device), max_col_indexes)
            max_col_restr = torch.minimum(torch.tensor(numSamplesRange-1, device=device), max_col_restr)     #Shape:(batch_size, 2*pixels_range+1)
            #Reshape to take a matrix grid in each batch
            row_sel = max_row_restr.reshape(batch_size, -1,1)
            col_sel = max_col_restr.reshape(batch_size, 1,-1)
        metric_thr = metric[torch.arange(batch_size, dtype=torch.long, device=device).reshape(batch_size,1,1), row_sel, col_sel]    #Shape:(batch_size, 2*pixels_angle+1, 2*pixels_range+1)
        #Substitute the values that correspond to incorrect indexes by -Inf
        #(same as discarding elements, but this way paralellizes operations)
        len_inf_idx_rows = len(inf_idx_rows[0])
        len_inf_idx_cols = len(inf_idx_cols[0])
        metric_thr[inf_idx_rows[0], inf_idx_rows[1], :] = float('-Inf')*torch.ones((len_inf_idx_rows,2*pixels_range+1), device=device)       ########Note: if not using softmax in the next step, be careful with the -Inf
        metric_thr[inf_idx_cols[0], :, inf_idx_cols[1]] = float('-Inf')*torch.ones((len_inf_idx_cols,2*pixels_angle+1), device=device)       #### Here cols and pixels_angle(associated with rows) should be crossed
        #Apply normalization to the selected elements
        soft_elements = (F.softmax(metric_thr.view(batch_size, -1), dim=1)).reshape(metric_thr.shape)       #Shape:(batch_size, 2*pixels_angle+1, 2*pixels_range+1)
        #Sum elements corresponding to the same angle or range (row or column)
        soft_row = torch.sum(soft_elements, dim=2).reshape(batch_size, -1)          #Shape:(batch_size, 2*pixels_angle+1)
        soft_col = torch.sum(soft_elements, dim=1).reshape(batch_size, -1)          #Shape:(batch_size, 2*pixels_range+1)

        #For angle we just care about rows, for range we just care about columns    
        angle_grid_con = angle_grid.flatten()[max_row_restr]            #Make sure vectors are flattened for indexing
        range_grid_con = range_grid_matrix[temp_row_indices, max_col_restr]            #Shape: (batch_size, 2*pixels_range+1) 
        #Estimate angle and range
        est_angle = torch.sum(soft_row * angle_grid_con, dim=1)         #Shape:(batch_size,)
        est_range = torch.sum(soft_col * range_grid_con, dim=1)
        # Compute maximum value of metric to calculate later probability of detection
        metric_max, _ = torch.max(metric.view(batch_size, -1), dim=1)

        #Compute matrix M from estimated angles and ranges
        with torch.no_grad():
            est_steringGrid = A_matrix[:,max_row.flatten()].transpose(-1,-2).reshape((batch_size, K, 1))    #Here we need to use the learned matrix, take columns based on max_row
            est_phaseGrid = P_matrix_con[temp_batch_indices,:,max_col].reshape((batch_size, S, 1))
            M = est_steringGrid @ est_phaseGrid.permute(0,2,1)   #(batch_size, K, S)

            #Append M matrix to calculate the LS solution
            M_vec = torch.cat((M_vec, M.view(batch_size,K*S,1)), dim=-1)
            #Compute LS solution for alpha
            Y_radar_vec = Y_radar.view(batch_size, K*S, 1)
            alpha_ls = torch.linalg.lstsq(M_vec, Y_radar_vec).solution

        #Update residual
        residual = (Y_radar_vec - M_vec@alpha_ls).view(batch_size, K, S)

        #Save estimated angle, range and metric
        theta_result[:,i] = est_angle.flatten()
        range_result[:,i] = est_range.flatten()
        metric_result[:,i] = metric_max.flatten()

    return metric_result, theta_result, range_result

def GOSPALoss(estimation, target, gamma, alpha=2, p=2, loss_method='norm', device='cuda'):
    '''Function that implements the GOSPA loss function. See 
    [Rahmathullah, A. S., et al., "Generalized optimal sub-pattern assignment 
    metric"].
    Inputs:
        - estimation: estimated values. Real tensor. It can be 1D, 2D or 3D
        tensor.
            + For 1D tensor, the number of elements is the number of estimated
            targets
            + For 2D and 3D tensors, the number of rows is the number of estimated 
            targets.
            + For 3D tensor, it is assumed that the 1st Python dimension is the
            batch size, and that it is desired to apply the GOSPA loss for each
            element in the batch dimension in parallel. 
        - target: true values. Real tensor. It can be 1D 2D or 3D tensor. Same
        description of vector sizes applies to the target, with some remarks:
            +For 2D tensors, it is expected that they both have the same number
            of columns (dimensions of the vectors to compare).
            +For 3D tensors, it is expected that the target and the estimation both
            have the same batch size and number of columns. 
        - gamma: cut-off value for the metric (see paper). Real tensor with gradient.
        - alpha: parameter alpha in the definition of the loss. Real tensor.
        - p: parameter p in the definition of the loss. Real tensor.
        - loss_method: type of loss to be used. It can be:
            + 'norm': computes ||x-y||
            + 'bce': computes y*log(x) + (1-y)*log(1-x)
    Outputs:
        - GOSPA loss with the given inputs
        - Average loss according to 'loss_method' (GOSPA loss without 
        cardinality penalty term and averaging instead of summing)
    '''
    #We always consider the estimation to be the set with lower cardinality
    if estimation.numel() > target.numel():
        estimation, target = target, estimation

    #Make the inputs 3D to work with any kind of input in general
    if estimation.dim() == 1:
        estimation = estimation.unsqueeze(0).unsqueeze(-1)
        target = target.unsqueeze(0).unsqueeze(-1)
    elif estimation.dim() == 2:
        estimation = estimation.unsqueeze(0)
        target = target.unsqueeze(0)

    # Extract useful data from tensors
    b_size, card_est, dimensions = estimation.shape
    card_tgt = target.shape[-2]                      #It is assumed that both have same batch size and dimensions

    #If one set is empty we can have errors in the Pytorch functions
    if card_est > 0:
        #Take into account all possible permutations of 'estimation' without repetition
        num_permutations_est = math.factorial(card_est)  
        permutation_list = list(itertools.permutations(range(card_est)))
        permutation_index = torch.tensor(permutation_list, dtype=torch.long, device=device)    #Shape: (num_permutations_est, card_est)
        estimation_comb = estimation[:,permutation_index,:].transpose(-2,-1).reshape(b_size,num_permutations_est,dimensions,1,card_est)
        
        #Take all combinations of card_est elements of the target
        combinations_tgt = torch.combinations(torch.arange(card_tgt, device=device), card_est)
        num_combinations_tgt = combinations_tgt.shape[0]
        tgt_comb = target[:,combinations_tgt].permute(0,3,1,2).reshape(b_size,dimensions,num_combinations_tgt,card_est)
        #Add dummy dimension to compare with estimation_comb
        tgt_comb = tgt_comb[:,None,:,:]

        #Loss for each combination
        if loss_method == 'norm':
            loss = torch.norm(estimation_comb-tgt_comb,dim=-3) #Shape: (batch_size, num_combinations_est, num_combinations_tgt, card_est)
        if loss_method == 'bce':
            loss = torch.sum(tgt_comb*torch.log(estimation) + (1-tgt_comb)*torch.log(1-estimation), dim=-3)

        cut_off = torch.minimum(loss,gamma)
        loss_sum = torch.sum(cut_off**p,dim=-1) #Shape: (batch_size, num_combinations_est, num_combinations_tgt)
        min_loss, _ = torch.min(loss_sum.reshape(b_size,-1), dim=-1)    #Shape: (batch_size, 1)    
        #Average loss with gamma=0
        avg_loss, _ = torch.min(torch.sum(loss**p, dim=-1).reshape(b_size,-1), dim=-1)
    else:
        min_loss = torch.zeros((b_size),device=device)
        avg_loss = float('nan')*torch.ones((b_size), device=device)
    
    gospa_out = (min_loss + gamma**p/alpha*(card_tgt-card_est))**(1/p)
    return gospa_out, avg_loss**(1/p)


###########################
# Communication functions #
###########################
def MPSK(M, rotation=0, device='cuda'):
    '''Function that returns an array with all the symbols of a M-PSK
    constellation.
    Inputs:
        - M: size of the constellation
        - rotation: angle of rotation of the constellation [rad]
    '''
    return torch.exp(1j * (2*np.pi/M * torch.arange(0,M, device=device) + rotation))

def commChannel(mean_rcs, theta_min, theta_max, r_min, r_max, f_spacing, K, pos, lamb, S, precoder, q_tx, numTargets, Rcp, noiseVariance_r, device='cuda'):
    '''Function that simulates the radar channel under multiple targets
    Inputs:
        - mean_rcs: mean radar cross section of the scatterers. Float tensor.
        - theta_min: minimum angle of the communication scene [rad]. Float tensor.
        - theta_max: maximum angle of the communication scene [rad]. Float tensor.
        - r_min: minimum range of the communication scene [m]. Float tensor.
        - r_max: maximum range of the communication scene [m]. Float tensor.
        - f_spacing: frequency spacing between subcarriers in OFDM [Hz]. Float tensor.
        - K: number of antennas of the Tx ULA. Float tensor.
        - pos: actual position of the antenna elements in the array (true impaired values). Real
        tensor of K elements.
        - lamb: wavelength [m]. Float tensor.
        - S: number of subcarriers in OFDM. Float tensor.
        - precoder: vector that steers the antenna energy into a particular direction.
        Float tensor of shape (batch_size, K, 1)
        - q_tx: vector that represents the transmitted constellation symbols for
        communication. Float tensor of shape (batch_size, S, 1)
        - numTargets: number of targets in the scene. Integer tensor of shape
        (batch_size,)
        - Rcp: equivalent range of the cyclic prefix time. Float tensor.
        - noiseVariance_r: variance of the complex noise of the sensing receiver.
        Float tensor.
    Output:
        - Y: output of the radar channel. Shape: (batch_size, S, 1)
        - theta_list: vector of realizations of the target AoAs. List of tensors,
        where each tensor depends on the number of targets per batch
        - range_list: vector of realizations of the targets ranges. List of tensors,
        where each tensor depends on the number of targets per batch
        - kappa_sum: channel state information for the receiver. Float tensor of
        shape (batch_size, S, 1)
    '''
    batch_size = len(numTargets)

    '''Idea to avoid loops: use a batch size equal to the sum of number of targets
    in all batches, then sum those matrices that correspond to the same batch.'''
    '''How to have different gains for LOS and NLOS paths? Assume all paths are NLOS
    and then replace the positions of LOS paths with LOS gains.'''
    batch_size_new = numTargets.sum()

    #We need to repeat the precoder and messages to match the new batch_size
    precoder_rep = torch.repeat_interleave(precoder, numTargets, dim=0)
    q_rep = torch.repeat_interleave(q_tx, numTargets, dim=0)
    #In case angle and ranges are different for different batch samples, we need to reshape them
    if theta_min.numel() > 1:
        theta_min = theta_min.reshape(batch_size,1,1)
    if theta_max.numel() > 1:
        theta_max = theta_max.reshape(batch_size,1,1)
    if r_min.numel() > 1:
        r_min = r_min.reshape(batch_size,1,1)
    if r_max.numel() > 1:
        r_max = r_max.reshape(batch_size,1,1)

    # == Draw range of NLOS scatterers for a given angle == #
    # Draw LOS range and repeat it according to numTargets (if no target, that value gets removed)
    range_los = torch.rand((batch_size,1,1), device=device) * (r_max - r_min) + r_min
    range_los_rep = range_los.repeat_interleave(numTargets,dim=0)
    # Draw LOS angle and repeat it according to numTargets (if no target, that value gets removed)
    angle_los = torch.rand((batch_size,1,1), device=device) * (theta_max - theta_min) + theta_min
    angle_los_rep = angle_los.repeat_interleave(numTargets,dim=0)
    # Draw NLOS angles
    if theta_min.numel() > 1:
        theta_min = torch.repeat_interleave(theta_min, numTargets, dim=0)
    if theta_max.numel() > 1:
        theta_max = torch.repeat_interleave(theta_max, numTargets, dim=0)
    angle_nlos = torch.rand((batch_size_new,1,1), device=device) * (theta_max - theta_min) + theta_min
    # Compute the difference in angle between the LOS and NLOS paths
    angle_diff = angle_los_rep-angle_nlos
    # Force the scatterers' angles to be at least 0.5 deg apart from LOS
    min_sep_tensor = torch.tensor(0.1*np.pi/180.0, device=device)
    angle_mask = torch.abs(angle_diff) < min_sep_tensor
    angle_nlos[angle_mask] = angle_los_rep[angle_mask] + torch.sign(angle_diff[angle_mask])*min_sep_tensor
    angle_diff = angle_los_rep-angle_nlos
    #Compute the maximum NLOS range of each scatterer
    range_nlos_max = Rcp * (Rcp/2.0 + range_los_rep)/(range_los_rep*(1-torch.cos(angle_diff))+Rcp)
    #Draw range from Tx to scatterer
    if r_min.numel() > 1:
        r_min = torch.repeat_interleave(r_min, numTargets, dim=0)
    range_tx_scat = torch.rand((batch_size_new,1,1), device=device) * (range_nlos_max - r_min) + r_min
    #Compute range from scatterer to Rx (cosine's theorem)
    range_scat_rx = torch.sqrt(range_los_rep**2 + range_tx_scat**2 - 2*range_los_rep*range_tx_scat*torch.cos(angle_diff))
    #Sum ranges to later compute the delay of the NLOS path
    range_nlos = range_tx_scat + range_scat_rx
    # NLOS complex channel gain, constant across all subcarriers, changes for each batch
    rcs_nlos = torch.empty((batch_size_new,1,1), device=device).exponential_(lambd=1/mean_rcs)       #Swerling model 1, exp distri. with mean 1/lambd
    mag_gain_nlos = torch.sqrt(rcs_nlos*lamb**2/((4*np.pi)**3*range_tx_scat**2*range_scat_rx**2))
    phase_gain_nlos = torch.rand((batch_size_new,1,1), device=device)*2*np.pi                       #Uniformly distributed within [0, 2pi]
    psi = mag_gain_nlos*torch.exp(1j*phase_gain_nlos)

    # LOS complex channel gain
    range_los = range_los[numTargets > 0]
    angle_los = angle_los[numTargets > 0]
    mag_gain_los = torch.sqrt(lamb**2/(4*np.pi*range_los)**2)
    phase_gain_los = torch.rand((len(range_los),1,1), device=device)*2*np.pi                       #Uniformly distributed within [0, 2pi]
    psi_los = mag_gain_los*torch.exp(1j*phase_gain_los)

    #Insert the LOS channel gains, angles and ranges into the corresponding positions of the corresponsing vectors
    insert_idx = torch.unique(torch.cat((torch.tensor([0], device=device), torch.cumsum(numTargets,dim=0))))[:-1] #Maybe a cumsum of non-zero elements would deal with 0 targets
    psi[insert_idx] = psi_los           #Shape: (batch_size_new,1,1)
    angle_nlos[insert_idx] = angle_los; angle_vector = angle_nlos;  #Change name to improve readability
    range_nlos[insert_idx] = range_los; range_vector = range_nlos;  #Change name to improve readability

    #Compute steering vectors for each path
    steering_vector, _ = steeringMatrix(angle_vector, angle_vector, pos, lamb)
    # Reshape it to have a column steering steering_vector in each batch
    steering_vector = steering_vector.T.view(batch_size_new,K,1)

    #3D matrix of size (batch_size_new, S, 1) that for each batch it contains a phase shift dependent on the range
    phase_shift = torch.exp(-1j * 4 * np.pi / 3e8 * f_spacing * range_vector.type(torch.cfloat) * torch.arange(S, device=device).reshape(1,S,1).type(torch.cfloat))

    # Dot product between phase vector and communication vector
    temp = q_rep * phase_shift
    #Calculate the reflection using the steering vector
    rx_noiseless = psi * torch.bmm(steering_vector.permute(0,2,1), precoder_rep) * temp  #Shape:(batch_size_new, S, 1)
    #sum those reflections corresponding to the same batch
    ind = torch.arange(batch_size, device=device).repeat_interleave(numTargets)
    rx_noiseless_sum = torch.index_add(torch.zeros((batch_size,S,1), dtype=torch.cfloat, device=device),0,ind, rx_noiseless)

    #Create noise to add to the reflected signal
    N = noise(noiseVariance_r, (batch_size,S,1), device=device)
    Y = rx_noiseless_sum + N

    # == Compute channel state information kappa ==#
    kappa = psi * torch.bmm(steering_vector.permute(0,2,1), precoder_rep) * phase_shift
    kappa_sum = torch.index_add(torch.zeros((batch_size,S,1), dtype=torch.cfloat, device=device),0,ind, kappa)

    #Return true angle and ranges as lists of tensors
    theta_list = angle_vector.flatten().split(numTargets.to(torch.int).tolist())
    range_list = range_vector.flatten().split(numTargets.to(torch.int).tolist())

    return Y, theta_list, range_list, kappa_sum

def computeCommMSE(y_comm, kappa, refConst):
    '''
    Function that computes -|y_comm-kappa*refConst[i]|^2 for each pair (y_comm, kappa).
    Inputs:
        - y_comm: received signal at the communication receiver. 
        Shape: (batch_size, S, 1)
        - kappa: estimated CSI. In this paper we assume that CSI 
        esimtation is perfect. Shape: (batch_size, S, 1)
        - refConst: used constellation to transmit data. Complex
        tensor.
    Outputs:
        - Matrix of shape (batch_size*S, len(refConst)) whose rows correspond to -|y_comm-kappa*refConst[i]|^2
    '''

    #Flatten inputs since we do subcarrier-wise estimation
    y_comm_flatten = y_comm.reshape(-1,1)
    kappa_flatten = kappa.reshape(-1,1)
    #Repeat constellation points across rows
    refConst_matrix = refConst.reshape(1,-1).repeat_interleave(y_comm_flatten.shape[0], dim=0)

    return torch.abs(y_comm_flatten - kappa_flatten*refConst_matrix)**2

def MLdecoder(rec, kappa, refConst):
    ''' Function that performs ML decoding per subcarrier in an OFDM system
    given some CSI.
    Inputs:
        - rec: received signal at the Rx. Shape: (batch_size, S, 1)
        - kappa: channel state information. Shape: (batch_size, S, 1)
        - refConst: reference constellation to discern which message was
        transmitted. Shape: (numMessages,)
    Outputs:
        - est_message: estimated messages flatten. Length: batch_size*S
    '''

    # Create quantity to compare with received signal
    compare = kappa * refConst.reshape(1,len(refConst))
    #This way is faster than temp * (refConst.reshape(len(refConst),1) @ b.permute(0,2,1))

    #Compare received signal with previous quantity
    metric = torch.abs(rec - compare)**2

    #Take the argmin to retrieve the message that minimizes the metric
    est_message = torch.argmin(metric, dim=-1)

    return est_message.flatten()


######################
# Training functions #
######################

def trainNetworkFixedEta(network, train_it, maxTargets, maxPaths, P_power, mean_rcs, Rcp,
                    theta_mean_min_sens_train, theta_mean_max_sens_train,
                    theta_span_min_sens_train, theta_span_max_sens_train, 
                    range_mean_min_sens_train, range_mean_max_sens_train,
                    range_span_min_sens_train, range_span_max_sens_train, 
                    theta_mean_min_comm_train, theta_mean_max_comm_train,
                    theta_span_min_comm_train, theta_span_max_comm_train, 
                    range_mean_min_comm_train, range_mean_max_comm_train,
                    range_span_min_comm_train, range_span_max_comm_train, 
                    numAntennas, numSubcarriers, Delta_f,
                    lamb, Ngrid_angle, Ngrid_range, angle_grid, pixels_angle, pixels_range, msg_card, 
                    refConst, varNoise, ant_pos, batch_size, optimizer, scheduler, loss_comm_fn, eta_weight, epoch_test_list, 
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
                    sch_flag=False, imp_flag = True, initial_pos = None, device='cuda'):
    '''
    Function that performs training of a network and returns the loss function
    across training iterations and intermediate evaluation results if desires.
    Inputs:
        - (some of the input variables are detailed in the 'README_LIB.md' file or 
        at the end of this document)
        - train_it: number of training iterations. Integer.
        - batch_size: number of samples to use in one batch. Integer.
        - optimizer: optimizer to update the parameters to learn.
        - scheduler: scheduler to change the learning rate over training iterations.
        - loss_comm_fn: loss function to use for communication training.
        - eta_weight: weight to combine sensing and communication loss functions. Real 
        value between 0 and 1.
        - epoch_test_list: list of epochs at which the network performance should be 
        tested during training. List of integers.
        - (more angle/range parameters during testing)
        - target_pfa: target value of the false alarm probability. Real value.
        - delta_pfa: maximum allowable error w.r.t target_pfa. Real value.
        - thresholds_pfa: thresholds to distinguish if a measurement is actually a
        target. List of real values.
        - nTestSamples: number of samples to use during testing.
        - sch_flag: flag indicating if a scheduler should be used. Defaul: False.
        - imp_flag: flag indicating if impairment learning or dictionary learning are
        being considered. True means that impairment learning would be considered.
        Defaul: True.
        - initial_pos: initial position of the first (reference) antenna element.
    Output:
        - loss_gospa_np: GOSPA loss throughtout training. List of Floats.
        - loss_comm_np: communication loss throughtout training. List of Floats.
        - loss_avg_np: GOSPA loss without cardinality penalty term and averaging 
        instead of summing. List of Floats
        - num_iterations: iterations at which the network is tested. List of integers.
        - pd_inter: detection probability for the iterations tested during training. 
        List of Floats.
        - pfa_inter: false alarm probability for the iterations tested during training.
        List of Floats. 
        - gospa_pos_inter: GOSPA loss for the iterations tested during training. List of Floats. 
        - avg_dist_inter: GOSPA loss w/o cardinality mismatch and averaging instead of summing
        for the iterations tested during training. List of Floats.  
        - snr_inter: SNR for the iterations tested during training. List of Floats.  
        - ser_inter: SER for the iterations tested during training. List of Floats. 
    '''

    print('*** Training started ***')
    network.train()

    #List to save the loss function values and the tested iterations
    loss_gospa_np, loss_avg_np, loss_comm_np, num_iterations = [], [], [], []
    #Lists to save intermediate results
    pd_inter, pfa_inter, gospa_pos_inter, avg_dist_inter = [], [], [], []
    snr_inter, ser_inter = [], []

    '''
    We perform training by fixing the true number of targets, but then average the
    loss for all possible number of targets strictly greater than 0.
    This avoid a for loop in the GOSPA loss function (at the cost of some memory).
    '''
    for iteration in range(0,train_it):
        loss_gospa_per_target = torch.zeros(5, device=device)
        avg_dist_per_target   = torch.zeros(5, device=device)
        loss_cce_per_target   = torch.zeros(5, device=device)
        
        # Update matrix (K x Ngrid_angle) of steering vectors that will be used later for learning
        if imp_flag:
            A_matrix, _ = steeringMatrix(angle_grid, angle_grid, network.pos, lamb)
        else:
            A_matrix = network.A

        for t in range(5):      #No matter how we choose maxTargets there will always be the same number of subiterations (asuming maxTargets <= 5) 
            #Fix the number of targets to be able to paralellize GOSPA computation
            if t<maxTargets:        #This is to guarantee that we always get all possible number of targets from 1 to maxTargets.
                fixT = torch.tensor(t+1, device=device)
            else:
                fixT = torch.randint(1,maxTargets+1,(), device=device)
            numTargets = fixT * torch.ones((batch_size, ), dtype=torch.int, device=device)
            numPaths = torch.randint(1,maxPaths+1,(batch_size,), device=device)

            #Generate random messages to transmit. 
            msg = torch.randint(0,msg_card,size=(batch_size*numSubcarriers,), dtype=torch.int64, device=device)
            #Reshape msgs to the needs of the comm CH function
            symbols = refConst[msg].reshape(batch_size, numSubcarriers, 1)
            
            # ================= TRANSMITTER =======================
            #Choose random angle interval vector (random mean and random span)
            theta_min_sens, theta_max_sens = generateInterval(theta_mean_min_sens_train, theta_mean_max_sens_train, theta_span_min_sens_train, theta_span_max_sens_train, batch_size, device=device)
            theta_min_comm, theta_max_comm = generateInterval(theta_mean_min_comm_train, theta_mean_max_comm_train, theta_span_min_comm_train, theta_span_max_comm_train, batch_size, device=device)
            #Choose random range interval vector (random mean and random span)
            r_min_sens, r_max_sens = generateInterval(range_mean_min_sens_train, range_mean_max_sens_train, range_span_min_sens_train, range_span_max_sens_train, batch_size, device=device)
            r_min_comm, r_max_comm = generateInterval(range_mean_min_comm_train, range_mean_max_comm_train, range_span_min_comm_train, range_span_max_comm_train, batch_size, device=device)
            #Beamformer 
            precoder_sens, b_matrix_angle_sens = createPrecoder(theta_min_sens, theta_max_sens, A_matrix, Ngrid_angle, P_power, device=device)
            precoder_comm, _ = createPrecoder(theta_min_comm, theta_max_comm, A_matrix, Ngrid_angle, P_power, device=device)                    #The angular prior info. in comms is irrelevant
            #Create ISAC precoder based on the input weighting value
            precoder = createCommonPrecoder(precoder_sens, precoder_comm, eta_weight, torch.tensor(0, device=device))        #We let phi=0 for ease of training.

            # ================= RADAR CHANNEL ================= #
            Y_radar, true_angle, true_range = radarCH(mean_rcs, theta_min_sens, theta_max_sens, r_min_sens, r_max_sens, Delta_f, 
                                                    numAntennas, ant_pos, lamb, numSubcarriers, precoder, symbols, numTargets, varNoise)
            y_comm, _, _, kappa = commChannel(mean_rcs, theta_min_comm, theta_max_comm, r_min_comm, r_max_comm, Delta_f, numAntennas, 
                                              ant_pos, lamb, numSubcarriers, precoder, symbols, numPaths, Rcp, varNoise, device)
            
            # ================= COMMUNICATION RECEIVER ================= #
            #Estimate message pmf
            comm_mse = computeCommMSE(y_comm, kappa, refConst)
            comm_mse_mask = (comm_mse > 0).all(dim=-1)
            neg_mse = -torch.log10(comm_mse[comm_mse_mask])

            # ================= RADAR RECEIVER ================= #
            #Remove the effect of the transmitted symbols (since the radar Rx is the same as the comm Tx)
            Y_radar /= torch.ones((numAntennas,1), device=device).type(torch.cfloat) @ symbols.permute(0,2,1)
            _, est_angle, est_range = diffOMPReceiver(Y_radar, A_matrix, b_matrix_angle_sens, angle_grid, r_min_sens, r_max_sens, Ngrid_range, Delta_f, pixels_angle, pixels_range, maxTargets)

            #Compute estimated position, but take only the true number of targets
            #Reshape angle and range knowing that the number of target is fixed in each batch
            est_angle_rsh = est_angle[:,0:fixT].reshape(batch_size, fixT, 1)    #It is ok to take just the first estimations, since they should
            est_range_rsh = est_range[:,0:fixT].reshape(batch_size, fixT, 1)    #correspond to targets. Althought not ordered, but GOSPA deals with it.
            est_x = est_range_rsh * torch.cos(est_angle_rsh)
            est_y = est_range_rsh * torch.sin(est_angle_rsh)
            est_pos = torch.cat((est_x, est_y), dim=-1) 

            #Compute true position
            #Reshape angle and range knowing that the number of target is fixed in each batch
            true_angle_rsh = true_angle.reshape(batch_size, fixT, 1)
            true_range_rsh = true_range.reshape(batch_size, fixT, 1)
            true_x = true_range_rsh * torch.cos(true_angle_rsh)
            true_y = true_range_rsh * torch.sin(true_angle_rsh)
            true_pos = torch.cat((true_x, true_y), dim=-1)

            #Compute GOSPA position loss between each pair of tensors (they correspond to sets of positions)
            gospa_individual, avg_dist_individual = GOSPALoss(est_pos, true_pos, gamma_gospa_train, mu_gospa, p_gospa, loss_method='norm', device=device) 

            loss_gospa_per_target[t] = torch.mean(gospa_individual)
            avg_dist_per_target[t] = torch.nanmean(avg_dist_individual)
            if len(neg_mse == 0):
                print('This would have given an error in the previous code', flush = True)
                loss_cce_per_target[t] = torch.tensor(0.0, dtype=torch.float32, device=device)
            else:
                loss_cce_per_target[t] = loss_comm_fn(neg_mse, msg[comm_mse_mask])

        # ================= BACKWARD =======================
        optimizer.zero_grad()
        loss_pos = torch.mean(loss_gospa_per_target)
        loss_comm = torch.mean(loss_cce_per_target)
        loss_combined = eta_weight*loss_pos + (1-eta_weight)*loss_comm   
        loss_combined.backward()
        optimizer.step()
        #Update numpy loss vectors
        loss_gospa_np.append(loss_pos.item())
        loss_comm_np.append(loss_comm.item())
        loss_avg_np.append(torch.nanmean(avg_dist_per_target).item())
        if sch_flag:
            #Scheduler step
            scheduler.step(loss_pos)

        #Order the positions of the ULA and fix the first position for impairment learning (physical constraints)
        if imp_flag:
            network.pos.data, _ = torch.sort(network.pos.data)
            network.pos.data = network.pos.data - (network.pos.data[0] - initial_pos)

        #Test network performance only if the number of iterations is within a list of values
        if (iteration+1) in epoch_test_list:
            network.eval()
            num_iterations.append(iteration+1)
            if imp_flag:
                rx_feature = network.pos
            else:
                rx_feature = torch.clone(A_matrix)
            A_tx = torch.clone(A_matrix)
            pd_temp, pfa_temp, gospa_pos_temp, avg_dist_temp = \
                      testSensingFixedPfa(maxTargets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                                          theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                                          range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, 
                                          Ngrid_angle, Ngrid_range, numAntennas, numSubcarriers, varNoise, Delta_f, 
                                          lamb, ant_pos, A_tx, rx_feature, refConst, target_pfa, delta_pfa, thresholds_pfa, gamma_gospa_test, 
                                          mu_gospa, p_gospa, batch_size, nTestSamples, imp_flag, device=device)
            snr_temp, ser_temp = testCommunication(maxPaths, P_power, mean_rcs, 
                                            theta_mean_min_comm_test, theta_mean_max_comm_test, theta_span_min_comm_test, theta_span_max_comm_test, 
                                            range_mean_min_comm_test, range_mean_max_comm_test, range_span_min_comm_test, range_span_max_comm_test,
                                            Ngrid_angle, numAntennas, numSubcarriers, varNoise, 
                                            Delta_f, Rcp, lamb, ant_pos, A_tx, refConst,
                                            batch_size, nTestSamples, device)

            pd_inter.append(pd_temp)
            pfa_inter.append(pfa_temp)
            gospa_pos_inter.append(gospa_pos_temp)
            avg_dist_inter.append(avg_dist_temp)
            snr_inter.append(snr_temp.item())
            ser_inter.append(ser_temp.item())
            network.train()
    return loss_gospa_np, loss_comm_np, loss_avg_np, num_iterations, pd_inter, pfa_inter, gospa_pos_inter, avg_dist_inter, snr_inter, ser_inter


####################
# Saving functions #
####################
def saveNetwork(save_path, network, optimizer):
    '''
    Function to save the state dictionary of a network and the optimizer.
    It print a message at the end.
    Inputs:
        - save_path: path where to save the network, including model name. String.
        - network: network to be saved. Instance of class torch.nn.Module.
        - optimizer: optimizer to be saved. Instance of class torch.optim.
    '''
    torch.save({
        'model': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_path)
    print('Model saved successfully', flush=True)



#####################            
# Testing functions #
#####################
def getSensingMetrics(maxTargets, est_metric, true_numTargets, threshold, est_pos=None,
                      true_pos=None, gamma_gospa=None, mu_gospa=None, p_gospa=None):
    '''
    Function that computes detection and false alarm probabilities and angle,
    range, and position RMSEs.
    Inputs:
        - maxTargets: maximum number of considered targets. Integer
        - est_metric: metric to threshold. Float list. Shape: (nTestSamples, maxTargets)
        - true_numTargets: true number of targets. Integer list.
        Shape: (nTestSamples, 1)
        - threshold: threshold to discern if there is a target. Float.
        - est_pos: estimated positions. Float tensor of shape (2, nTestSamples, maxTargets)
        - true_pos: true positions. List of nTestSamples tensors, each containing a 2D tensor
        of shape (T,2), where T is the true number of targets for each sample. 
        - gamma_gospa: hyper-parameter gamma in the GOSPA loss function. Float.
        - mu_gospa: hyper-parameter alpha in the GOSPA loss function. Float, recommended
        to be equal to 2.
        - p_gospa: hyper-parameter p in the GOSPA loss function. Float.
    Output:
        - pd: detection probability. Float.
        - pfa: false alarm probability. Float.
        - gospa_pos: Position GOSPA loss. Float.
        - avg_dist_pos: Minimum distance error considering all permutations. Float
    '''
    device = est_metric.device

    #Get information from input data
    nTestSamples = len(true_numTargets)

    #Threshold metric to estimate the number of targets
    mask = est_metric >= threshold
    est_num_targets = torch.sum(mask, dim=1, keepdim=True)
    #Probability of correct detection
    pd = ( torch.minimum(est_num_targets, true_numTargets).sum()*1.0/true_numTargets.sum() ).item()
    #Probability of false alarm
    pfa =  ( (torch.maximum(est_num_targets, true_numTargets)-true_numTargets).sum()*1.0 \
            /(nTestSamples*maxTargets - true_numTargets.sum()) ).item()

    if (est_pos==None) or (true_pos==None):
        gospa_pos = None
        avg_dist_pos = None
    else:
        #Group angle, range and pos. estimations based on the estimated number of targets
        est_num_tgt_list = est_num_targets.flatten().tolist()
        est_pos_mask = est_pos[:,mask].transpose(-1,-2).view(-1,2)
        est_pos_thr = est_pos_mask.split(est_num_tgt_list)
        #GOSPA Position
        position_loss_gospa, avg_dist_loss = map(list,zip(*[GOSPALoss(est_pos_thr[i], true_pos[i], gamma_gospa, mu_gospa, p_gospa, device=device) for i in range(nTestSamples)]))
        gospa_pos = torch.mean(torch.stack(position_loss_gospa)).item()
        avg_dist_pos = torch.nanmean(torch.stack(avg_dist_loss)).item()

    return pd, pfa, gospa_pos, avg_dist_pos

def obtainThresholdsFixedPfa(maxTargets, est_metric, true_numTargets, target_pfa, delta_pfa, init_thr):
    '''
    Function that empirically estimates the thresholds to yield a target false
    alarm probability with a maximum allowable error.
    Since obtaining exactly pfa = target_pfa is very difficult, we compute 3
    pfa's, so that for any of those probabilities,
    target_pfa - delta_pfa < pfa < target_pfa + delta_pfa.
    We then linearly interpolate the results.
    Inputs:
        - maxTargets: maximum number of targets assumed in the scene. Integer.
        - est_metric: metric to threshold to estimate the number of targets.
        Float list of length nTestSamples.
        - true_presence: true number of targets per batch sample. Integer list
        of length nTestSamples.
        - target_pfa: target false alarm proability. Float.
        - delta_pfa: maximum allowable error for the target_pfa. Float.
        - init_thr: initial thresholds to start the algorithm. Float numpy array
        with more than 1 element (usually 3).
    Outputs
        - init_thr: final thresholds that achieve target_pfa - delta_pfa < pfa,
        pfa < target_pfa + delta_pfa.
    '''
    device = est_metric.device

    with torch.no_grad():
        #Reset Pfa
        pfa = np.zeros((1,len(init_thr)))   #Set initial Pfa to enter loop
        while ((np.max(pfa) > target_pfa + delta_pfa) or (np.min(pfa) < target_pfa - delta_pfa)):
            #Lists to save final results
            pd, pfa = [], []

            #Compute detection and false alarm probabilities
            for t in range(len(init_thr)):
                pd_temp, pfa_temp, _, _ = getSensingMetrics(maxTargets, est_metric, true_numTargets, init_thr[t])
                pd.append(pd_temp)
                pfa.append(pfa_temp)

            #Check if target_pfa - delta_pfa < pfa < target_pfa + delta_pfa. We use the std to update the threshold vector
            if target_pfa < np.min(pfa):
                init_thr += torch.std(init_thr)
            elif target_pfa > np.max(pfa):
                init_thr -= torch.std(init_thr)
            else:
                #Check that the pfa is not e.g [1,0,0]
                if np.max(pfa) > target_pfa + delta_pfa:
                    init_thr = torch.linspace((init_thr[0] + init_thr[1]) / 2.0, init_thr[2], 3, device=device)
                if np.min(pfa) < target_pfa - delta_pfa:
                    init_thr = torch.linspace(init_thr[0], (init_thr[1] + init_thr[2]) / 2.0, 3, device=device)

    return init_thr

def computePositionsTesting(maxTargets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                            theta_span_min_sens_test, theta_span_max_sens_test,
                            range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                            range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                            numAntennas, numSubcarriers, varNoise, Delta_f, lamb, ant_pos, A_tx, rx_feature, refConst,
                            batch_size, nTestSamples, dyn_flag=True, device='cuda'):
    '''
    Function that computes the estimated targets' positions and metric to threshold during sensing testing.
    Inputs: 
        - (Check function 'trainNetworkFixedEta' for a description of most of the inputs)
    Outputs:
        - est_max_admap_list: maximum of the angle-delay map corresponding to each target peak. Float tensor of
        shape (nTestSamples, maxTargets)
        - true_numTargets_list: true number of targets in each iteration. Float tensor of shape (nTestSamples, 1).
        - est_pos_list: estimated positions of all potential targets (more estimated targets than needed). 
        Float tensor of shape (2, nTestSamples, maxTargets). We save x and y coordinates.
        - true_pos_list: true positions of targets. List of nTestSamples tensors, each of 
        size (T,2), where T is the true number of targets in an iteration.
    '''
    with torch.no_grad():
        #Get information from the input data
        msg_card = len(refConst)
        numTestIt = nTestSamples // batch_size

        #Create lists to save results in each iteration
        true_numTargets_list = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        true_angle_list      = []
        true_range_list      = []
        est_max_admap_list   = torch.empty(nTestSamples, maxTargets, dtype=torch.float32, device=device)
        est_angle_list       = torch.empty(nTestSamples, maxTargets, dtype=torch.float32, device=device)
        est_range_list       = torch.empty(nTestSamples, maxTargets, dtype=torch.float32, device=device)

        for i in range(numTestIt):
            #Generate random number of targets in the scene
            numTargets = torch.randint(0,maxTargets+1,(batch_size,), device=device)
            #Generate random messages to transmit. 
            msg = torch.randint(0,msg_card,size=(batch_size*numSubcarriers,), dtype=torch.int64, device=device)
            symbols = refConst[msg].reshape(batch_size, numSubcarriers, 1)

            #Choose random angle interval vector (random mean and random span)
            theta_min, theta_max = generateInterval(theta_mean_min_sens_test, theta_mean_max_sens_test, theta_span_min_sens_test, theta_span_max_sens_test, batch_size, device=device)
            #Choose random range interval vector (random mean and random span)
            r_min, r_max = generateInterval(range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, batch_size, device=device)
            #Create precoder for testing
            precoder_sens, _ = createPrecoder(theta_min, theta_max, A_tx, Ngrid_angle, P_power, device=device)

            #Sensing channel
            Y_radar, true_angle, true_range = radarCH(mean_rcs, theta_min, theta_max, r_min, r_max, Delta_f, 
                                                    numAntennas, ant_pos, lamb, numSubcarriers, 
                                                    precoder_sens, symbols, numTargets, varNoise)
            #Return true angle and ranges as lists of tensors
            true_angle = true_angle.flatten().split(numTargets.to(torch.int).tolist())
            true_range = true_range.flatten().split(numTargets.to(torch.int).tolist())

            ### Sensing Receiver (nondifferentiable OMP) ###
            #Remove the effect of the transmitted symbols (since the radar Rx is the same as the comm Tx)
            Y_radar /= torch.ones((numAntennas,1), device=device).type(torch.cfloat) @ symbols.permute(0,2,1)
            if dyn_flag:
                #Compute A_matrix dynamically
                delta_angle = (theta_max-theta_min)/(Ngrid_angle-1)
                theta_grid_matrix = theta_min + delta_angle*torch.arange(Ngrid_angle, device=device).reshape(1,-1)  #Shape: (batch_size, numSamplesAngle)
                A_rx, _ = steeringMatrix(theta_grid_matrix, theta_grid_matrix, rx_feature, lamb) #Shape: (numAntennas, batch_size*numSamplesRange)
                A_rx = A_rx.transpose(-1,-2).reshape(batch_size, Ngrid_angle, numAntennas).permute(0,2,1)
                max_admap, est_angle, est_range = OMPReceiver(Y_radar, theta_min, theta_max, r_min, r_max, Delta_f, 
                                                            maxTargets, A_rx, Ngrid_angle, Ngrid_range)
            else:
                max_admap, est_angle, est_range = OMPReceiverFixed(Y_radar, theta_min, theta_max, 
                                                            r_min, r_max, Delta_f, maxTargets, 
                                                            rx_feature, Ngrid_angle, Ngrid_range)
            #Save true values
            true_numTargets_list[i*batch_size:(i+1)*batch_size, 0] = numTargets
            true_angle_list.extend(true_angle)
            true_range_list.extend(true_range)
            #Save estimations
            est_max_admap_list[i*batch_size:(i+1)*batch_size, :] = max_admap
            est_angle_list[i*batch_size:(i+1)*batch_size, :] = est_angle
            est_range_list[i*batch_size:(i+1)*batch_size, :] = est_range

        #Compute positions from angle and range
        est_x = est_range_list * torch.cos(est_angle_list)
        est_y = est_range_list * torch.sin(est_angle_list)
        est_pos_list = torch.cat((est_x[None,:,:], est_y[None,:,:]), dim=0)
        #For true position note that range and angle are lists of tensors
        true_pos_list = [torch.cat(( (true_range_list[i]*torch.cos(true_angle_list[i])).view(-1,1), 
                                    (true_range_list[i]*torch.sin(true_angle_list[i])).view(-1,1) ), dim=1) 
                                    for i in range(len(true_angle_list))]

    return est_max_admap_list, true_numTargets_list, est_pos_list, true_pos_list



def testSensingROC(maxTargets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                   theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                   range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, 
                   Ngrid_angle, Ngrid_range,
                   numAntennas, numSubcarriers, varNoise, Delta_f, lamb, ant_pos, A_tx, rx_feature, refConst,
                   thresholds, gamma_gospa, mu_gospa, p_gospa, batch_size, nTestSamples,
                   dyn_flag=True, device='cuda'):
    '''
    Function to test the sensing performance as a function of the false alarm
    probability.
    Inputs:
        - A_tx: steering matrix to compute the precoder with the assumed TX positions. 
        Complex tensor of shape (numAntennas, Ngrid_angle)
        - thresholds: list of thresholds to evaluate the sensing performance. List of floats.
    Outputs:
        - pd: probability of detection for each threshold. List of Floats.
        - pfa: probability of false alarm for each threshold. List of Floats.
        - gospa_pos: GOSPA loss for each threshold. List of Floats.
        - avg_dist_pos: GOSPA loss w/o cardinality mismatch and taking average intead of a summation for each threshold. 
        List of Floats.
    '''
    print('**STARTED TESTING TO OBTAIN ROC CURVES**')
    with torch.no_grad():
        est_metric, true_numTargets, est_pos, true_pos = computePositionsTesting(maxTargets, P_power, mean_rcs, 
                                                                                 theta_mean_min_sens_test, theta_mean_max_sens_test, 
                                                                                 theta_span_min_sens_test, theta_span_max_sens_test, 
                                                                                 range_mean_min_sens_test, range_mean_max_sens_test, 
                                                                                 range_span_min_sens_test, range_span_max_sens_test, 
                                                                                 Ngrid_angle, Ngrid_range, numAntennas, 
                                                                                 numSubcarriers, varNoise, Delta_f, 
                                                                                 lamb, ant_pos, A_tx, rx_feature, refConst,
                                                                                 batch_size, nTestSamples, dyn_flag, 
                                                                                 device=device)
        
        #Lists to save final results
        pd, pfa, gospa_pos, avg_dist_pos = [], [], [], []
        #Compute detection and false alarm probabilities, and RMSEs
        for t in range(len(thresholds)):
            pd_temp, pfa_temp, gospa_pos_temp, avg_dist_temp = getSensingMetrics(maxTargets, est_metric, true_numTargets, 
                                                                  thresholds[t], est_pos, true_pos, 
                                                                  gamma_gospa, mu_gospa, p_gospa)
            pd.append(pd_temp)
            pfa.append(pfa_temp)
            gospa_pos.append(gospa_pos_temp)
            avg_dist_pos.append(avg_dist_temp)

    return pd, pfa, gospa_pos, avg_dist_pos



#This function only works for a single testing case, not for random cases


def testSensingFixedPfa(maxTargets, P_power, mean_rcs, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                       theta_span_min_sens_test, theta_span_max_sens_test, range_mean_min_sens_test,
                       range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, 
                       Ngrid_angle, Ngrid_range,
                       numAntennas, numSubcarriers, varNoise, Delta_f, lamb, ant_pos, A_tx, rx_feature, refConst,
                       target_pfa, delta_pfa, thresholds, gamma_gospa, mu_gospa, p_gospa, batch_size, nTestSamples,
                       dyn_flag=True, device='cuda'):
    '''
    Function to test the sensing performance as a function of the false alarm
    probability.
    Inputs:
       - (Please, check the end of the document or the README_LIB.md file under the lib/ folder)
    Outputs:
        - pd_inter: probability of detection. Float.
        - pfa_inter: probability of false alarm (to check with target_pfa). Float.
        - gospa_pos_inter: GOSPA loss. Float.
        - avg_dist_inter: GOSPA loss w/o cardinality mismatch and averaging instead of summing. Float.
    '''
    print('**STARTED TESTING WITH FIXED PFA**')
    with torch.no_grad():
        est_metric, true_numTargets, est_pos, true_pos = computePositionsTesting(maxTargets, P_power, mean_rcs, 
                                                                                 theta_mean_min_sens_test, theta_mean_max_sens_test, 
                                                                                 theta_span_min_sens_test, theta_span_max_sens_test,
                                                                                 range_mean_min_sens_test, range_mean_max_sens_test, 
                                                                                 range_span_min_sens_test, range_span_max_sens_test,
                                                                                 Ngrid_angle, Ngrid_range, numAntennas, 
                                                                                 numSubcarriers, varNoise, Delta_f, 
                                                                                 lamb, ant_pos, A_tx, rx_feature, refConst,
                                                                                 batch_size, nTestSamples, dyn_flag, 
                                                                                 device=device)
        
        #Get thresholds that give relatively close to the target Pfa
        init_thr = torch.clone(thresholds)      #To avoid that the next function overwrites the thresholds.
        final_thr = obtainThresholdsFixedPfa(maxTargets, est_metric, true_numTargets, target_pfa, delta_pfa, init_thr)

        #Lists to save final results
        pd, pfa, gospa_pos, avg_dist_pos = [], [], [], []
        #Compute detection and false alarm probabilities, and RMSEs
        for t in range(len(final_thr)):
            pd_temp, pfa_temp, gospa_pos_temp, avg_dist_temp = getSensingMetrics(maxTargets, est_metric, true_numTargets, 
                                                                  final_thr[t], est_pos, true_pos, 
                                                                  gamma_gospa, mu_gospa, p_gospa)
            pd.append(pd_temp)
            pfa.append(pfa_temp)
            gospa_pos.append(gospa_pos_temp)
            avg_dist_pos.append(avg_dist_temp)
        
        #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa, but interpolation expects x-values to be ordered
        pfa_ordered = np.sort(pfa)
        idx_order = np.argsort(pfa)         #Save indices to keep the correspondence of pd-pfa values
        pd_inter = np.interp(-2,np.log10(pfa_ordered),np.array(pd)[idx_order])     
        pfa_inter = np.array(pfa).mean()
        gospa_pos_inter = np.interp(-2,np.log10(pfa_ordered),np.array(gospa_pos)[idx_order])
        avg_dist_inter = np.interp(-2,np.log10(pfa_ordered),np.array(avg_dist_pos)[idx_order])

    return pd_inter, pfa_inter, gospa_pos_inter, avg_dist_inter

def testCommunication(maxPaths, P_power, mean_rcs, 
                      theta_min_mean, theta_max_mean, theta_min_span, theta_max_span, 
                      range_min_mean, range_max_mean, range_min_span, range_max_span,
                      Ngrid_angle, numAntennas, numSubcarriers, varNoise, 
                      Delta_f, Rcp, lamb, ant_pos, A_tx, refConst,
                      batch_size, nTestSamples, device='cpu'):
    '''
    Inputs:
        - maxPaths: maximum number of paths from the ISAC transceiver to the UE, including LoS. Integer.
        - (The angles and ranges play the same role as in the training functions, please see the end of 
        the document of the README_LIB.md file under the lib/ folder).
    Outputs:
        - snr_total: estimated SNR with the current A_tx [dB]. Float
        - ser: estimated symbol error rate. Float
    '''
    numIterations = nTestSamples // batch_size
    constCard = len(refConst)

    #Tensors to save intermediate results (initialize to nonsense results)
    num_errors = torch.tensor(0,dtype=torch.float32, device=device)
    mean_abs_kappa_sq = torch.zeros(numSubcarriers,1, device=device)

    with torch.no_grad():
        for b in range(numIterations):
            #== Transmitter ==#
            #Create precoder for testing
            theta_min, theta_max = generateInterval(theta_min_mean, theta_max_mean, theta_min_span, theta_max_span, batch_size, device=device)
            precoder, _ = createPrecoder(theta_min, theta_max, A_tx, Ngrid_angle, P_power, device=device)
            range_min, range_max = generateInterval(range_min_mean, range_max_mean, range_min_span, range_max_span, batch_size, device=device)
            #Generate number of communication paths (LOS always present)
            numPaths = torch.randint(1,maxPaths+1,(batch_size,), device=device)
            #Generate random messages to transmit
            msg = torch.randint(0,constCard,size=(batch_size*numSubcarriers,), dtype=torch.int64, device=device)
            #Reshape msgs to the needs of the comm CH function
            q_tx = refConst[msg].reshape(batch_size, numSubcarriers, 1)

            #== Communication Channel ==#
            y_comm, _, _, kappa =  commChannel(mean_rcs, theta_min, theta_max, range_min, range_max, Delta_f, numAntennas, ant_pos, 
                                               lamb, numSubcarriers, precoder, q_tx, numPaths, Rcp, varNoise, device=device)

            #== Communication Receiver ==#
            est_messages = MLdecoder(y_comm, kappa, refConst)

            #Save results for later evaluation
            num_errors += (msg != est_messages).sum()
            mean_abs_kappa_sq += torch.mean(torch.abs(kappa)**2, dim=0)

            if (b+1)%500 == 0:
                print(f'Iteration {b+1} out of {numIterations}', flush=True)

        # Compute SNR and SER
        mean_abs_kappa_sq /= numIterations
        snr_per_subcarrier = 10*torch.log10(mean_abs_kappa_sq/varNoise)
        snr_total = torch.mean(snr_per_subcarrier)
        ser = num_errors/(nTestSamples * numSubcarriers)

    return snr_total, ser

##################
# ISAC functions #
##################
def testISAC(maxTargets_sens, maxPaths_comm, P_power, mean_rcs, theta_min_mean_sens, theta_max_mean_sens, 
             theta_min_span_sens, theta_max_span_sens, theta_min_mean_comm, theta_max_mean_comm, 
             theta_min_span_comm, theta_max_span_comm, range_min_mean_sens, range_max_mean_sens, 
             range_min_span_sens, range_max_span_sens, range_min_mean_comm, range_max_mean_comm, 
             range_min_span_comm, range_max_span_comm,
             Ngrid_angle, Ngrid_range, numAntennas, numSubcarriers, Rcp, varNoise, Delta_f, lamb, ant_pos, 
             A_tx, rx_feature, refConst, target_pfa, delta_pfa, thresholds, eta_list, phi_list, gamma_gospa, mu_gospa, p_gospa, 
             batch_size, nTestSamples, dyn_flag=True, device='cuda'):
    '''
    Function to test the sensing performance as a function of the false alarm
    probability.
    Inputs:
        - maxTargets_sens: maximum number of targets in the environment. Integer.
        - maxPaths_comm: maximum number of paths from ISAC transceiver to UE,
        including LoS. Integer.
        - eta_list: list of parameters controlling how much power is allocated to the sensing
        precoder compared to communications. List of Floats.
        - phi: list of parameters controlling the phase to add to the comm. precoder. List of
        Floats.
    Outputs:
        - ser_isac: symbol error rate for each (eta, phi) pair. List of Floats.
        - pd_inter: probability of detection for each (eta, phi) pair. List of Floats.
        - pfa_inter: probability of false alarm for each (eta, phi) pair. List of Floats.
        - gospa_pos_isac: GOSPA loss for each (eta, phi) pair. List of Floats.
        - avg_dist_isac: GOSPA loss w/o cardinality mismatch and averaging instead of 
        summing for each (eta, phi) pair. List of Floats.
    '''
    print('**STARTED ISAC TESTING**')
    with torch.no_grad():
        #Get information from the input data
        msg_card = len(refConst)
        numTestIt = nTestSamples // batch_size

        #Tensors to save at the end
        pd_isac, pfa_isac, ser_isac, gospa_pos_isac, avg_dist_isac = [],[],[],[],[]

        for k in range(len(eta_list)):
            for l in range(len(phi_list)):
                #Create lists to save results in each iteration
                true_numTargets_list = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                true_angle_list      = []
                true_range_list      = []
                est_max_admap_list   = torch.empty(nTestSamples, maxTargets_sens, dtype=torch.float32, device=device)
                est_angle_list       = torch.empty(nTestSamples, maxTargets_sens, dtype=torch.float32, device=device)
                est_range_list       = torch.empty(nTestSamples, maxTargets_sens, dtype=torch.float32, device=device)
                num_errors           = torch.tensor(0,device=device)


                for i in range(numTestIt):
                    #Generate random number of targets and paths in the scene
                    numTargets_sens = torch.randint(0,maxTargets_sens+1,(batch_size,), device=device)
                    numPaths_comm = torch.randint(1,maxPaths_comm+1,(batch_size,), device=device)       #We always have the LOS path
                    #Generate random angular and range sectors for sensing and communication
                    theta_min_sens, theta_max_sens = generateInterval(theta_min_mean_sens, theta_max_mean_sens, theta_min_span_sens, 
                                                                      theta_max_span_sens, batch_size, device=device)
                    theta_min_comm, theta_max_comm = generateInterval(theta_min_mean_comm, theta_max_mean_comm, theta_min_span_comm, 
                                                                      theta_max_span_comm, batch_size, device=device)
                    range_min_sens, range_max_sens = generateInterval(range_min_mean_sens, range_max_mean_sens, range_min_span_sens, 
                                                                      range_max_span_sens, batch_size, device=device)
                    range_min_comm, range_max_comm = generateInterval(range_min_mean_comm, range_max_mean_comm, range_min_span_comm, 
                                                                      range_max_span_comm, batch_size, device=device)
                    #Create precoder by a weighted sum of sensing and communication precoders
                    precoder_sens, _ = createPrecoder(theta_min_sens, theta_max_sens, A_tx, Ngrid_angle, P_power, device=device)
                    precoder_comm, _ = createPrecoder(theta_min_comm, theta_max_comm, A_tx, Ngrid_angle, P_power, device=device)
                    precoder = createCommonPrecoder(precoder_sens, precoder_comm, eta_list[k], phi_list[l])   #Shape: (batch_size, K ,1)
                    #Generate random messages to transmit. 
                    msg = torch.randint(0,msg_card,size=(batch_size*numSubcarriers,), dtype=torch.int64, device=device)
                    symbols = refConst[msg].reshape(batch_size, numSubcarriers, 1)

                    #Sensing channel
                    Y_radar, true_angle, true_range = radarCH(mean_rcs, theta_min_sens, theta_max_sens, 
                                                            range_min_sens, range_max_sens, Delta_f, 
                                                            numAntennas, ant_pos, lamb, numSubcarriers, 
                                                            precoder, symbols, numTargets_sens, varNoise)
                    #Return true angle and ranges as lists of tensors
                    true_angle = true_angle.flatten().split(numTargets_sens.to(torch.int).tolist())
                    true_range = true_range.flatten().split(numTargets_sens.to(torch.int).tolist())

                    #Comm. channel
                    Y_comm, _, _, kappa = commChannel(mean_rcs, theta_min_comm, theta_max_comm, 
                                                      range_min_comm, range_max_comm, Delta_f, numAntennas, 
                                                      ant_pos, lamb, numSubcarriers, precoder, symbols, 
                                                      numPaths_comm, Rcp, varNoise, device=device)

                    ### Sensing Receiver (nondifferentiable OMP) ###
                    #Remove the effect of the transmitted symbols (since the radar Rx is the same as the comm Tx)
                    Y_radar /= torch.ones((numAntennas,1), device=device).type(torch.cfloat) @ symbols.permute(0,2,1)
                    if dyn_flag:
                        #Compute A_matrix dynamically
                        delta_angle = (theta_max_sens-theta_min_sens)/(Ngrid_angle-1)
                        theta_grid_matrix = theta_min_sens + delta_angle*torch.arange(Ngrid_angle, device=device).reshape(1,-1)  #Shape: (batch_size, numSamplesAngle)
                        A_rx, _ = steeringMatrix(theta_grid_matrix, theta_grid_matrix, rx_feature, lamb) #Shape: (numAntennas, batch_size*numSamplesRange)
                        A_rx = A_rx.transpose(-1,-2).reshape(batch_size, Ngrid_angle, numAntennas).permute(0,2,1)
                        max_admap, est_angle, est_range = OMPReceiver(Y_radar, theta_min_sens, theta_max_sens, 
                                                                      range_min_sens, range_max_sens, Delta_f, 
                                                                      maxTargets_sens, A_rx, Ngrid_angle, Ngrid_range)
                    else:
                        max_admap, est_angle, est_range = OMPReceiver(Y_radar, -np.pi/2.0, np.pi/2.0, 
                                                                    range_min_sens, range_max_sens, Delta_f, maxTargets_sens, 
                                                                    rx_feature, Ngrid_angle, Ngrid_range)

                    ### Communication Receiver ###
                    est_messages = MLdecoder(Y_comm, kappa, refConst)

                    #Save true values
                    true_numTargets_list[i*batch_size:(i+1)*batch_size, 0] = numTargets_sens
                    true_angle_list.extend(true_angle)
                    true_range_list.extend(true_range)
                    #Save estimations and compute the number of symbol errors
                    est_max_admap_list[i*batch_size:(i+1)*batch_size, :] = max_admap
                    est_angle_list[i*batch_size:(i+1)*batch_size, :] = est_angle
                    est_range_list[i*batch_size:(i+1)*batch_size, :] = est_range
                    num_errors += (msg.flatten() != est_messages.flatten()).sum()

                #Compute positions from angle and range
                est_x = est_range_list * torch.cos(est_angle_list)
                est_y = est_range_list * torch.sin(est_angle_list)
                est_pos_list = torch.cat((est_x[None,:,:], est_y[None,:,:]), dim=0)
                #For true position note that range and angle are lists of tensors
                true_pos_list = [torch.cat(( (true_range_list[i]*torch.cos(true_angle_list[i])).view(-1,1), 
                                            (true_range_list[i]*torch.sin(true_angle_list[i])).view(-1,1) ), dim=1) 
                                            for i in range(len(true_angle_list))]
                
                # ================= CALCULATE SER =======================
                ser = num_errors/(nTestSamples * numSubcarriers)
                #Add SER to vector
                ser_isac.append(ser.item())

                # ================= CALCULATE sensing metrics =======================
                #Get thresholds that give relatively close to the target Pfa
                init_thr = torch.clone(thresholds)      #To avoid that the next function overwrites the thresholds.
                final_thr = obtainThresholdsFixedPfa(maxTargets_sens, est_max_admap_list, true_numTargets_list, target_pfa, delta_pfa, init_thr)

                #Lists to save final results
                pd, pfa, gospa_pos, avg_dist_pos = [], [], [], []
                #Compute detection and false alarm probabilities, and RMSEs
                for t in range(len(final_thr)):
                    pd_temp, pfa_temp, gospa_pos_temp, avg_dist_temp = getSensingMetrics(maxTargets_sens, est_max_admap_list, true_numTargets_list, 
                                                                        final_thr[t], est_pos_list, true_pos_list, 
                                                                        gamma_gospa, mu_gospa, p_gospa)
                    pd.append(pd_temp)
                    pfa.append(pfa_temp)
                    gospa_pos.append(gospa_pos_temp)
                    avg_dist_pos.append(avg_dist_temp)
                
                #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa, but interpolation expects x-values to be ordered
                pfa_ordered = np.sort(pfa)
                idx_order = np.argsort(pfa)         #Save indices to keep the correspondence of pd-pfa values
                pd_isac.append( np.interp(-2,np.log10(pfa_ordered),np.array(pd)[idx_order]) )
                pfa_isac.append(np.array(pfa).mean())
                gospa_pos_isac.append( np.interp(-2,np.log10(pfa_ordered),np.array(gospa_pos)[idx_order]) )
                avg_dist_isac.append( np.interp(-2,np.log10(pfa_ordered),np.array(avg_dist_pos)[idx_order]) )

    return ser_isac, pd_isac, pfa_isac, gospa_pos_isac, avg_dist_isac


def baselineLearningSeveralTargets(iterations, maxTargets, P_power, mean_rcs, numAntennas, 
                                   numSubcarriers, Delta_f, angle_grid, Ngrid_range, 
                                    assumed_pos, ant_pos, est_ant_pos, spacings_try, 
                                    varNoise, lamb, msg_card, refConst,
                                    theta_mean_min_sens_train, theta_mean_max_sens_train,
                                    theta_span_min_sens_train, theta_span_max_sens_train,
                                    range_mean_min_sens_train, range_mean_max_sens_train,
                                    range_span_min_sens_train, range_span_max_sens_train, batch_size,
                                    gamma_gospa_train, mu_gospa, p_gospa, device='cuda'):
    '''
    Function that performs learning of the inter-antenna spacing distances applying a greedy approach.
    For each batch of observations, we try Ngrid samples of the inter-antenna spacing. We first 
    optimize the spacing between the first and the second element. With the best spacing, we proceed
    with the spacing between the 2nd and the 3rd element, and so on.
    The best spacing is determined by the MSE between the estimated and true positions.
    Inputs:
        - iterations: number of iterations to perform the greedy search. Integer.
        - assumed_pos: positions assumed to perform beamforming (lambda/2 spacing). 
        Float tensor of shape (numAntennas,)
        - ant_pos: true positions of the antenna array. Float tensor of shape (numAntennas,)
        - est_ant_pos: estimated antenna positions to be learned (the first element is 
        forced to be the same as the true one, just reference). Float tensor of shape (numAntennas, )
        - spacings_try: list of inter-antenna spacings to try. Float tensor of shape (Ngrid_spacing, )
    Output:
        - est_ant_pos: calibrated antenna positions. Float tensor of shape (numAntennas,)
        - best_gospa_list: list containing the GOSPA loss for each optimization step. List of FLoats
        of length numAntennas*iterations. 
    '''
    #Infer parameters from input data
    num_try = len(spacings_try)
    Ngrid_angle = len(angle_grid)

    #Force the first element of the estimated positions to be the true one (reference)
    est_ant_pos[0] = ant_pos[0]

    #List to save the best MSE loss for each saved antenna index
    best_gospa_list = []

    #This will be useful later on
    tgt_arange = torch.arange(maxTargets, device=device)
    
    with torch.no_grad():
        for m in range(iterations):
            #1. Get a batch of observations assuming lambda/2 spacing for precoder 
            print(f'Iteration {m+1}/{iterations}', flush=True)
            #Update matrix (K x Ngrid_angle) of steering vectors that will be used later for learning
            A_matrix_tx, _ = steeringMatrix(angle_grid, angle_grid, assumed_pos, lamb)

            #Generate random number of targets in the environemt, but always 1 present
            numTargets = torch.randint(1,maxTargets+1,(batch_size,), device=device)
            mask_tgt = tgt_arange < numTargets.reshape(-1,1)
            
            #Generate random messages to transmit. 
            msg = torch.randint(0,msg_card,size=(batch_size*numSubcarriers,), dtype=torch.int64, device=device)
            #Reshape msgs to the needs of the comm CH function
            symbols = refConst[msg].reshape(batch_size, numSubcarriers, 1)
            
            # ================= TRANSMITTER =======================
            #Choose random angle interval vector (random mean and random span)
            theta_min_sens, theta_max_sens = generateInterval(theta_mean_min_sens_train, theta_mean_max_sens_train, theta_span_min_sens_train, theta_span_max_sens_train, batch_size, device=device)
            #Choose random range interval vector (random mean and random span)
            r_min_sens, r_max_sens = generateInterval(range_mean_min_sens_train, range_mean_max_sens_train, range_span_min_sens_train, range_span_max_sens_train, batch_size, device=device)
            #Beamformer 
            precoder_sens, b_matrix_angle_sens = createPrecoder(theta_min_sens, theta_max_sens, A_matrix_tx, Ngrid_angle, P_power, device=device)
            
            # ================= RADAR CHANNEL ================= #
            Y_radar, true_angle, true_range = radarCH(mean_rcs, theta_min_sens, theta_max_sens, r_min_sens, r_max_sens, Delta_f, 
                                                    numAntennas, ant_pos, lamb, numSubcarriers, precoder_sens, symbols, numTargets, varNoise)
            
            #Return true angle and ranges as lists of tensors
            true_angle = true_angle.flatten().split(numTargets.to(torch.int).tolist())
            true_range = true_range.flatten().split(numTargets.to(torch.int).tolist())
            # ================= RADAR RECEIVER ================= #
            #Remove the effect of the transmitted symbols (since the radar Rx is the same as the comm Tx)
            Y_radar /= torch.ones((numAntennas,1), device=device).type(torch.cfloat) @ symbols.permute(0,2,1)

            #Compute true position
            true_pos_list = [torch.cat(( (true_range[i]*torch.cos(true_angle[i])).view(-1,1), 
                                    (true_range[i]*torch.sin(true_angle[i])).view(-1,1) ), dim=1) 
                                    for i in range(len(true_angle))]

            #Compute a few things outside the main loop
            delta_angle = (theta_max_sens-theta_min_sens)/(Ngrid_angle-1)
            theta_grid_matrix = theta_min_sens + delta_angle*torch.arange(Ngrid_angle, device=device).reshape(1,-1)  #Shape: (batch_size, Ngrid_angle)
            
            #2. Try different impairments and apply greedy search to find the best antenna position vector
            for k in range(1,numAntennas):
                # print(f'Optimizing spacing {k}/{numAntennas-1}', flush=True)

                #List to save the loss function values and the tested iterations
                loss_gospa = []
                
                for n in range(num_try):
                    #Compute antenna positions based on the grid sample of the spacing
                    distance_try = spacings_try[n]
                    est_ant_pos[k] = est_ant_pos[k-1] + distance_try
                    # A_matrix_temp, _ = steeringMatrix(angle_grid, angle_grid, est_ant_pos, lamb)
                    #Compute A_matrix dynamically
                    A_matrix_temp, _ = steeringMatrix(theta_grid_matrix, theta_grid_matrix, est_ant_pos, lamb) #Shape: (numAntennas, batch_size*numSamplesRange)
                    A_matrix_temp = A_matrix_temp.transpose(-1,-2).reshape(batch_size, Ngrid_angle, numAntennas).permute(0,2,1)
                
                    #Estimate the position of a single target in the scene
                    # _, est_angle, est_range = diffOMPReceiver(Y_radar, A_matrix_temp, b_matrix_angle_sens, angle_grid, r_min_sens, r_max_sens, Ngrid_range, Delta_f, pixels_angle, pixels_range, maxTargets=1)
                    _, est_angle, est_range = OMPReceiver(Y_radar, theta_min_sens, theta_max_sens, r_min_sens, r_max_sens, Delta_f, maxTargets, A_matrix_temp, Ngrid_angle, Ngrid_range)
                    
                    #Compute positions from angle and range
                    est_x = est_range * torch.cos(est_angle)
                    est_y = est_range * torch.sin(est_angle)
                    est_pos= torch.cat((est_x[None,:,:], est_y[None,:,:]), dim=0)   #Shape: (2, batch_size, maxTargets)

                    #Compute estimated position, but take only the true number of targets
                    numTargets_list = numTargets.flatten().tolist()
                    est_pos_mask = est_pos[:,mask_tgt].transpose(-1,-2).view(-1,2)
                    est_pos_thr = est_pos_mask.split(numTargets_list)
                    #GOSPA Position
                    position_loss_gospa, _ = map(list,zip(*[GOSPALoss(est_pos_thr[i], true_pos_list[i], gamma_gospa_train, mu_gospa, p_gospa, device=device) for i in range(batch_size)]))
                    gospa_pos = torch.mean(torch.stack(position_loss_gospa)).item()
        
                    #Compute the MSE for that sample of the inter-antenna spacing and save it
                    loss_gospa.append(gospa_pos)

                #Compute the best sample of the inter-antenna spacing and fix that antenna position
                best_index = np.argmin(loss_gospa)
                best_gospa_list.append(np.min(loss_gospa))
                est_ant_pos[k] = est_ant_pos[k-1] + spacings_try[best_index]
    return est_ant_pos, best_gospa_list

'''
Inputs to several functions in this file:
- network: network with learnable parameters to optimize.
- train_it: number of training iterations. Integer.
- maxTargets: maximum number of targets in the radar channel. Integer.
- maxPaths or maxPaths_comm: maximum number of scatterers for communications. Integer.
- P_power: transmitted power. Float.
- mean_rcs: mean value of the radar cross section of the targets. Float.
- Rcp: equivalent range of the cyclic prefix time. Real tensor.
- theta_mean_min_sens_train or theta_mean_min_comm_train: minimum value of the mean of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- theta_mean_max_sens_train or theta_mean_max_comm_train: maximum value of the mean of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- theta_span_min_sens_train or theta_span_min_comm_train: minimum value of the span of the a priori angular
sector of the targets or the UE. Real tensor of shape (batch_size, 1).
- theta_span_max_sens_train or theta_span_max_comm_train: maximum value of the span of the a priori angular
sector of the targets. Real tensor of shape (batch_size, 1).
- range_mean_min_sens_train or range_mean_min_comm_train: minimum value of the mean of the a priori range
sector of the targets. Real tensor of shape (batch_size, 1).
- range_mean_max_sens_train or range_mean_max_comm_train: maximum value of the mean of the a priori range
sector of the targets. Real tensor of shape (batch_size, 1).
- range_span_min_sens_train or range_span_min_comm_train: minimum value of the span of the a priori range
sector of the targets. Real tensor of shape (batch_size, 1).
- range_span_max_sens_train or range_span_max_comm_train: maximum value of the span of the a priori range
sector of the targets. Real tensor of shape (batch_size, 1).
- (The theta_ and range_ variables before also have a _test counterpart for inference)
- numAntennas or K: number of antennas of the ISAC transceiver. Integer.
- numSubcarriers or S: number of subcarriers of the OFDM signal. Integer.
- Delta_f: subcarrier spacing. Float.
- lamb: wavelength. Float.
- Ngrid_angle: number of candidate angles to try in beamforming and at the Rx.
Integer.
- Ngrid_range: number of candidate ranges to try in beamforming and at the Rx.
Integer.
- angle_grid: grid of angles to try. Real tensor of shape (Ngrid_angle, )
- range_grid: grid of ranges to try. Real tensor of shape (Ngrid_range, )
- pixels_angle: number of angular elements to consider in the angle-delay map around
the maximum value (see OMP functions for more details). Integers.
- pixels_range: number of range elements to consider in the angle-delay map around
the maximum value (see OMP functions for more details). Integers.
- msg_card: cardinality of the constellation of messages for communication. Integer.
- refConst: constellation to use for communications. Complex tensor of shape
(msg_card,).
- varNoise or N0: variance of the additive noise at the Rx. Real value.
- ant_pos: true (impaired) antenna positions. Real tensor of shape (numAntennas, 1).
- batch_size: number of samples to use in one batch. Integer.
- optimizer: optimizer to update the parameters to learn.
- scheduler: scheduler to change the learning rate over training iterations. It may
be activated or deactivated depending on the binary flag sch_flag.
- loss_comm_fn: loss function to use for communication training.
- eta_weight: weight to combine sensing and communication loss functions. Real 
value between 0 and 1.
- epoch_test_list: list of epochs at which the network performance should be 
tested during training. List of integers.
- target_pfa: target value of the false alarm probability. Real value.
- delta_pfa: maximum allowable error w.r.t target_pfa. Real value.
- thresholds_pfa: thresholds to distinguish if a measurement is actually a
target. List of real values.
- gamma_gospa_train: value of the gamma parameter in the GOSPA loss. Float
- mu_gospa: value of the mu parameter in the GOSPA loss. Float between 0 (exclusive) and 2.
- p_gospa: value of the p parameter in the GOSPA loss. Integer greater than or equal to 1.
- nTestSamples: number of samples to use during testing.
- sch_flag: flag indicating if a scheduler should be used. Defaul: False.
- imp_flag: flag indicating if impairment learning or dictionary learning are
being considered. True means that impairment learning would be considered.
Defaul: True.
- initial_pos: initial position of the first (reference) antenna element.
- device: device where to perform simulations ('cpu' or 'cuda'). Default 'cuda'.
- rx_feature: It can be either a matrix of steering vectors or the assumed positions. This depends on the dyn_flag. 
If dyn_flag=True, rx_feature represents the assumed antenna positions and the matrix of steering vectors is
dynamically computed for each batch sample in each iteration. Otherwise, rx_feature is directly such matrix.
- dyn_flag: binary flag that indicates if rx_feature corresponds to the assumed antenna positions and we 
compute a matrix of steering vectors from those positions or if rx_feature direcly corresponds to a matrix of
steering vectors for the receiver side.
'''