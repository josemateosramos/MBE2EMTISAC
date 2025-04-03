# Model-Based End-to-End Learning for Multi-Target Integrated Sensing and Communication Under Hardware Impairments

## Getting Started
This code is based on Pytorch 1.12.1 and CUDA 11.3.1, and may not work with other versions. For more information about how to install these versions, check the Pytorch documentation.

The simulation parameters to train and test different scenarios are located in the ```simulation_parameters.py``` file within the lib/ directory. The methods/ directory contains the scripts to train and test all methods: (i) baseline with known impairments, (ii) baseline with unknown impairments, (iii) dictionary learning, (iv) impairmnent learning, and (v) model-based greedy calibration. Note that some files are only used for inference and require that the corresponding training script has been run in advance.

## Additional information 
If you decide to use the source code for your research, please make sure to cite our paper:

J. Miguel Mateos-Ramos, C. Häger, M. Furkan Keskin, L. Le Magoarou and H. Wymeersch, "Model-Based End-to-End Learning for Multi-Target Integrated Sensing and Communication Under Hardware Impairments," in IEEE Transactions on Wireless Communications, vol. 24, no. 3, pp. 2574-2589, March 2025.
