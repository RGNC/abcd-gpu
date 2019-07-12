readme.txt: This file explains the code and the structure of the algorithm.

Version 1.0 beta


I. Motivation

This simulator is able to simulate Population Dynamics P systems (based on multienvironment functional probabilistic P systems with extended active membranes). It implements the Direct distribution based on Consistent Blocks Algorithm (DCBA), in both sequential and parallel (OpenMP and CUDA) way. The phases of the algorithm are:

1. Selection of rules to be executed.
1.1 Phase 1 (Distribution)
1.2 Phase 2 (Maximality)
1.3 Phase 3 (Probability)
2. Execution of rules selected.


II. Installation and requirements:

2.1. Requirements and dependencies:

- A Linux based distribution (only Ubuntu has been tested; maybe a Windows system with cygwin could work, but has not been tested).
- A CUDA installation, from version 5.5, including: 
     * NVIDIA toolkit, its associated libraries, and the nvcc compiler.
     * Configure LD_LIBRARY_PATH to contain the CUDA lib(64) folder, e.g. in .bashrc, add "export LD_LIBRARY_PATH=/usr/local/cuda/lib"
     * CUDA SDK examples.
- The GNU g++ compiler
- The GNU Scientific Library (GSL). E.g. apt-get install gsl-bin libgsl0-dev
- Electric Fence, in order to debug the simulator.
- The counterslib library, available with PMCGPU, inside the folder 8_pmcgpu 

2.2. Installation:

a. Install all the required packages from 2.1.
b. Inside the folder of CUDA SDK samples, create a new folder named 8_pmcgpu.
c. Extract the contents of files abcd-gpu-1.0b.tar.gz and counterslib.tar.gz inside this new folder.
d. Go to folder abcd-gpu-1.0b, and type "make". You should see the binary file inside the folder.


III. Usage:

Type ./abcdgpu -h to list the different options. In this version, input files of binary format and randomly generated PDP systems are supported.

* A random PDP system sequential simulation: ./abcdgpu -X 2
* A random PDP system (OpenMP) parallel simulation using 4 CPU cores: export OMP_NUM_THREADS=4; ./abcdgpu -X 2
* A random PDP system (CUDA) parallel simulation: ./abcdgpu -X 2 -I 1
* A random PDP system profiling execution (CPU vs GPU): ./abcdgpu -X 2 -I 1 -M 1
* A profiling execution (CPU vs GPU) for a random PDP system with 100000 rule blocks, 1000 objects in the alphabet, q degree of 4, maximum of 5 rules per block, and maximum of 3 objects in LHS membranes using 100 simulations, 20 environments (m degree) and 1 step: ./abcd -I 1 -M 1 -R -b 100000 -o 1000 -q 4 -r 5 -l 3 -s 100 -e 20 -t 1
* A profiling execution (CPU vs GPU) for the Bearded Vulture model (in plingua folder, previously generated from the .pli file), using 1000 simulations, 42 steps, 3 steps per cycle, verbosity 1: ./abcdgpu -f plingua/bv_model_bwmc12.bin -s 1000 -t 42 -I 1 -M 1 -v 1 -c 3
* A simulation of the Bearded Vulture model on the GPU using 100 simulations, 42 steps, 3 steps per cycle, verbosity 1, and the output of a csv (which will be named after the input file plus the extension of .csv): ./abcdgpu -f plingua/bv_bwmc12.bin -I 1 -s 100 -t 42 -v 1 -c 3 -O 0



IV. Source:

The objective of each file is the following:

- main.cpp: Contains the main function, generating the PDP system and calling the chosen simulator.

- simulator.h: Virtual class of simulators.

- pdp_psystem.h: Virtual class for PDP systems data structures.

- pdp_psystem_source.h: Virtual class for different sources of PDP systems.

- pdp_system_sab.cpp/.h: Class of P system structures based on struct of arrays.

- pdp_system_redix.cpp/.h: Class of P system structures based on struct of arrays, reduced by storing indexes.

- pdp_psystem_source_random.cpp/.h: A random generator of PDP systems.

- pdp_psystem_source_binary.cpp/.h: An input module to read binary files.

- pdp_psystem_output.cpp/.h: Class of P system output modules.

- pdp_psystem_output_csv.cpp/.h: An output module for csv files.

- pdp_psystem_output_binary.cpp/.h: An experimental output module for binary files.

- simulator_seq_table.cpp/.h: An out-of-dated simulator, fully implementing the DCBA (i.e. using a full table).

- simulator_seq_dir.cpp/.h: A sequential simulator implementing the virtual table for DCBA.

- simulator_omp_dir.cpp/.h: A parallel simulator based on the simulator_seq_dir simulator, for multicore processors using OpenMP. Several strategies are implemented.

- simulator_omp_redir.cpp/.h: A parallel simulator improving the data structures and algorithm in simulator_omp_dir simulator. It is used for the comparisons with GPU simulator.

- simulator_gpu_dir.cu/.h: A parallel simulator based on simulator_omp_redir structures, for manycore GPUs using CUDA. All the kernels are implemented within the same file .cu.

- curng_binomial.h: A cuRAND based library for binomial random variate generation on the GPU.

- binbit.cpp/.h: esentials to write/read binary files (adapted from PCUDA project).



V. Acknowledgements

The authors acknowledge the support of the projects TIN2012-37434 of the "Ministerio de Economía y Competitividad" of Spain, co-financed by FEDER funds, and "proyecto de excelencia con Investigador de Reconocida Valía" P08-TIC04200 of the "Junta de Andalucía". They also acknowledge the CUDA Research Center program, granted by NVIDIA to the University of Seville in 2014 and 2015, and their donation of a Tesla K40 GPU. M.A. Martínez-del-Amor also acknowledges the support of the 3rd Postdoctoral phase of the PIF program of project P08-TIC04200.

/*$Id: readme.txt 2015-07-19 20:38:45 mdelamor $*/
