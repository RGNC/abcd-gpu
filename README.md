# Parallel simulation of Population Dynamics P systems (ABCD-GPU) #

----------
## 1. Simulator based on OpenMP ##

After defining a new simulation algorithm for PDP systems, called DCBA, the challenge was to efficiently implement it somehow. The first step was an implementation within pLinguaCore. However, the simulation was slow. Then, the solution was to develop a version in C++ version, in a stand-alone simulator. This projects was called ABCD, but now is ABCD-GPU. 

The implementation in C++ reproduces the stages of DCBA, selection and execution, plus the three micro-stages for selection: phase 1 (distribution), phase 2 (maximality) and phase 3 (probability). The main problem arises when simulating large PDP system models. The static table required in selection phase 1 is too large, and also it is sparse. Therefore, ABCD-GPU simulators save on memory by avoiding the creation of a static table. It is carried out by translating the operations over the table to operations directly to the rule blocks information.

This C++ implementation is parallelized in three ways: 1) simulations, 2) environments and  3) a hybrid approach. All of them are implemented using the parallel standard library for multicore platforms, OpenMP. The experiments were ran on two multi-core processors: the Intel i5 Nehalem and i7 Sandy Bridge. We achieve runtime gains of up to 2.5x by using all the cores of a single socket 4-core Intel i7.

Experiments indicate the simulations are memory bound and the portion of the code we parallelized consumes over 98% of the runtime in serial. From this initial work we conclude that parallelizing by simulations or hybrid techniques yields the largest speedups. Also, using hardware that has more memory bandwidth is an easy way for scientists to improve the speed of our simulator. It can also be concluded that performance tuning to decrease data movement is important for P-system simulators.

The simulator is flexible and scalable enough to run a wide range of PDP systems, but the performance is still low.

----------
## 2. Simulator based on CUDA ##

The major expansion of the ABCD-GPU project came with the creation of a CUDA simulator. The simulation algorithm is the DCBA, together with the optimizations made in the previous simulators in ABCD-GPU. Moreover, the memory utilization was improved (saving up to 27% of memory) for both the GPU-based version and the previous OpenMP-based version. 

The CUDA design for the GPU part of the simulator is as follows: environments and simulations are distributed through thread blocks, and rule blocks among threads. Phases 1, 3 and 4 were efficiently executed on the GPU, however Phase 2 was poorly accelerated, since it is inherently sequential. Furthermore, phase 3 requires the creation of random binomial variates. For this purpose, a new CUDA library was developed, called cuRNG_BINOMIAL. It uses the normal approximation for large parameters values, and the BINV algorithm for low ones. The cuRAND library was utilized to generate the required uniform and normal random numbers.

Version 0.8 of ABCD-GPU was benchmarked with a set of randomly generated PDP systems (without biological meaning), achieving speedups of up to 7x for large sizes (running 50 simulations, 20 environments and more than 20000 rule blocks) on NVIDIA Tesla C1060 GPU over the multi-core version. 

The new innovation in version 1.0 is an input module which supports files with a binary format, aiming to compress the information provided in the models. The purpose of using a restricted, binary format is for efficiency in its communication. Moreover, this input module has allowed to run tests with real ecosystem models. Specifically, we have chosen the ecosystem of the Bearded Vulture in the Catalan Pyrenees presented in BWMC 2010 (in DCBA paper), in order to experimentally validate the simulator. Furthermore, performance analysis has been made using a Tesla C1060 and a Tesla K40 GPUs, achieving up to 4.9x and 18.1x respectivelly with the Bearded Vultured model (1 environment, 1000 simulations, 42 time steps). These experiments also allowed to experimentally validate the GPU and CPU simulators.

The results show that with this real ecosystem model, phase 1 of DCBA is the bottleneck, since there is no competing blocks, what effectively disable phase 2 in the simulation. Moreover, we have shown that next generation GPUs, such as K40, achieves better performance, given their higher memory bandwidth and their L2 caches. For example, phases 1 and 4, which are the most data intensive in DCBA, are 10 times faster in K40 than in its predecessor, demonstrating our theory that P system simulations are memory bandwidth bounded.

----------
## 3. Installation and how to use ##

### 3.1. Requirements and dependencies ###

  - A Linux based distribution (only Ubuntu has been tested; maybe a Windows system with cygwin could work, but has not been tested).
  - A CUDA installation, from version 5.5, including: 
     * NVIDIA toolkit, its associated libraries, and the nvcc compiler.
     * Configure LD_LIBRARY_PATH to contain the CUDA lib(64) folder, e.g. in .bashrc, add "export LD_LIBRARY_PATH=/usr/local/cuda/lib"
     * CUDA SDK examples.
  - The GNU g++ compiler
  - The GNU Scientific Library (GSL). E.g. apt-get install gsl-bin libgsl0-dev
  - Electric Fence, in order to debug the simulator.
  - The counterslib library, available with PMCGPU, inside the folder 8_pmcgpu 

### 3.2. Installation ###

  a. Install all the required packages from 2.1.
  b. Inside the folder of CUDA SDK samples, create a new folder named 8_pmcgpu.
  c. Extract the contents of files abcd-gpu-1.0b.tar.gz and counterslib.tar.gz inside this new folder.
  d. Go to folder abcd-gpu-1.0b, and type "make". You should see the binary file inside the folder.

### 3.3 Usage ###

Type ./abcdgpu -h to list the different options. In this version, input files of binary format and randomly generated PDP systems are supported.

  * A random PDP system sequential simulation: ./abcdgpu -X 2
  * A random PDP system (OpenMP) parallel simulation using 4 CPU cores: export OMP_NUM_THREADS=4; ./abcdgpu -X 2
  * A random PDP system (CUDA) parallel simulation: ./abcdgpu -X 2 -I 1
  * A random PDP system profiling execution (CPU vs GPU): ./abcdgpu -X 2 -I 1 -M 1
  * A profiling execution (CPU vs GPU) for a random PDP system with 100000 rule blocks, 1000 objects in the alphabet, q degree of 4, maximum of 5 rules per block, and maximum of 3 objects in LHS membranes using 100 simulations, 20 environments (m degree) and 1 step: ./abcd -I 1 -M 1 -R -b 100000 -o 1000 -q 4 -r 5 -l 3 -s 100 -e 20 -t 1
  * A profiling execution (CPU vs GPU) for the Bearded Vulture model (in plingua folder, previously generated from the .pli file), using 1000 simulations, 42 steps, 3 steps per cycle, verbosity 1: ./abcdgpu -f plingua/bv_model_bwmc12.bin -s 1000 -t 42 -I 1 -M 1 -v 1 -c 3
  * A simulation of the Bearded Vulture model on the GPU using 100 simulations, 42 steps, 3 steps per cycle, verbosity 1, and the output of a csv (which will be named after the input file plus the extension of .csv): ./abcdgpu -f plingua/bv_bwmc12.bin -I 1 -s 100 -t 42 -v 1 -c 3 -O 0

----------
## 4. Publications ##

### 4.1. Journals ###

* Miguel Ángel Martinez-del-Amor, Luis Felipe Macías-Ramos, Luis Valencia-Cabrera, Mario J. Pérez--Jiménez. **Parallel simulation of Population Dynamics P systems: Updates and roadmap**. *Natural Computing*, 15, 4 (2016), 565-573.
* Miguel A. Martínez-del-Amor, Ignacio Pérez-Hurtado, Adolfo Gastalver-Rubio, Anne C. Elster, Mario J. Pérez-Jiménez. **Population Dynamics P Systems on CUDA**, *Lecture Notes in Bioinformatics*, 7605 (2012), 247-266.

### 4.2. Conference contributions ###

* Miguel A. Martínez-del-Amor, Ian Karlin, Rune E. Jensen, Mario J. Pérez-Jiménez, Anne C. Elster. **Parallel Simulation of Probabilistic P Systems on Multicore Platforms**, *Tenth Brainstorming Week on Membrane Computing*, Vol. II (2012), 17-26.

### 4.3 Ph.D. Thesis ###

* Miguel Á. Martínez-del-Amor. [Accelerating Membrane Systems Simulators using High Performance Computing with GPU.](http://www.cs.us.es/~mdelamor/research.html#thesis) May 2013, University of Seville. Advised by Mario J. Pérez-Jiménez and Ignacio Pérez-Hurtado.

----------
## 5. Downloads ##

[Required Counterslib library](http://sourceforge.net/projects/pmcgpu/files/counterslib)

Read the howto.pdf (extract from Miguel A. Martínez-del-Amor's thesis) for futher information about the simulators. It is in the [root folder of files of PMCGPU](http://sourceforge.net/projects/pmcgpu/files).

----------
## 6. How to acknowledge ##

If you intend to create a branch of ABCD-GPU, or use its produced results, please consider citing the following publications:

  * M.A. Martínez-del-Amor, L.F. Macías-Ramos, L. Valencia-Cabrera, M.J. Pérez-Jiménez. Parallel simulation of Population Dynamics P systems: updates and roadmap. Natural Computing, 15, 4 (2016), 565-573.*

  * Miguel A. Martínez-del-Amor, Ignacio Pérez-Hurtado, Adolfo Gastalver-Rubio, Anne C. Elster, Mario J. Pérez-Jiménez. Population Dynamics P Systems on CUDA, Lecture Notes in Bioinformatics, 7605 (2012), 247-266.*

  * Miguel A. Martínez-del-Amor, Ian Karlin, Rune E. Jensen, Mario J. Pérez-Jiménez, Anne C. Elster. Parallel Simulation of Probabilistic P Systems on Multicore Platforms, Tenth Brainstorming Week on Membrane Computing, Vol. II (2012), 17-26.*

----------
## 7. Funding ##

This work has been supported by the "Proyecto de Excelencia con Investigador de Reconocida Valía" of the "Junta de Andalucía" under grant P08-TIC04200, and by the projects TIN2017-89842-P of the "Ministerio de Economía, Industria y Competitividad",  TIN2012-37434 of the "Ministerio de Economía y Competitividad" and TIN2009-13192 of the "Ministerio de Educación y Ciencia" of Spain, both co-financed by FEDER funds. NVIDIA's supports and donations to the RGNC and to HPCLAB at NTNU are also gratefully acknowledged.
