################################################################################
#
#    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
#    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
#                                        Computing on the GPU)   
# 
#    Copyright (c) 2015  Research Group on Natural Computing, Universidad de Sevilla
#    					 Dpto. Ciencias de la Computación e Inteligencia Artificial
#    					 Escuela Técnica Superior de Ingeniería Informática,
#    					 Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)
#
#    Author: Miguel Ángel Martínez-del-Amor
#    
#    This file is part of ABCD-GPU.
#  
#    ABCD-GPU is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    ABCD-GPU is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with ABCD-GPU.  If not, see <http://www.gnu.org/licenses/>. */
################################################################################

################################################################################
#
# Makefile to compile the PMCGPU project ABCD-GPU:
#
# Requirements of your system:
# - A Linux based distribution (only Ubuntu has been tested)
# - A CUDA installation, from version 5.5, including: 
#      * NVIDIA toolkit, its associated libraries, and the nvcc compiler.
#      * Configure LD_LIBRARY_PATH to contain the CUDA lib(64) folder, e.g. 
#          in .bashrc, add "export LD_LIBRARY_PATH=/usr/local/cuda/lib"
#      * CUDA SDK examples. Create the folder 8_pmcgpu, and extract abcd-gpu
#		   inside the folder
# - The GNU g++ compiler
# - The GNU Scientific Library (GSL). E.g. apt-get install gsl-bin libgsl0-dev
# - Electric Fence, in order to debug the simulator.
# - The counterslib library, available with PMCGPU, inside the folder 8_pmcgpu
#
################################################################################

################################################################################
#
# This version is based on CUDA SDK examples (from 5.5 to 6.5), and uses the
# libraries already defined there. First download and install CUDA and the
# SDK, then copy the source code inside the examples, e.g. inside the folder
# "NVIDIA_CUDA-X.Y_Samples/8_pmcgpu/abcdgpu". You must agree with the NVIDIA
# CUDA EULA in order to install and use it.
#
################################################################################

# Provide the absolute path to your CUDA installation
CUDA_PATH:= /usr/local/cuda

# GNU g++ compiler is required
GCC := g++
	
# NVIDIA nvcc compiler is required
NVCC := nvcc -ccbin $(GCC)

################################################################################
# Additional configuration

INCLUDES	:=  -I../counterslib/inc -I../../common/inc\
		    -I$(CUDA_PATH)/include -I.

LIB		:= -L../../common/lib -L../counterslib/lib \
                   -ltimestat -lgsl -lgslcblas 

# You need to have GSL installed, if you are on linux, install the packages
# gsl-bin libgsl0-dev

CXXFLAGS	:= -fopenmp

#Flags for nvcc compiler
#NVCCFLAGS	+= -g -G -L../../common/counterslib/lib -I../../common/counterslib/inc -lefence
#NVCCFLAGS	+= -gencode=arch=compute_12,code=\"sm_12,compute_12\"
#NVCCFLAGS	+= -arch sm_12
# Just let use the value by default
#NVCCFLAGS	:= -m${OS_SIZE}

################################################################################
# Debug mode
#dbg=1

ifeq ($(dbg),1)
      NVCCFLAGS += -g -G -D DEBUG
      CXXFLAGS += -g -O0 -D DEBUG
      LIB += -lefence
      TARGET := debug
else ifeq ($(dbg),2)
      NVCCFLAGS += -g -G -D DEBUG -D BIN_DEBUG
      CXXFLAGS += -g -O0 -D DEBUG -D BIN_DEBUG
      LIB += -lefence
      TARGET := debug
else
      CXXFLAGS += -O2
      TARGET := release
endif

#NVCCFLAGS += $(addprefix --compiler-options ,$(CXXFLAGS)) 
#NVCXXFLAGS ?= --compiler-options $(CXXFLAGS)
NVCCFLAGS += -Xcompiler $(CXXFLAGS)

ALL_CFLAGS := $(INCLUDES) $(CXXFLAGS)
ALL_LDFLAGS := $(LIB)


################################################################################
# CUDA code generation flags
# TODO: provide here the most used architectures. Here I place only for the GPUs
# I have. Perhaps do a configure.ac, to detect the GPU and tune the makefile?.

# Add here your generation of GPU
GENCODE_SM12    := -gencode arch=compute_12,code=sm_12
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM35    := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM12) $(GENCODE_SM13) $(GENCODE_SM20)\
	$(GENCODE_SM35)


################################################################################
# Source and object files
# Add source files here
OBJDIR	:= objs

# CUDA source files (compiled with nvcc)
CUFILES	:= simulator_gpu_dir.cu

# C++ source files (compiled with g++)
CCFILES	:= binbit.cpp pdp_psystem_source_random.cpp\
	pdp_psystem_source_binary.cpp pdp_psystem_output_binary.cpp\
	pdp_psystem_output_csv.cpp\
	pdp_psystem_sab.cpp pdp_psystem_redix.cpp\
	simulator_seq_table.cpp simulator_seq_dir.cpp\
	simulator_omp_dir.cpp simulator_gpu_dir.cpp simulator_omp_redir.cpp\
	main.cpp

CCHFILES := binbit.h pdp_psystem.h pdp_psystem.h\
	pdp_psystem_output.h pdp_psystem_output_binary.h\
	pdp_psystem_output_csv.h\
	pdp_psystem_redix.h pdp_psystem_sab.h\
	pdp_psystem_source.h pdp_psystem_source_binary.h\
	pdp_psystem_source_random.h simulator.h\
	simulator_gpu_omp_dir.h simulator_omp_redir.h simulator_seq_dir.h\
	simulator_seq_table.h

CUHFILES := curng_binomial.h simulator_gpu_dir.h

OBJS := $(patsubst %.cpp,$(OBJDIR)/%.cpp.o, $(CCFILES))
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu.o, $(CUFILES))
#OBJECTS :=  $(CCFILES:.cpp=.o) 
#$(CUFILES:.cu=.o)


# Silent mode
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif


################################################################################
# Rules and targets
#TARGETDIR := ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$
#TARGETDIR := ../../bin/$(OSLOWER)/$(TARGET)
TARGETDIR := ../../bin/linux/$(TARGET)

all: makedirectoryobjs abcdgpu
	
makedirectoryobjs:
	$(VERBOSE)mkdir -p $(OBJDIR)
	
abcdgpu: $(OBJS)
	$(VERBOSE)$(NVCC) $(GENCODE_FLAGS) $(ALL_LDFLAGS) $(NVCCFLAGS) -o $@ $+
	$(VERBOSE)mkdir -p $(TARGETDIR)
	$(VERBOSE)cp $@ $(TARGETDIR)
	
clean:
	$(VERBOSE)rm -f $(OBJDIR)/*.o *.o

$(OBJDIR)/%.cpp.o: %.cpp %.h
	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu %.h $(CUHFILES)
	$(VERBOSE)$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# TODO: makes the following to work!
	
#$(OBJDIR)/main.o: main.cpp\
	pdp_psystem.h pdp_psystem_source_random.h\
	pdp_psystem_source_binary.h simulator.h simulator_seq_table.h\
	simulator_seq_dir.h simulator_omp_dir.h simulator_gpu_dir.h\
	simulator_omp_redir.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

#$(OBJDIR)/simulator_gpu_dir.o: simulator_gpu_dir.cu simulator_gpu_dir.cpp\
	$(OBJDIR)/pdp_psystem_redix.o\
	simulator_gpu_dir.h simulator.h
#	$(VERBOSE)$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c simulator_gpu_dir.cu
#	$(VERBOSE)$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c simulator_gpu_dir.cpp
	
#$(OBJDIR)/simulator_omp_redir.o: simulator_omp_redir.cpp\
	$(OBJDIR)/pdp_psystem_redix.o\
	simulator_omp_redir.h simulator.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<

#$(OBJDIR)/simulator_omp_dir.o: simulator_omp_dir.cpp\
	$(OBJDIR)/pdp_psystem_sab.o\
	simulator_omp_dir.h simulator.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<

#$(OBJDIR)/simulator_seq_dir.o: simulator_seq_dir.cpp\
	$(OBJDIR)/pdp_psystem_sab.o\
	simulator_seq_dir.h simulator.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<
	
#$(OBJDIR)/simulator_seq_table.o: simulator_seq_table.cpp\
	$(OBJDIR)/pdp_psystem_sab.o\
	simulator_seq_table.h simulator.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<

#$(OBJDIR)/pdp_psystem_redix.o: pdp_psystem_redix.cpp\
	pdp_psystem_redix.h pdp_psystem.h pdp_psystem_source.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<
	
#$(OBJDIR)/pdp_psystem_sab.o: pdp_psystem_sab.cpp\
	pdp_psystem_sab.h pdp_psystem.h pdp_psystem_source.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<
	
#$(OBJDIR)/pdp_psystem_source_binary.o: pdp_psystem_source_binary.cpp\
	$(OBJDIR)/binbit.o\
	pdp_psystem_source_binary.h pdp_psystem_source.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<
	
#$(OBJDIR)/pdp_psystem_source_random.o: pdp_psystem_source_random.cpp\
	pdp_psystem_source_random.h pdp_psystem_source.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<
	
#$(OBJDIR)/binbit.o: binbit.cpp binbit.h
#	$(VERBOSE)$(GCC) $(ALL_CFLAGS) -o $@ -c $<

#.cu.o:
#	$(GCC) $(ALL_NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@
	
#
#.cu.o:
#	$(GCC) $(ALL_NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@
#
#clean: 
#	rm -f *.o
#	rm -rf ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/$(EXECUTABLE)
