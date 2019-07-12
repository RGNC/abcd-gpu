/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2015  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Author: Miguel Ángel Martínez-del-Amor
    
    This file is part of ABCD-GPU.
  
    ABCD-GPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ABCD-GPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ABCD-GPU.  If not, see <http://www.gnu.org/licenses/>. */

/*
 * This GPU simulator for PDP systems was introduced and analysed in the paper:
 * M.A. Martínez-del-Amor, I. Pérez-Hurtado, A. Gastalver-Rubio, A.C. Elster
 * M.J. Pérez-Jiménez. Population Dynamics P systems on CUDA. Proceedings of
 * the 10th Conference on Computational Methods in Systems Biology (CMSB2012),
 * London, 2012. Lecture Notes in Bioinformatics, 7605 (2012), 247-266.
 *
 * And extended for the paper:
 * M.A. Martínez-del-Amor, L.F. Macías-Ramos, L. Valencia-Cabrera, M.J. Pérez-
 * Jiménez. Parallel simulation of Population Dynamics P systems: updates and
 * roadmap. Natural Computing. Accepted.
 */

#include "simulator_gpu_dir.h"

#include "curng_binomial.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <timestat.h>
#include <cstdlib>
using namespace std;

#define CU_THREADS 256
#define CU_LOG_THREADS 8

/************************************************************/
/* The following sets how the arrays are indexed on the CPU */
/************************************************************/
#define AD_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))
#define NB_IDX (sim*options->num_environments*besize+env*besize+block)
#define NR_P_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+env*rpsize+r)
#define NR_E_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+options->num_environments*rpsize+(r-rpsize))
#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))

/************************************************************/
/* The following sets how the arrays are indexed on the GPU */
/************************************************************/
#define D_AD_IDX(o,m) (sim*options.num_environments*esize+env*esize+(m)*msize+(o))
#define D_NB_IDX(b) (sim*options.num_environments*besize+env*besize+(b))
#define D_NR_P_IDX(r) (sim*(options.num_environments*rpsize+(resize-rpsize))+env*rpsize+(r))
#define D_NR_E_IDX(r) (sim*(options.num_environments*rpsize+(resize-rpsize))+options.num_environments*rpsize+((r)-rpsize))
#define D_CH_IDX(m) (sim*(options.num_environments*options.num_membranes)+env*options.num_membranes+(m))
#define D_MU_IDX(o,m) (sim*options.num_environments*esize+env*esize+(m)*msize+(o))

/**************************************/
/* How to construct collision objects */
/**************************************/
#define EMPTY UINT_MAX
#define IS_EMPTY(o) ((o)==UINT_MAX)
#define OBJECT(obj,membr,mult) ((((mult)&0x7FF)<<20)|(((obj)+(membr)*msize)&0xFFFFF))
#define GET_OBJECT(o) (o&0xFFFFF)
#define OBJECT_COLLISION(init,b,o) ((0x80000000|((init)&0xFFF00000)) | (((b)&0x3FF)<<10) | ((o)&0x3FF))
#define COLLISION_GET_TID(o) (((o)>>10)&0x3FF)
#define COLLISION_GET_OBJ(o) ((o)&0x3FF)
#define IS_COLLISION(o) ((o)>>31)
#define COLLISION(o1,o2) (!(((o1)>>31)^((o2)>>31)) && (((o1)&0xFFFFF) == ((o2)&0xFFFFF)))
#define SET_CONF_MULT(obj,mult) (((obj)&0xFFF00000)|((mult)&0xFFFFF))
#define GET_CONF_MULT(o) (o&0xFFFFF)
#define GET_MULT(o) ((o>>20)&0x7FF)

/***************************************************************************/

/* Constructor of the class */

Simulator_gpu_dir::Simulator_gpu_dir(PDP_Psystem_REDIX* PDPps,int mode,bool accurate, PDP_Psystem_output* PDPout) {
	this->structures=PDPps->structures;
	this->options=PDPps->options;
	runcomp=(mode>=1);
	this->mode=mode;
	this->accurate=accurate; // use n/d mode for normalization by default
	error=false;

	init();

	// The real out (binary, csv...)
	this->PDPout=PDPout;

	/* Configure the standard output class (debugging purposes only) */
	/* Let keep it here, after init(). The initial configuration is */
	/* backed up there, and d_structures is initialized */
	pdp_out = new PDP_Psystem_redix_out_std_gpuwrapper(PDPps,this->d_structures,runcomp,&counters);
}


/*************/
/* MAIN LOOP */
/*************/

bool Simulator_gpu_dir::run() {

	return step(options->time);
}

bool Simulator_gpu_dir::step(int k){

	pdp_out->print_simulator_process("[2] STARTING THE SIMULATION: Using virtual table (direct) algorithm, with CUDA");

	/* Repeat for each Parallel Simulation Block (PSB) */
	for (uint psb=0; psb < options->num_simulations; psb+=sim_parallel) {
		
		if (sim_parallel > options->num_simulations - psb)
			options->num_parallel_simulations=options->num_simulations - psb;
		
		reset(psb);

        pdp_out->print_sim_range(psb,psb+options->num_parallel_simulations);


		/* MAIN LOOP */
		for (uint i=0; i<k; i++) {
                        
            pdp_out->print_step(i);

			if (selection())
				return false;

			if (execution())
				return false;

            pdp_out->print_configuration();

            if ((i+1)%options->cycles==0) {
            	retrieve(psb);
            	for (uint simu=psb; (simu <psb+sim_parallel) && (simu < options->num_simulations); simu++)
            		PDPout->write_configuration(structures->configuration.multiset,structures->configuration.membrane,simu,i+1,structures->stringids.id_objects);
            }
		}
	}

    /* Output profiling information */
	pdp_out->print_profiling_table();
	
	return true;
}


/***************************************************************************/
/***************************************/
/* Auxiliary functions Implementations */
/***************************************/

/* Safely add and mult uint numbers, returns true if overflow */
/*
bool safe_u_add(uint& op1, uint op2){
	uint c_test=op1+op2;
	if (c_test-op2 != op1) 
		return true;
	else op1=c_test;
	
	return false;
}

bool safe_u_mul(uint& op1, uint op2) {
	uint c_test=op1*op2;
	if (c_test/op2 != op1) 
		return true;
	else op1=c_test;	
	
	return false;
}*/


//TODO: Make this member to return a boolean value, to check errors
bool Simulator_gpu_dir::init() {

	/* Set auxiliary sizes info */
	esize=options->num_objects*options->num_membranes;
	msize=options->num_objects;
	bpsize=options->num_rule_blocks;
	besize=options->num_blocks_env+options->num_rule_blocks;
	rpsize=structures->pi_rule_size;
	resize=structures->pi_rule_size+structures->env_rule_size;
	asize=((besize>>ABV_LOG_WORD_SIZE)+1);

	/* Initialize GPU */
	char * def_dev = getenv("DEFAULT_DEVICE");
	unsigned int dev;
	if (def_dev!=NULL)
		cudaSetDevice(dev= atoi(def_dev));
	else
		cudaSetDevice(dev = gpuGetMaxGflopsDeviceId());
	
	checkCudaErrors(cudaGetDeviceProperties(&dev_property, dev));

	/* Calculating the amount of memory, and simulations to handle */
	unsigned int dep_mem;
	
	// GPU only
	unsigned int max_memory_gpu = dev_property.totalGlobalMem;

	// Temporally like this until auto-compression process
	//cutilCondition(options->mem < max_memory_gpu);
	dep_mem=options->num_membranes*options->num_environments*sizeof(CHARGE); //membrane
	dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(MULTIPLICITY); //multiset
	dep_mem+=(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sizeof(MULTIPLICITY);//nb
	dep_mem+=((structures->pi_rule_size)*options->num_environments+structures->env_rule_size)*sizeof(MULTIPLICITY);//nr
	dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(float)*2; //addition
	dep_mem+=asize*options->num_environments*sizeof(ABV_T); // ABV activations
	dep_mem+=(1+options->num_membranes*options->num_environments)*sizeof(uint); //data error
	dep_mem+=curng_sizeof_state(CU_THREADS*options->num_environments); //random data

	// Add new data structures depending on the number of simulations

	sim_parallel=gsl_min(options->num_simulations,(((unsigned int) max_memory_gpu*0.8)-options->mem)/dep_mem);
	options->num_parallel_simulations=sim_parallel;
	options->mem+=dep_mem*sim_parallel;

	/* Printing information */

	if (sim_parallel==0) {
		cout << "Error: no enough memory to run even a single simulation!" << endl;
		return false; // TODO: Catch this!
	}

	if (options->verbose>=1)
		cout << "Selected GPU device:" << endl <<
				"=> Device: " << dev << " (" << dev_property.name << "), Multiprocessors=" << dev_property.multiProcessorCount <<
				", Total GPU memory=" << dev_property.totalGlobalMem << endl;
	if (options->verbose>1)
		cout << "Information about required memory and parallel simulations" << endl <<
				"=> Static memory: " << options->mem << endl <<
				"=> Algorithm memory for one simulation: " << dep_mem << endl <<
                "=> Parallel simulations: " << sim_parallel << endl <<
                "=> Total memory: " << options->mem + dep_mem*sim_parallel << endl;

	/************************************/
	/* Initialization of GPU structures */
	/************************************/
	
	/* Create initial configuration */
	ini_cfg = structures->configuration;

	structures->configuration.membrane_size=options->num_membranes*options->num_environments*options->num_simulations;
	structures->configuration.membrane = new CHARGE[structures->configuration.membrane_size];

	structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments*options->num_simulations;
	structures->configuration.multiset = new MULTIPLICITY[structures->configuration.multiset_size];

	/* Init configurations */
	for (int sim=0; sim<options->num_simulations; sim++) {
		for (int env=0; env<options->num_environments; env++) {
			for (int m=0;m<options->num_membranes; m++) {
				structures->configuration.membrane[CH_IDX(m)]=ini_cfg.membrane[env*options->num_membranes+m];
			}
			for (int o=0;o<options->num_membranes*options->num_objects;o++) {
				structures->configuration.multiset[MU_IDX(o,0)]=ini_cfg.multiset[env*esize+o];
			}
		}
	}	
	
	/* Initialize the new data structure for activation bit vector */
	initialize_abv();
	
	/* Initialize new data structure for data error */
	data_error_size = 1+2*options->num_membranes*options->num_environments*sim_parallel;
	data_error = new uint[data_error_size];
	for (unsigned int i=0; i<data_error_size; i++)
		data_error[i]=0;
	
	/* Initialize aux data on CPU only if neccessary */
	structures->nb_size=(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sim_parallel;
		
	/* Initialize Nb only inf CPU is going to be executed */
	if (runcomp) {
		structures->nb = new MULTIPLICITY [structures->nb_size];	
	} else
		structures->nb = NULL;

	/* Initialize Nr only if the CPU is used, or for verbosity stuff */
	structures->nr_size= ((structures->pi_rule_size)*options->num_environments+structures->env_rule_size)*sim_parallel;
		
	if (options->verbose>1 || runcomp) {	
		structures->nr = new MULTIPLICITY [structures->nr_size]; 
	} else
		structures->nr = NULL;

	/* For printting purposes */
	unsigned int d_nb_size=(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sim_parallel;
	
	if (options->verbose>1) {
		d_nb = new unsigned int[d_nb_size];
	} else
		d_nb = NULL;
	
	/* Initialize auxiliary structures for normalization */
	addition_size=options->num_objects*options->num_membranes*options->num_environments*sim_parallel;
	bool finished = false;
	
	while (!finished) {
		/* Use n/d notation for row additions */
		if (accurate) {
			/* Check overflows */
			finished=true;
			bool overflow=false;
			
			ini_denominator = new uint[esize];
			ini_numerator = new uint[esize];
			denominator = ini_denominator;
			numerator = new uint [addition_size];
			addition = NULL;
			
			for (int i=0;i<esize;i++) {
				ini_denominator[i]=1;
				ini_numerator[i]=0;
			}
			
			for (uint block=0; block<besize; block++) {
				for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
					unsigned int obj=structures->lhs.object[o];
					unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
					unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
					
					uint a=ini_denominator[membr*options->num_objects+obj];
					uint b=mult;
					
					int multiple1 = a % b;
					int multiple2=1;
					if (multiple1!=0)
						multiple2 = b % a;
					
					/* If a is multiple of b */
					if (multiple1==0) {
						overflow=safe_u_add(ini_numerator[membr*options->num_objects+obj],a/b);
					}
					/* If b is multiple of a */
					else if (multiple2==0) {
						overflow=safe_u_mul(ini_numerator[membr*options->num_objects+obj],b/a);
						overflow=overflow||safe_u_add(ini_numerator[membr*options->num_objects+obj],1);
						ini_denominator[membr*options->num_objects+obj]=b;
					}
					/* If they are no multiple */
					else {
						overflow=safe_u_mul(ini_numerator[membr*options->num_objects+obj],b);
						overflow=overflow||safe_u_add(ini_numerator[membr*options->num_objects+obj],a);
						overflow=overflow||safe_u_mul(ini_denominator[membr*options->num_objects+obj],b);
					}
					if (overflow) break;
				}
				if (overflow) {
					if (options->verbose>0) {
						cout << "Warning: overflow detected in initialization of row sums (accurate mode n/d), switching to float" << endl;
					}
					accurate=false;
					finished=false;
					delete [] ini_denominator;
					delete [] ini_numerator;
					delete [] numerator;
					ini_numerator=numerator=ini_denominator=denominator=NULL;
					break;
				}
			}
		}
		/* Use float notation for row additions */
		else {
			if (runcomp)
				addition = new float[addition_size];
			else
				addition = NULL;
			finished=true;
		}
	}
	
	/* Select a phase2 kernel */
	size_t sh_mem=((CU_THREADS >> ABV_LOG_WORD_SIZE) + 2*CU_THREADS + options->max_lhs*CU_THREADS)*sizeof(uint);
	if (sh_mem > dev_property.sharedMemPerBlock)
		mode=2; // Use generic kernel
			
	
	/************************************/
	/* Initialization of GPU structures */
	/************************************/
	d_structures = new PDP_Psystem_REDIX::struct_structures;
	d_structures->ruleblock_size = structures->ruleblock_size;
	d_structures->env_rule_size = structures->env_rule_size;
	d_structures->lhs_size = structures->lhs_size;
	d_structures->rhs_size = structures->rhs_size;
	d_structures->pi_rule_size = structures->pi_rule_size;
	d_structures->probability_size = structures->probability_size;

	d_structures->configuration.membrane_size = options->num_membranes*options->num_environments*sim_parallel;
	d_structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments*sim_parallel;
	
	d_structures->nr_size = //(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sim_parallel;
			(structures->pi_rule_size+structures->env_rule_size)*options->num_environments*sim_parallel;
	d_structures->nb_size = (options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sim_parallel;

	/* Print new information */
	if (options->verbose>1) {
		cout << " => Memory used by the virtual table algorithm on the GPU:" << endl;

		if (accurate) {
			cout << "Denominator: " << esize*sizeof(uint) << " (" << esize*sizeof(uint)/1024 << "KB)" << endl;
			cout << "Numerator: " << addition_size*sizeof(uint) << " (" << addition_size*sizeof(uint)/1024 << "KB)" << endl;
		} else
			cout << "Addition: " << addition_size*sizeof(float) << " (" << addition_size*sizeof(float)/1024 << "KB)" << endl;
		
		cout << "Nb: " << d_structures->nb_size*sizeof(unsigned int) << " (" << d_structures->nb_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "Nr: " << d_structures->nr_size*sizeof(unsigned int) << " (" << d_structures->nr_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "ABV: " << abv_size*sizeof(ABV_T) << " (" << abv_size*sizeof(ABV_T)/1024 << "KB)" << endl;
		int rngsize=curng_sizeof_state(CU_THREADS*options->num_environments*sim_parallel);
		cout << "RNG: " << rngsize << " (" << rngsize/1024 << "KB)" << endl;
		cout << "Errors: " << data_error_size*sizeof(uint) << " (" << data_error_size*sizeof(uint)/1024 << "KB)" << endl;
		cout << "Membrane charges: " << structures->configuration.membrane_size*sizeof(char) << " (" << structures->configuration.membrane_size*sizeof(char)/1024 << "KB)" << endl;
		cout << "Multisets: " << structures->configuration.multiset_size*sizeof(unsigned int) << " (" << structures->configuration.multiset_size*sizeof(unsigned int)/1024 << "KB)" << endl;

		int count=0;
		float div=1;
		char unit[6]={' ','K','M','G','T','P'};
		while ((options->mem/div)>1023 && count<3){
			div*=1024;
			count++;
		}
		cout << "TOTAL: " << options->mem << " (" << options->mem/div << " " << unit[count] << "B)" << endl << endl;
	}

	/* Allocation */
	// Allocate Ruleblock
	checkCudaErrors(cudaMalloc((void**)&(d_structures->ruleblock.lhs_idx), (d_structures->ruleblock_size+1)*sizeof(LHS_IDX)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->ruleblock.rule_idx), (d_structures->ruleblock_size+1)*sizeof(RULE_IDX)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->ruleblock.membrane), d_structures->ruleblock_size*sizeof(MEMBRANE)));

	// Allocate LHS
	checkCudaErrors(cudaMalloc((void**)&(d_structures->lhs.object), d_structures->lhs_size*sizeof(OBJECT)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->lhs.mmultiplicity), d_structures->lhs_size*sizeof(MULTIPLICITY)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->lhs.imultiplicity), d_structures->lhs_size*sizeof(INV_MULTIPLICITY)));

	// Allocate RHS
	checkCudaErrors(cudaMalloc((void**)&(d_structures->rhs.object), d_structures->rhs_size*sizeof(OBJECT)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->rhs.mmultiplicity), d_structures->rhs_size*sizeof(MULTIPLICITY)));

	// Allocate Rule
	checkCudaErrors(cudaMalloc((void**)&(d_structures->rule.rhs_idx), (d_structures->pi_rule_size+d_structures->env_rule_size+1)*sizeof(RHS_IDX)));

	// Allocate Probability
	checkCudaErrors(cudaMalloc((void**)&(d_structures->probability), d_structures->probability_size*sizeof(PROBABILITY)));

	// Allocate Nr
	checkCudaErrors(cudaMalloc((void**)&(d_structures->nr), d_structures->nr_size*sizeof(MULTIPLICITY)));

	// Allocate Nb
	checkCudaErrors(cudaMalloc((void**)&(d_structures->nb), d_structures->nb_size*sizeof(MULTIPLICITY)));

	// Allocate Configuration
	checkCudaErrors(cudaMalloc((void**)&(d_structures->configuration.multiset), d_structures->configuration.multiset_size*sizeof(MULTIPLICITY)));
	checkCudaErrors(cudaMalloc((void**)&(d_structures->configuration.membrane), d_structures->configuration.membrane_size*sizeof(CHARGE)));

	// Allocate Additions
	if (!accurate)
		checkCudaErrors(cudaMalloc((void**)&d_addition,addition_size*sizeof(float)));
	else {
		checkCudaErrors(cudaMalloc((void**)&d_denominator,esize*sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_ini_numerator,esize*sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_numerator,addition_size*sizeof(uint)));
	}

	// Allocate ABV
	checkCudaErrors(cudaMalloc((void**)&d_abv,abv_size*sizeof(ABV_T)));
	checkCudaErrors(cudaMemset(d_abv,0xFF,abv_size*sizeof(ABV_T)));
	
	// Allocate Errors
	checkCudaErrors(cudaMalloc((void**)&d_data_error,data_error_size*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_data_error,0,data_error_size*sizeof(uint)));
	
	// Allocate RNG states
	curng_binomial_init(dim3(options->num_environments,options->num_parallel_simulations),CU_THREADS);
	
	/* Copies */
	// Copy Ruleblock
	checkCudaErrors(cudaMemcpy(d_structures->ruleblock.lhs_idx, structures->ruleblock.lhs_idx, (d_structures->ruleblock_size+1)*sizeof(LHS_IDX), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->ruleblock.rule_idx, structures->ruleblock.rule_idx, (d_structures->ruleblock_size+1)*sizeof(RULE_IDX), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->ruleblock.membrane, structures->ruleblock.membrane, d_structures->ruleblock_size*sizeof(MEMBRANE), cudaMemcpyHostToDevice));

	// Copy LHS
	checkCudaErrors(cudaMemcpy(d_structures->lhs.object, structures->lhs.object, d_structures->lhs_size*sizeof(OBJECT), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->lhs.mmultiplicity, structures->lhs.mmultiplicity, d_structures->lhs_size*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->lhs.imultiplicity, structures->lhs.imultiplicity, d_structures->lhs_size*sizeof(INV_MULTIPLICITY), cudaMemcpyHostToDevice));

	// Copy RHS
	checkCudaErrors(cudaMemcpy(d_structures->rhs.object, structures->rhs.object, d_structures->rhs_size*sizeof(OBJECT), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->rhs.mmultiplicity, structures->rhs.mmultiplicity, d_structures->rhs_size*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice));

	// Copy Rule
	checkCudaErrors(cudaMemcpy(d_structures->rule.rhs_idx, structures->rule.rhs_idx, (d_structures->pi_rule_size+d_structures->env_rule_size+1)*sizeof(RHS_IDX), cudaMemcpyHostToDevice));

	// Copy Probability
	checkCudaErrors(cudaMemcpy(d_structures->probability, structures->probability, d_structures->probability_size*sizeof(PROBABILITY), cudaMemcpyHostToDevice));

	// Copy Additions
	if (accurate) {
		checkCudaErrors(cudaMemcpy(d_denominator, ini_denominator, esize*sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_ini_numerator, ini_numerator, esize*sizeof(uint), cudaMemcpyHostToDevice));
	}	

	// Create a timer
	sdkCreateTimer(&counters.timer);
	
	return true;
}

void Simulator_gpu_dir::del() {
	if (addition) delete [] addition;
	if (denominator) delete [] denominator;
	if (ini_numerator) delete [] ini_numerator;
	if (numerator) delete [] numerator;
	
	delete [] d_nb;
	PDP_Psystem_REDIX::Configuration aux;
	aux=structures->configuration;
	structures->configuration=ini_cfg;
	delete []aux.membrane;
	delete []aux.multiset;
	if (structures->nb) delete []structures->nb;
	if (structures->nr) delete []structures->nr;
	if (abv) delete []abv;
	if (data_error) delete []data_error;
	
	// Deallocate Ruleblocks
	checkCudaErrors(cudaFree(d_structures->ruleblock.lhs_idx));
	checkCudaErrors(cudaFree(d_structures->ruleblock.rule_idx));
	checkCudaErrors(cudaFree(d_structures->ruleblock.membrane));

	// Deallocate LHS
	checkCudaErrors(cudaFree(d_structures->lhs.object));
	checkCudaErrors(cudaFree(d_structures->lhs.mmultiplicity));
	checkCudaErrors(cudaFree(d_structures->lhs.imultiplicity));

	// Deallocate RHS
	checkCudaErrors(cudaFree(d_structures->rhs.object));
	checkCudaErrors(cudaFree(d_structures->rhs.mmultiplicity));

	// Deallocate Rule
	checkCudaErrors(cudaFree(d_structures->rule.rhs_idx));

	// Deallocate Probability
	checkCudaErrors(cudaFree(d_structures->probability));

	// Deallocate Nr
	checkCudaErrors(cudaFree(d_structures->nr));

	// Deallocate Nb
	checkCudaErrors(cudaFree(d_structures->nb));

	// Deallocate Configuration
	checkCudaErrors(cudaFree(d_structures->configuration.multiset));
	checkCudaErrors(cudaFree(d_structures->configuration.membrane));

	// Deallocate Additions
	if (!accurate) checkCudaErrors(cudaFree(d_addition));
	else {
		checkCudaErrors(cudaFree(d_denominator));
		checkCudaErrors(cudaFree(d_ini_numerator));
		checkCudaErrors(cudaFree(d_numerator));
	}

	// Deallocate ABV
	checkCudaErrors(cudaFree(d_abv));
	
	// Deallocate Errors
	checkCudaErrors(cudaFree(d_data_error));
	
	// Deallocate RNG states
	curng_binomial_free();	

	sdkDeleteTimer(&counters.timer);
	
	cudaThreadExit();
}

void Simulator_gpu_dir::reset(int sim_ini) {
	checkCudaErrors(cudaMemcpy(d_structures->configuration.membrane, structures->configuration.membrane+sim_ini*options->num_environments*options->num_membranes, options->num_parallel_simulations*options->num_environments*options->num_membranes*sizeof(CHARGE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structures->configuration.multiset, structures->configuration.multiset+sim_ini*options->num_environments*esize, options->num_parallel_simulations*options->num_environments*esize*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice));
}

void Simulator_gpu_dir::retrieve(int sim_ini) {

	checkCudaErrors(cudaMemcpy(structures->configuration.membrane+sim_ini*options->num_environments*options->num_membranes, d_structures->configuration.membrane, options->num_parallel_simulations*options->num_environments*options->num_membranes*sizeof(CHARGE), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(structures->configuration.multiset+sim_ini*options->num_environments*esize, d_structures->configuration.multiset, options->num_parallel_simulations*options->num_environments*esize*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
}

/***************************************************************************/
/***************/
/* MAIN PHASES */
/***************/

int Simulator_gpu_dir::selection(){

	/* PHASE 1: DISTRIBUTION */
	if (!selection_phase1())
		return 1;

	/* PHASE 2: MAXIMALITY */
	if (!selection_phase2())
		return 2;

	/* PHASE 3: PROBABILITY */
	if (!selection_phase3())
		return 3;
	
	return 0;
}


/***************************************************************************/
/*********************/
/* Selection methods */
/*********************/


/*********************************************/
/*********************/
/* Selection Phase 1 */
/*********************/

/*******************************************/
/* Using activation bit vectors on the GPU */
/*******************************************/
__device__ inline bool d_is_active (uint block, uint * abv) {
	return (abv[(block>>ABV_LOG_WORD_SIZE)]
	               >> ((~block)&ABV_DESPL_MASK))
	        & 0x1;
}

__device__ inline void d_deactivate(uint block, uint * abv) {
	atomicAnd(&(abv[(block>>ABV_LOG_WORD_SIZE)]), ~(0x1<<((~block)&ABV_DESPL_MASK)));
}



/*****************************************/
/* Step 1 (filters) of phase1 on the GPU */
/*****************************************/
__global__ void kernel_phase1_filters(
			PDP_Psystem_REDIX::Ruleblock ruleblock,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Lhs lhs,
			PDP_Psystem_REDIX::NR nb,
			struct _options options,
			uint * d_abv,
			uint * d_data_error) {
// TODO: load all next data as symbols
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
	extern __shared__ uint sData[];
	uint *s_abv=sData;
	uint *m_c_charges=sData+(blockDim.x>>ABV_LOG_WORD_SIZE);
	uint *m_c_conflicts=m_c_charges+options.num_membranes;
	__shared__ bool c_conflict;//=false;

	bool active=false;

	// TODO: do not assume that number of threads is always > num membranes
	if (threadIdx.x < options.num_membranes) {
		c_conflict=false;
		m_c_charges[threadIdx.x]=4;
		m_c_conflicts[threadIdx.x]=UINT_MAX;
	}

	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
		block=bchunk*blockDim.x+threadIdx.x;

		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=ABV_INIT_WORD;
		}

		__syncthreads();

		if (block < besize) {
			/*** Filter 1 ***/
			uint membr=ruleblock.membrane[block];

			// Case for rule blocks in Pi
			if (IS_MEMBRANE(membr)) {
				uint am=GET_MEMBRANE(membr);
				char ch=GET_ALPHA(membr);
				// only active those with charge alpha in LHS
				active=(configuration.membrane[sim*options.num_environments*options.num_membranes+env*options.num_membranes+am] == ch);
			}
			// Case for rule blocks for communication, active only if in corresponding environment
			else if (IS_ENVIRONMENT(membr)) {
				active=(GET_ENVIRONMENT(membr)==env);
			}

			/** Filter 2 **/
			if (active) {
				// Using new registers avoid memory accesses on the for loop
				uint o_init=ruleblock.lhs_idx[block];
				uint o_end=ruleblock.lhs_idx[block+1];
				for (int o=o_init; o < o_end; o++) {
					uint obj=lhs.object[o];
					uint membr=lhs.mmultiplicity[o];
					uint mult=GET_MULTIPLICITY(membr);
					membr=GET_MEMBR(membr);

					// Check if we have enough objects to apply the block
					if (configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]<mult) {
						active=false;
						//break;
					}
				}
			}

			if (!active)
				d_deactivate(threadIdx.x,s_abv);
			else if (active && IS_MEMBRANE(membr)) {
				uint am=GET_MEMBRANE(membr);
				char chb=GET_BETA(membr);

				char setch= (char) atomicCAS(&m_c_charges[am],4,(uint)chb);
				if (setch!=4 && setch!= chb)
				/*if (m_c_charges[am]==4) {
					m_c_charges[am]=chb;//GET_BETA(membr);
					//printf("S=%d,B=%d,T=%d: (am=%d,beta=%d) -> (charge=%d)\n",sim,block,threadIdx.x,am,chb,m_c_charges[am]);
				}
				else if (m_c_charges[am]!= chb) /*GET_BETA(membr)*/ {
					//printf("!!S=%d,B=%d,T=%d: (am=%d,beta=%d) -> (charge=%d)\n",sim,block,threadIdx.x,am,chb,m_c_charges[am]);
					m_c_conflicts[am] = block;
					c_conflict = true;
				}
			}

			nb[D_NB_IDX(block)] = 0;
		}

		__syncthreads();

		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x]=s_abv[threadIdx.x];
		}
	}
	
	if (c_conflict && (threadIdx.x < options.num_membranes)) {
		d_data_error[1+sim*options.num_environments*options.num_membranes+env*options.num_membranes+threadIdx.x]=m_c_charges[threadIdx.x];
		d_data_error[1+gridDim.y*options.num_environments*options.num_membranes+
			sim*options.num_environments*options.num_membranes+env*options.num_membranes+threadIdx.x]=m_c_conflicts[threadIdx.x];
		if (threadIdx.x==0)// && d_data_error[0]!=CONSISTENCY_ERROR)
			d_data_error[0]=CONSISTENCY_ERROR;
	}
	//__syncthreads();
}

/******************************************************************************************************/
/* Atomic Addition operation for floats                                                               *
 * Solution taken from http://forums.nvidia.com/index.php?showtopic=158039&st=0&p=991561&#entry991561 */
/******************************************************************************************************/
__device__ inline void atomicAddf(float* address, float value){
	#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
		atomicAdd(address,value);
	#elif __CUDA_ARCH__ >= 110
		// float-atomic-add
		float old = value;
		while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
	#endif
}


/************************************************************/
/* Step 2 (normalization and minimums) of phase1 on the GPU */
/************************************************************/
__global__ void kernel_phase1_normalization(
		PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nr,
		struct _options options,
		float * d_addition,
		uint * d_abv,
		uint obj_chunks) {

	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
	extern __shared__ uint s_abv[];

	// Initialize addition vector
	for (int ochunk=0; ochunk < obj_chunks; ochunk++) {
		uint obj=ochunk*blockDim.x+threadIdx.x;
		if (obj>=esize) break;
		d_addition[sim*options.num_environments*esize+env*esize+obj]=1.0f;
	}
	__syncthreads();

	/* Normalization - step 1 *
	 *  calculate the sum of objects in lhs */
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
		block=bchunk*blockDim.x+threadIdx.x;

		if (block >= besize) break;
		
		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		__syncthreads();
		
		// If the block is activated
		if (d_is_active(threadIdx.x,s_abv)) {
			uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				//uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				// TODO: Check if using imultiplicity is more efficient
				float inv=lhs.imultiplicity[o];
				atomicAddf(d_addition+sim*options.num_environments*esize+env*esize+membr*msize+obj,inv);
				// TODO: Replace this for using d / n format
			}
		}
		__syncthreads();
	}
	__syncthreads();

	/* Normalization - step 2 *
	 * Column minimum calculation */
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
                uint min=0;
		
		block=bchunk*blockDim.x+threadIdx.x;
		if (block >= besize) break;

		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		__syncthreads();
		
		// If the block is activated
		if (d_is_active(threadIdx.x,s_abv)) {
			min=UINT_MAX;
                        uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);
				
				//uint value = configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]/(mult*mult);
				//value =	value / (d_addition[sim*options.num_environments*esize+env*esize+membr*msize+obj]-1.0f);
				uint value = configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]/(mult*mult*(d_addition[sim*options.num_environments*esize+env*esize+membr*msize+obj]-1.0f));
				
				min=(value < min) ? value : min;
				if(min==0) break;
			}
		}
		__syncthreads();
		nr[sim*options.num_environments*besize+env*besize+block]=min;
	}
	__syncthreads();
}

//TODO: Implemented, but not used. I keep it just for interest
__device__ inline void atomicMul(uint* address, uint value){
	#if __CUDA_ARCH__ >= 110
		// atomic-mul
	if (value!=1) {
		uint old = value;
		while ((old = atomicExch(address, atomicExch(address, 1)*old))!=1);
	}
	#endif
}

__global__ void kernel_phase1_normalization_acu (
		PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nr,
		struct _options options,
		uint * d_denominator,
		uint * d_numerator,
		uint * d_ini_numerator,
		uint * d_abv,
		uint obj_chunks) {

	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
	extern __shared__ uint s_abv[];

	// Initialize addition vector
	for (int ochunk=0; ochunk < obj_chunks; ochunk++) {
		uint obj=ochunk*blockDim.x+threadIdx.x;
		if (obj>=esize) break;		
		d_numerator[D_AD_IDX(obj,0)]=d_ini_numerator[obj];
	}
	__syncthreads();

	/* Normalization - step 2 *
	 *  calculate the sum of objects in lhs */
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
		block=bchunk*blockDim.x+threadIdx.x;
		
		if ((block < besize) && threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		__syncthreads();
		
		// If the block is activated
		if ((block < besize) && !d_is_active(threadIdx.x,s_abv)) {
			uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				atomicSub(d_numerator+D_AD_IDX(obj,membr),d_denominator[membr*options.num_objects+obj]/mult);
			}
		}
		__syncthreads();
	}

	/* Normalization - step 2 *
	 * Column minimum calculation */
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
                uint min=0;
		
		block=bchunk*blockDim.x+threadIdx.x;
		//if (block >= besize) break;

		if ((block < besize) && threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		__syncthreads();
		
		// If the block is active
		if ((block < besize) && d_is_active(threadIdx.x,s_abv)) {
			min=UINT_MAX;
                        uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);
				
				uint value = (configuration.multiset[D_MU_IDX(obj,membr)] * d_denominator[membr*options.num_objects+obj]) / (mult*mult*d_numerator[D_AD_IDX(obj,membr)]);
				min=(value < min) ? value : min;
				if(min==0) break;
			}
		}
		__syncthreads();
		
		if (block < besize)
			nr[D_NB_IDX(block)]=min;
		//sim*options.num_environments*besize+env*besize+block]=min;
	}
	//__syncthreads();
}

/*****************************************************/
/* Step 3 (update and filter 2) of phase1 on the GPU */
/*****************************************************/
__global__ void kernel_phase1_update(
		PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		struct _options options,
		uint * d_abv,
		uint * d_data_error) {
	
	extern __shared__ uint s_abv[];
	__shared__ bool block_sel;
	bool update_error=false;
	uint block_upd_error=0;
	
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
	
	/* Deleting LHS *
	 * Adding block applications */
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
	
		block=bchunk*blockDim.x+threadIdx.x;
		if (block >= besize) break;
		
		uint bapp=nr[D_NB_IDX(block)];
		
		if (bapp>0) {
            if (!block_sel) block_sel=true;
			
			/* Consume LHS */
                        uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);
                        
				/* Delete block application and check errors */
				if (atomicSub(configuration.multiset+sim*options.num_environments*esize+env*esize+membr*msize+obj,bapp*mult)
					< bapp*mult)
					if (!update_error) update_error=true;
					block_upd_error = 1+block;
					/* Pre-filter: only filter last rules consuming objects */
					//bapp*mult+mult) ;
					//d_deactivate(threadIdx.x,s_abv);
			}

			/* Add applications to block */
			nb[D_NB_IDX(block)]+=bapp;
		}
	}
	
	/** Filter 2 **/
	
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
	
		block=bchunk*blockDim.x+threadIdx.x;
		if (block >= besize) break;

		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		__syncthreads();
		
		if (d_is_active(threadIdx.x,s_abv)) {
			// Using new registers avoid memory accesses on the for loop
			uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				// Check if we have enough objects to apply the block
				if (configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]<mult) {
					d_deactivate(threadIdx.x,s_abv);
					break;
				}
			}
		}		
		
		__syncthreads();
		
		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)) {
			d_abv[sim*options.num_environments*asize+env*asize+((bchunk*blockDim.x)>>ABV_LOG_WORD_SIZE)+threadIdx.x]=s_abv[threadIdx.x];
		}
	}
	
	if (threadIdx.x==0 && update_error) {
		d_data_error[1+sim*options.num_environments*options.num_membranes+env*options.num_membranes]=block_upd_error;
		if (d_data_error[0]!=UPDATING_CONFIGURATION_ERROR)
			d_data_error[0]=UPDATING_CONFIGURATION_ERROR;		
	}
	
	__syncthreads();
}


/************************************************/
/* Implementation of Phase 1 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase1() {
    
    pdp_out->print_dcba_phase(1);

    pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 1");
	
	/* Create and start timers */
	if (runcomp) {
		counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0;
		counters.timek1cpu = counters.timek2cpu = counters.timek3cpu = 10;
	}

	
	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;

	dim3 dimGrid (cu_blocksx, cu_blocksy);
	dim3 dimBlock (cu_threads);
	size_t sh_mem=((cu_threads>>ABV_LOG_WORD_SIZE) + 2*options->num_membranes)*sizeof(uint);
	uint obj_chunks=(esize + cu_threads -1)/cu_threads;

	/* Apply kernel for filters */
	if (runcomp) {
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for filters");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}

	kernel_phase1_filters <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nb, *options,
			d_abv, d_data_error);
	
	getLastCudaError("kernel for phase 1 (filters) launch failure");	
	//cutilDeviceSynchronize();	
	cudaDeviceSynchronize(); // Compatible for CUDA < 4.0
	if (runcomp) {
		sdkStopTimer(&counters.timer);
		counters.timek1gpu=sdkGetTimerValue(&counters.timer);
		pdp_out->print_profiling_dcba_microphase_result(counters.timek1gpu);
	}
	
	checkCudaErrors(cudaMemcpy(data_error, d_data_error, data_error_size*sizeof(uint), cudaMemcpyDeviceToHost));
	
	/* Checking mutual consistency */
	pdp_out->print_profiling_dcba_microphase_name("Checking mutual consistency");
	
	if (data_error[0]==CONSISTENCY_ERROR) {
		pdp_out->print_profiling_dcba_microphase_result(false);

		checkCudaErrors(cudaMemcpy(this->abv, this->d_abv, this->abv_size*sizeof(ABV_T), cudaMemcpyDeviceToHost));

		cout << "Found inconsistent blocks:" << endl;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++)
			for (unsigned int env=0; env < options->num_environments; env++)
				for (unsigned int membr=0; membr<options->num_membranes; membr++) {
					uint charge=data_error[1+sim*options->num_environments*options->num_membranes+env*options->num_membranes+membr];
					uint block=data_error[1+options->num_parallel_simulations*options->num_environments*options->num_membranes+
											sim*options->num_environments*options->num_membranes+env*options->num_membranes+membr];
					if (block!=UINT_MAX) {
						cout << "For sim " << sim << ", env " << env << ", membr " << membr <<
							" conflicts with charge " << charge << " for block " <<  block << endl;

						for (int blk=0; blk<options->num_rule_blocks; blk++) {
							uint am=GET_MEMBRANE(structures->ruleblock.membrane[blk]);
							char ch=GET_BETA(structures->ruleblock.membrane[blk]);
							if (is_active(blk,env,sim) && am==membr && ch==charge)
								cout << "   Possibly conflicted with " << blk << endl;
						}
					}
				}
			
		checkCudaErrors(cudaMemset(d_data_error,0,data_error_size*sizeof(uint)));

		return false;
	}
	else
		pdp_out->print_profiling_dcba_microphase_result(true);

		
	
	for (int a=0; a<options->accuracy; a++) {
		/* Apply kernel for normalization */
		
		sh_mem=(cu_threads>>ABV_LOG_WORD_SIZE)*sizeof(uint);
		
		if (runcomp) {
			pdp_out->print_profiling_dcba_microphase_name("Launching kernel for normalization");
			sdkResetTimer(&counters.timer);
			sdkStartTimer(&counters.timer);
		}

		if (! accurate)
		kernel_phase1_normalization <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nr,
			*options,d_addition,d_abv,obj_chunks);
		else
		kernel_phase1_normalization_acu <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nr,
			*options,d_denominator,d_numerator,d_ini_numerator,d_abv,obj_chunks);
	
	
		getLastCudaError("kernel for phase 1 (normalization) launch failure");
		//cutilDeviceSynchronize();	
		cudaDeviceSynchronize(); // Compatible for CUDA < 4.0
			
		if (runcomp) {
			sdkStopTimer(&counters.timer);
			counters.timek2gpu+=sdkGetTimerValue(&counters.timer);
			pdp_out->print_profiling_dcba_microphase_result(counters.timek2gpu);
		}
	
		/* Apply kernel for update and filter 2 */
	
		if (runcomp) {
			pdp_out->print_profiling_dcba_microphase_name("Launching kernel for updating");
			sdkResetTimer(&counters.timer);
			sdkStartTimer(&counters.timer);
		}

		kernel_phase1_update <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nb,
			d_structures->nr, *options, d_abv, d_data_error);
	
		getLastCudaError("kernel for phase 1 (update) launch failure");
		//cutilDeviceSynchronize();	
		cudaDeviceSynchronize(); // Compatible for CUDA < 4.0
		
		if (runcomp) {
			sdkStopTimer(&counters.timer);
			counters.timek3gpu+=sdkGetTimerValue(&counters.timer);
			pdp_out->print_profiling_dcba_microphase_result(counters.timek3gpu);
		}
		
		/* Checking updating errors */
		pdp_out->print_profiling_dcba_microphase_name("Checking updating errors");
		
		if (data_error[0]==UPDATING_CONFIGURATION_ERROR) {
			pdp_out->print_profiling_dcba_microphase_result(false);

			cout << "Stopped. Found errors:" << endl;
			for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++)
				for (unsigned int env=0; env < options->num_environments; env++)
					for (unsigned int membr=0; membr<options->num_membranes; membr++)
						cout << "For sim " << sim << ", env " << env << ", membr " << membr <<
						" error for block " << data_error[1+sim*options->num_environments*options->num_membranes+env*options->num_membranes+membr]-1 << endl;
			
			checkCudaErrors(cudaMemset(d_data_error,0,data_error_size*sizeof(uint)));
			return false;
		}
		else pdp_out->print_profiling_dcba_microphase_result(true);
	}
	
	pdp_out->print_block_selection();

	pdp_out->print_temporal_configuration();


	/**************************************/
	/* PROFILING AND CHECK-OUT PROCEDURES */
	/**************************************/

//	if (options->verbose > 1) {
//		/* RETRIEVING DATA */
//		checkCudaErrors(cudaMemcpy(d_nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
//		print_block_applications(d_nb);
//
//		d_cfg.multiset = new MULTIPLICITY[structures->configuration.multiset_size];
//
//		checkCudaErrors(cudaMemcpy(d_cfg.multiset, d_structures->configuration.multiset, d_structures->configuration.multiset_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
//
//		print_configuration(d_cfg);
//
//		delete [] d_cfg.multiset;
//	}
	
	/*******************************/
	/* REPRODUCING CODE ON THE CPU */
	/*******************************/
	if (runcomp) {
		gold_selection_phase1_acu();
	
		/* Retrieving times */
		counters.timesp1gpu= counters.timek1gpu+counters.timek2gpu+counters.timek3gpu;
	}
	
	if (runcomp && pdp_out->will_print_dcba_phase()) {
		/***************************************/
		/***** TEMPORAL TESTING PROCEDURE ******/
		/***************************************/

		/* Temporal checking addition, only for debugging */
		/*float * debug_addition = new float[addition_size];
		checkCudaErrors(cudaMemcpy(debug_addition, d_addition, addition_size*sizeof(float), cudaMemcpyDeviceToHost));

		cout << "Checking addition vectors: " ;//<< endl;
		//cout.precision(15);
		double sum=0.0;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (unsigned int env=0; env < options->num_environments; env++) {
				//cout << endl << "Addition vector on GPU (env " << env << "): ";
				for (unsigned int m=0; m < options->num_membranes; m++) {
					for (int o=0;o<options->num_objects;o++) {
						sum+=fabs((debug_addition[AD_IDX(o,m)]-1.0f-addition[AD_IDX(o,m)]));
							//(debug_addition[AD_IDX(o,m)]-1.0f-addition[AD_IDX(o,m)]);
					//if (fabs(addition[AD_IDX(o,m)]-(debug_addition[AD_IDX(o,m)]-1.0f))>0.5) {
					//	cout << "For sim " << sim << ", env " << env << ", membr " << m << ", obj " << o << ":" ;
					//	cout << "CPU=" << addition[AD_IDX(o,m)] << ", GPU=" << debug_addition[AD_IDX(o,m)]-1.0f <<endl;
					//}
					}
				}
			}
		}
		double deviation=sum;//(double)addition_size;

		delete [] debug_addition;
		//cout.precision(15);
		cout << "Deviation: " << deviation << endl;
		*/
		
		/* Temporal checking of numerators and denominators, only debugging*/
		/*
		uint *d_d = new uint[esize];
		uint *d_n = new uint[addition_size];
		
		checkCudaErrors(cudaMemcpy(d_nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(d_d, d_denominator, esize*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(d_n, d_numerator, addition_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));

		
		cout.precision(4);
		for (unsigned int sim=0; sim < options->num_simulations; sim++) {
			cout << "For sim " << sim << ":" << endl;
			for (unsigned int env=0; env < options->num_environments; env++) {
				cout << "For env " << env << ":" << endl;
				for (unsigned int block=0; block <besize; block++) {
					//cout << " b_" << block << "{" << GET_MEMBRANE(structures->ruleblock.membrane[block]) << "," << GET_ALPHA(structures->ruleblock.membrane[block]) << "," << GET_BETA(structures->ruleblock.membrane[block]) << "}*" << structures->nb[NB_IDX];
					if (structures->nb[NB_IDX]!=d_nb[NB_IDX]) {
						cout << " b_" << block << "{am=" << GET_MEMBRANE(structures->ruleblock.membrane[block]) << ", a=" << GET_ALPHA(structures->ruleblock.membrane[block]) << ", b=" << GET_BETA(structures->ruleblock.membrane[block]) << "}: ";
						cout << "CPU=" << structures->nb[NB_IDX] << "vs GPU=" << d_nb[NB_IDX] << ", LHS:" << endl;
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							uint val=(ini_cfg.multiset[MU_IDX(obj,membr)]*denominator[options->num_objects*membr+obj]) / (mult*mult*numerator[AD_IDX(obj,membr)]);
							cout << "\t[o" << obj << "]_" << membr << "*" << mult << "=(mult=" << ini_cfg.multiset[MU_IDX(obj,membr)] << ", n/d(CPU)=" << numerator[AD_IDX(obj,membr)] << "/" << denominator[membr*options->num_objects+obj] << ", n/d(GPU)=" << d_n[AD_IDX(obj,membr)] << "/" << d_d[options->num_objects*membr+obj] <<", val=" << val << "), ";
							for (unsigned int block2=0; block2 <besize; block2++) {
								if (block==block2) continue;
								for (unsigned int o2=structures->ruleblock.lhs_idx[block2]; o2<structures->ruleblock.lhs_idx[block2+1]; o2++) {
									unsigned int obj2=structures->lhs.object[o2];
									unsigned int mult2=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o2]);
									unsigned int membr2=GET_MEMBR(structures->lhs.mmultiplicity[o2]);

									if (obj==obj2 && membr==membr2)
										cout << "cb_" << block2 << ",o" << obj2 << "*" << mult2 <<", ";
								}
							}
							cout << endl;
						}
						cout << "}" << endl;
					}
				}
				cout << endl;
			}
		}

		delete [] d_d;
		delete [] d_n;*/
	
		/* Checking ABV */
		ABV_T *debug_abv= new ABV_T[abv_size];
		checkCudaErrors(cudaMemcpy(debug_abv, d_abv, abv_size*sizeof(ABV_T), cudaMemcpyDeviceToHost));

		int count_errors=0;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (unsigned int env=0; env < options->num_environments; env++) {
				for (unsigned int block=0; block<besize; block++) {
					bool is_gpu_active_abv=(debug_abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
    			 		               >> ((~block)&ABV_DESPL_MASK))& 0x1;
					bool is_cpu_active_abv = is_active(block,env,sim);

					if (is_gpu_active_abv!=is_cpu_active_abv) {
					//cout << "ABV for GPU-CPU fails: " << block << "-e" << env << "-s" << sim << ": GPU="
					//<< is_gpu_active_abv << " VS CPU=" << is_cpu_active_abv  << endl;
						count_errors++;
					}
				}
			}
		}

		cout << "Checking ABV: there are " << count_errors << " diferences." << endl;

		delete [] debug_abv;
		
		//d_nb=new uint[d_structures->nb_size];
		/* CALCULATING DIFFERENCES */
		checkCudaErrors(cudaMemcpy(d_nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
		int diff=0;
		int s_diff=0;
		int g_diff=0;
		int sig_diff=0;

		for (unsigned int i=0; i < structures->nb_size; i++) {
			if (structures->nb[i]<d_nb[i]) {
				diff++; s_diff++;
			}
			else if (structures->nb[i]>d_nb[i]) {
				diff++; g_diff++;
			}
			int w_diff = structures->nb[i]-d_nb[i];

			if (w_diff < -1 || w_diff > 1)
				sig_diff++;
		}
		delete []d_nb;

		cout << "Checking NB: there are " << diff << " differences, " << s_diff << " smaller, " << g_diff << " bigger, and " << sig_diff << " significative differences." << endl;
	
		/* PRINTING TIMES */
		cout << endl << "Time for kernel_phase1_filters: GPU=" << counters.timek1gpu << " ms, CPU=" << counters.timek1cpu << " ms, speedup=" << counters.timek1cpu/counters.timek1gpu << "x" << endl;
		cout << endl << "Time for kernel_phase1_normalization: GPU=" << counters.timek2gpu << " ms, CPU=" << counters.timek2cpu << " ms, speedup=" << counters.timek2cpu/counters.timek2gpu << "x" << endl;
		cout << endl << "Time for kernel_phase1_update: GPU=" << counters.timek3gpu << " ms, CPU=" << counters.timek3cpu << " ms, speedup=" << counters.timek3cpu/counters.timek3gpu << "x" << endl;
		cout << endl << "Time for phase 1: GPU=" << counters.timesp1gpu << " ms, CPU=" << counters.timesp1cpu << " ms, speedup=" << counters.timesp1cpu/counters.timesp1gpu << "x" << endl;
	}
	
	pdp_out->print_end_profiling_dcba_phase();

	/** END OF PROCEDURE **/
	return true;
}



/*********************************************/
/*********************/
/* Selection Phase 2 */
/*********************/

/*****************************************/
/* Kernel for Phase 2 version 1: generic */
/*****************************************/
__global__ void kernel_phase2_generic(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		struct _options options,
		uint * d_abv) {
	
	extern __shared__ uint sData[];
	__shared__ uint next_b,max_b;
	
	uint bdim = blockDim.x - 1;
	uint * s_abv = sData;
	uint * s_blocks = sData+(bdim >> ABV_LOG_WORD_SIZE);
	uint * s_blocks_update = s_blocks + bdim;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + bdim -1)>>CU_LOG_THREADS;
	
	/* One extra iteration, the calculation of sblocks and minimums are pipelined */
	for (int bchunk=0; bchunk < block_chunks+1; bchunk++) {

		block=bchunk*bdim+threadIdx.x;

		/* Only first 256 threads will calculate activations */
		if ((threadIdx.x<bdim) && (bchunk<block_chunks)) {
			s_blocks[threadIdx.x]=UINT_MAX;
			if (threadIdx.x < (bdim>>ABV_LOG_WORD_SIZE)
				&& threadIdx.x < asize-((bchunk*bdim)>>ABV_LOG_WORD_SIZE)) {
				s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*bdim)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
			}
		}		
		/* and thread 257 will do the hard work, iterate and update configuration */
		else if (threadIdx.x==bdim) {
			max_b=next_b;
			next_b=0;
		}
		
		__syncthreads();
		
		/* Simulating a random re-ordering through thread scheduling */
		if ((threadIdx.x<bdim) && (block < besize) && d_is_active(threadIdx.x,s_abv)) {
			s_blocks[atomicInc(&next_b,bdim+2)]=block;
		}
		/* Hard work for thread 257 */
		else if (threadIdx.x==bdim && bchunk>0) {
			for (int b=0; b<max_b; b++) {
				uint min=UINT_MAX;
				
				block=s_blocks_update[b];

				uint o_init=ruleblock.lhs_idx[block];
				uint o_end=ruleblock.lhs_idx[block+1];
				for (int o=o_init; o < o_end; o++) {;
					uint obj=lhs.object[o];
					uint membr=lhs.mmultiplicity[o];
					uint mult=GET_MULTIPLICITY(membr);
					membr=GET_MEMBR(membr);

					uint value=configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]/mult;
										
					min=(value < min) ? value : min;
				}
				if (min>0) {
					for (int o=o_init; o < o_end; o++) {
						uint obj=lhs.object[o];
						uint membr=lhs.mmultiplicity[o];
						uint mult=GET_MULTIPLICITY(membr);
						membr=GET_MEMBR(membr);

						configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]-=min*mult;
					}
					nb[sim*options.num_environments*besize+env*besize+block]+=min;
				}
			}
		}
		
		__syncthreads();
		
		//TODO: delete this, just for debuggin purposes
		/*if (threadIdx.x < bdim && block < besize && bchunk < block_chunks)
			nr[sim*options.num_environments*besize+env*besize+block] = s_blocks[threadIdx.x];
		*/
		uint* aux=s_blocks;
		s_blocks=s_blocks_update;
		s_blocks_update=aux;
		
		__syncthreads();
		
		
		//__syncthreads();
		
		// TODO: First solution (following KISS methodology): 
		//       ThreadIdx.x==0 will update everything, and rest of threads will
		//       compute next s_blocks
	}
}


/******************************************************/
/* Kernel for Phase 2, version 2: attempt for speedup */
/******************************************************/
__global__ void kernel_phase2_blhs(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		struct _options options,
		uint * d_abv) {
	
	extern __shared__ uint sData[];
	__shared__ uint next_b,max_it;
	
	uint bdim = blockDim.x;
	uint * s_abv = sData;
	uint * s_blocks = sData+(bdim >> ABV_LOG_WORD_SIZE);
	uint * s_itorder = s_blocks + bdim;
	uint * s_blhs = s_itorder + bdim;
	
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint block_chunks=(besize + bdim -1)>>CU_LOG_THREADS;
	
	uint o_init,o_end,o_length;
	
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {

		block=bchunk*bdim+threadIdx.x;

		// Initialize s_blocks
		s_blocks[threadIdx.x]=UINT_MAX;
		// Initialize order
		s_itorder[threadIdx.x]=0;
		// Initialize s_blhs
		/*for (int i=0;i<options.max_lhs;i++) {
			s_blhs[threadIdx.x+i*bdim]=EMPTY;
		}*/
		// Initialize s_abv
		if (threadIdx.x < (bdim>>ABV_LOG_WORD_SIZE)
			&& threadIdx.x < asize-((bchunk*bdim)>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=d_abv[sim*options.num_environments*asize+env*asize+((bchunk*bdim)>>ABV_LOG_WORD_SIZE)+threadIdx.x];
		}
		else if (threadIdx.x==(bdim>>ABV_LOG_WORD_SIZE)) {
			next_b=0;
			max_it=0;
		}
		
		__syncthreads();
		
		// Simulating a random re-ordering through thread scheduling 
		// TODO: Implement real random order
		if (block < besize && d_is_active(threadIdx.x,s_abv)) {
			s_blocks[atomicInc(&next_b,bdim+2)]=block;
		}
		
		__syncthreads();
		
		// If there are not active blocks in the chunk
		if (next_b==0) continue;
		
		// Initialize s_blhs with objects from active blocks
		if (threadIdx.x<next_b) {
			block=s_blocks[threadIdx.x];

			o_init=ruleblock.lhs_idx[block];
			o_end=ruleblock.lhs_idx[block+1];
			o_length=o_end-o_init;
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);
				s_blhs[threadIdx.x*options.max_lhs+o-o_init]=
					OBJECT(obj,membr,mult);
			}
			for (int o=o_length; o<options.max_lhs; o++) {
				s_blhs[threadIdx.x*options.max_lhs+o]=EMPTY;
			}
		}
		
		// Initialize order
		//s_itorder[threadIdx.x]=0;
		
		__syncthreads();
		
		// Calculate object collisions
		for (int i=0; i<next_b; i++) {
			if (threadIdx.x > i && threadIdx.x < next_b) {
				for (int o=0; o < o_length; o++) {
					uint tobj=s_blhs[threadIdx.x*options.max_lhs+o];
					for (int o2=0; o2 < options.max_lhs; o2++) {
						uint iobj=s_blhs[i*options.max_lhs+o2];
						if (IS_EMPTY(iobj))
							break;
						if (COLLISION(iobj,tobj)) {
							s_blhs[threadIdx.x*options.max_lhs+o] =
								OBJECT_COLLISION(tobj,i,o2);
							break; // TODO: Check if this is inneficient
						}
					}
				}
			}
			__syncthreads();
		}
		
		// Calculate iteration order
		for (int i=0; i<next_b; i++) {
			if (threadIdx.x==i) {
				for (int o=0; o < o_length; o++) {
					uint obj=s_blhs[threadIdx.x*options.max_lhs+o];
					if (IS_COLLISION(obj)) {
						uint a=s_itorder[threadIdx.x];
						uint b=s_itorder[COLLISION_GET_TID(obj)]+1;
						a = (a>b)? a : b;
						s_itorder[threadIdx.x] = a;
						max_it = (max_it < a)? a : max_it;
					}							
				}
			}
			__syncthreads();
		}
				
		// Upload multiplicities
		if (threadIdx.x<next_b)
		for (int o=0;o<o_length;o++) {
			uint obj=s_blhs[threadIdx.x*options.max_lhs+o];
			if (!IS_COLLISION(obj)) {
				uint mult = configuration.multiset[D_MU_IDX(GET_OBJECT(obj),0)];
				s_blhs[threadIdx.x*options.max_lhs+o]=SET_CONF_MULT(obj,mult);
			}
		}
		
		__syncthreads();
		
		// Calculate minimum applications
		for (int it=0;it<=max_it;it++) {
			if (threadIdx.x<next_b && s_itorder[threadIdx.x]==it) {
				uint min=UINT_MAX;
				uint value=0;
				// Calculate minimums
				for (int o=0; o < o_length; o++) {
					uint obj=s_blhs[threadIdx.x*options.max_lhs+o];
					if (IS_COLLISION(obj)) {
						uint obj2;
						s_blhs[threadIdx.x*options.max_lhs+o] = obj2 =
						SET_CONF_MULT(obj,
							GET_CONF_MULT(s_blhs[COLLISION_GET_TID(obj)*options.max_lhs+COLLISION_GET_OBJ(obj)]));
						s_blhs[COLLISION_GET_TID(obj)*options.max_lhs+COLLISION_GET_OBJ(obj)]=EMPTY;
						//obj=s_blhs[threadIdx.x*options.max_lhs+o];
						obj=obj2;
					}
					value=GET_CONF_MULT(obj)/GET_MULT(obj);
					min = (value < min)? value : min;
				}
				if (min>0) { // TODO: how efficient is without this?
					s_itorder[threadIdx.x]=max_it+min;
					// Update multiplicities
					for (int o=0; o < o_length; o++) {
						uint obj=s_blhs[threadIdx.x*options.max_lhs+o];
						s_blhs[threadIdx.x*options.max_lhs+o] =
							SET_CONF_MULT(obj,
								GET_CONF_MULT(obj)-min*GET_MULT(obj));
					}
				}
			}
			__syncthreads();
		}
		
		// Update nb
		if (threadIdx.x < next_b && s_itorder[threadIdx.x]>max_it) {
			//nb[sim*options.num_environments*besize+env*besize+s_blocks[threadIdx.x]]+=s_itorder[threadIdx.x]-max_it;
			//nb[D_NB_IDX(s_blocks[threadIdx.x])]+=s_itorder[threadIdx.x]-max_it;
			nb[D_NB_IDX(block)]+=s_itorder[threadIdx.x]-max_it;
		}
		
		__syncthreads();
		
		// Update configuration
		if (threadIdx.x<next_b) {
			for (int o=0; o < o_length; o++) {
				uint obj=s_blhs[threadIdx.x*options.max_lhs+o];
				if (!IS_EMPTY(obj)) {
					configuration.multiset[D_MU_IDX(lhs.object[o+o_init],GET_MEMBR(lhs.mmultiplicity[o+o_init]))]
						=GET_CONF_MULT(obj);
				}
			}
		}
		
		//TODO: delete this, just for debuggin' purposes
		//if (threadIdx.x==0 && block < besize)
			//nr[sim*options.num_environments*besize+env*besize+block] = next_b;//s_blocks[threadIdx.x];
			//nr[sim*options.num_environments*besize+env*besize+threadIdx.x+bchunk] = next_b;//s_blocks[threadIdx.x];
		//if (threadIdx.x==2 && block < besize)
			//nr[sim*options.num_environments*besize+env*besize+threadIdx.x+bchunk] = max_it;//s_blocks[threadIdx.x];
		
		//__syncthreads();
	}
}


/************************************************/
/* Implementation of Phase 2 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase2(){
	
//	if (options->verbose>0)	cout << endl << "--------------------------" << endl <<	"Launching GPU code for phase 2" << endl;
	pdp_out->print_dcba_phase(2);

    pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 2");
	
	if (runcomp) {
	//counters.timer = 0;
	counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0.0f;
	counters.timek1cpu = counters.timek2cpu = counters.timek3cpu = 1.0f; }
	
	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;

	/* Apply kernel for Phase 2 */
	if (runcomp) {
		//	if (options->verbose>0) cout << endl << "Launching kernel for phase 2: ";
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 2");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}

	if (mode==2) {
		if (pdp_out->will_print_dcba_phase())
			cout << "(using basic kernel)"<<endl;
		
		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads+1);
		size_t sh_mem=((cu_threads>>ABV_LOG_WORD_SIZE) + 2*cu_threads)*sizeof(uint);
	
		kernel_phase2_generic <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nb, 
			d_structures->nr, *options, d_abv);
	
		getLastCudaError("kernel for phase 2 launch failure");
		//cutilDeviceSynchronize();	
		cudaDeviceSynchronize(); // Compatible for CUDA < 4.0
	}
	else if (mode<2) {
		if (pdp_out->will_print_dcba_phase())
			cout << "(using blhs kernel)"<<endl;
	
		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads);
		size_t sh_mem=((cu_threads>>ABV_LOG_WORD_SIZE) + 2*cu_threads + options->max_lhs*cu_threads)*sizeof(uint);
	
		kernel_phase2_blhs <<<dimGrid,dimBlock,sh_mem>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nb, 
			d_structures->nr, *options, d_abv);
	
		getLastCudaError("kernel for phase 2 launch failure");
		//cutilDeviceSynchronize();	
		cudaDeviceSynchronize(); // Compatible for CUDA < 4.0
	}
	
	if (runcomp) {
		sdkStopTimer(&counters.timer);
		counters.timesp2gpu=sdkGetTimerValue(&counters.timer);
		pdp_out->print_profiling_dcba_microphase_result(counters.timesp2gpu);
//		if (options->verbose>0) cout << counters.timesp2gpu << "ms." << endl;
	}

	pdp_out->print_block_selection();

	pdp_out->print_temporal_configuration();


	/**************************************/
	/* PROFILING AND CHECK-OUT PROCEDURES */
	/**************************************/

	if (runcomp && pdp_out->will_print_dcba_phase()) {
		/***************************************/
		/***** TEMPORAL TESTING PROCEDURE ******/
		/***************************************/
		
		/* Checking ABV */
		pdp_out->print_profiling_dcba_microphase_name("Checking maximality");
		ABV_T *debug_abv= new ABV_T[abv_size];
		
		checkCudaErrors(cudaMemcpy(debug_abv, d_abv, abv_size*sizeof(ABV_T), cudaMemcpyDeviceToHost));

		d_cfg.multiset = new MULTIPLICITY[structures->configuration.multiset_size];
				
		checkCudaErrors(cudaMemcpy(d_cfg.multiset, d_structures->configuration.multiset, d_structures->configuration.multiset_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));

		//checkCudaErrors(cudaMemcpy(structures->nr, d_structures->nr, d_structures->nr_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(d_nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
		
//		print_block_applications(d_nb);
//
//		print_configuration(d_cfg);
		
		
		/*for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++) {
			cout << "Sim " << sim;
			for (unsigned int env=0; env<options->num_environments; env++) {
				cout << endl;
				cout << ", Env " << env;
				cout << ". Next_b: " << structures->nr[sim*options->num_environments*besize+env*besize] <<
					", " << structures->nr[sim*options->num_environments*besize+env*besize+1];
				cout << ". Max_it: " << structures->nr[sim*options->num_environments*besize+env*besize+2] <<
					", " << structures->nr[sim*options->num_environments*besize+env*besize+3];
			}
			cout << endl;
		}*/
		
		/* Check maximality from the GPU */
		uint num_ap_b=0;
		uint num_abv_b=0;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (unsigned int env=0; env < options->num_environments; env++) {
				for (unsigned int block=0; block<besize; block++) {
					bool is_block_active=(debug_abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
						       >> ((~block)&ABV_DESPL_MASK))& 0x1;
					if (!is_block_active) continue;

					num_abv_b++;
					bool applicable=true;
					uint o_init=structures->ruleblock.lhs_idx[block];
					uint o_end=structures->ruleblock.lhs_idx[block+1];
					for (uint o=o_init; o < o_end; o++) {
						uint obj=structures->lhs.object[o];
						uint membr=structures->lhs.mmultiplicity[o];
						uint mult=GET_MULTIPLICITY(membr);
						membr=GET_MEMBR(membr);

						// Check if we have enough objects to apply the block
						if (d_cfg.multiset[MU_IDX(obj,membr)]<mult)
							applicable=false;
					}
					if (applicable) {
						num_ap_b++;
						cout << "Error: at sim " << sim << ", env " << env << ", block:" << block;
						uint min=UINT_MAX,val=0;
						for (uint o=o_init; o < o_end; o++) {
							uint obj=structures->lhs.object[o];
							uint membr=structures->lhs.mmultiplicity[o];
							uint mult=GET_MULTIPLICITY(membr);
							membr=GET_MEMBR(membr);

							// Check if we have enough objects to apply the block
							//if (d_cfg.multiset[MU_IDX(obj,membr)]>=mult)
							cout << " [obj" << obj << "]" << membr << "*" << mult << "--" << d_cfg.multiset[MU_IDX(obj,membr)];
							
							val=d_cfg.multiset[MU_IDX(obj,membr)]/mult;
							min=(val<min)?val:min;
						}
						cout << " ==> " << min << " apps" << endl;
					}
				}
			}
		}
		delete []debug_abv;
		delete []d_cfg.multiset;

		pdp_out->print_profiling_dcba_microphase_result(num_ap_b==0 && num_abv_b==0);

		if (num_ap_b>0 || num_abv_b>0)
			cout << "Error from phase2 at GPU: we have " << num_ap_b << " block applications still to perform ("<< num_abv_b << " from abv active blocks)"  << endl;
	}
	
	
	/*******************************/
	/* REPRODUCING CODE ON THE CPU */
	/*******************************/
	if (runcomp)
		gold_selection_phase2();
	
	if (runcomp && pdp_out->will_print_dcba_phase()) {
		// Checking maximality from CPU
		uint num_ap_b=0;
		uint num_abv_b=0;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (unsigned int env=0; env < options->num_environments; env++) {
				for (unsigned int block=0; block<besize; block++) {
					if (! is_active(block,env,sim)) continue;
					num_abv_b++;
					bool applicable=true;
					uint o_init=structures->ruleblock.lhs_idx[block];
					uint o_end=structures->ruleblock.lhs_idx[block+1];
					for (uint o=o_init; o < o_end; o++) {
						uint obj=structures->lhs.object[o];
						uint membr=structures->lhs.mmultiplicity[o];
						uint mult=GET_MULTIPLICITY(membr);
						membr=GET_MEMBR(membr);

						// Check if we have enough objects to apply the block
						if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult)
							applicable=false;
					}
					if (applicable)
						num_ap_b++;
				}
			}
		}

		if (num_ap_b>0 || num_abv_b>0)
			cout << "Error from phase2 at CPU: we have " << num_ap_b << " block applications still to perform (" << num_abv_b << " from abv active blocks)"  << endl;

		// Printing times
		cout << endl << "Time for phase 2: GPU=" << counters.timesp2gpu << " ms, CPU=" << counters.timesp2cpu << " ms, speedup=" << counters.timesp2cpu/counters.timesp2gpu << "x" << endl;
	}
	
	pdp_out->print_end_profiling_dcba_phase();

	return true;
}


/*********************************************/
/*********************/
/* Selection Phase 3 */
/*********************/

/**********************/
/* Kernel for Phase 3 */
/**********************/
__global__ void kernel_phase3(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		PDP_Psystem_REDIX::Probability probability,
		uint rpsize,
		uint resize,
		struct _options options) {
		
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
	
	for (int bchunk=0; bchunk < block_chunks; bchunk++) {
		block=bchunk*blockDim.x+threadIdx.x;
		
		if (block >= besize) break;
		
		int rule_ini=ruleblock.rule_idx[block];
		int rule_end=ruleblock.rule_idx[block+1];
		
		uint N=0;//nb[D_NB_IDX(block)];
		uint membr=ruleblock.membrane[block];

		if (block<options.num_rule_blocks || env==GET_ENVIRONMENT(membr))
			N=nb[D_NB_IDX(block)];
		
		if (N==0) {
			for (uint r = rule_ini; r < rule_end; r++) {
				if (block < options.num_rule_blocks)
					nr[D_NR_P_IDX(r)] = 0;
				else if (env==GET_ENVIRONMENT(membr))
					nr[D_NR_E_IDX(r)] = 0;
			}		 		
		}
		else {
			// Update charges
			configuration.membrane[D_CH_IDX(GET_MEMBRANE(membr))]=GET_BETA(membr);
			
			float cr=0.0f,d=1.0f;

			for (uint r = rule_ini; r < rule_end; r++) {
				float p=0.0f;
				uint val=0;

				if (IS_ENVIRONMENT(membr)) {
					p=probability[options.num_environments*rpsize+(r-rpsize)];
				}
				else {
					p=probability[env*rpsize+r];
				}

				cr = fdividef(p,d);
				
				if (cr > 0.0f) {
					if (r == rule_end-1)
						val=N;
					else {
						val=curng_binomial_random (N, cr);
					}
				}

				if (!IS_ENVIRONMENT(membr))
					nr[D_NR_P_IDX(r)] = val;
				else 
					nr[D_NR_E_IDX(r)] = val;

				N-=val;
				d*=(1-cr);
			}
		}
		__syncthreads();
	}
}

/************************************************/
/* Implementation of Phase 3 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase3() {

//	if (options->verbose>0) { cout << endl << "--------------------------" << endl; cout << "Launching GPU code for phase 3" << endl; }
	pdp_out->print_dcba_phase(3);

	pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 3");
	
	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;

	/* Apply kernel for Phase 3 */
	
	if (runcomp) {
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 3");
//		if (options->verbose>0)	cout << endl << "Launching kernel for phase 3: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}

	dim3 dimGrid (cu_blocksx, cu_blocksy);
	dim3 dimBlock (cu_threads);

	kernel_phase3 <<<dimGrid,dimBlock>>> (d_structures->ruleblock,
		d_structures->configuration, d_structures->nb, 
		d_structures->nr, d_structures->probability, d_structures->pi_rule_size,
		d_structures->pi_rule_size+d_structures->env_rule_size, *options);
	 
	getLastCudaError("kernel for phase 3 launch failure");	
	cudaDeviceSynchronize(); // Compatible for CUDA < 4.0

	if (runcomp) {
		sdkStopTimer(&counters.timer);
		counters.timesp3gpu=sdkGetTimerValue(&counters.timer);
//		if (options->verbose>0)	cout << counters.timesp3gpu << "ms." << endl;
		pdp_out->print_profiling_dcba_microphase_result(counters.timesp3gpu);
	}
	
	pdp_out->print_rule_selection();


	/**************************************/
	/* PROFILING AND CHECK-OUT PROCEDURES */
	/**************************************/

	/* if (options->verbose>1) {
		// Temporal testing procedure
	
		checkCudaErrors(cudaMemcpy(structures->nr, d_structures->nr, d_structures->nr_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(d_nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
	
		print_block_applications(d_nb);
		cout << "Checking data from GPU:" << endl;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++)
			for (unsigned int env=0; env<options->num_environments; env++)
				for (unsigned int block=0; block<besize; block++) {
					int rule_ini=structures->ruleblock.rule_idx[block];
					int rule_end=structures->ruleblock.rule_idx[block+1];

					unsigned int N=d_nb[NB_IDX];

					cout << "Sim=" << sim << ", env=" << env << ", block=" << block << ", N=" << N << ":";

					for (unsigned int r = rule_ini; r < rule_end; r++) {
						float p=0.0;

						if (r<rpsize) {
							p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
							cout << "r_" << r-rule_ini << "(p=" << p << ",n=" << structures->nr[NR_P_IDX] << "), ";
						}
						else if (GET_ENVIRONMENT(structures->ruleblock.membrane[block])==env) {
							p=structures->probability[env*rpsize+r];
							cout << "r_" << r-rule_ini << "(p=" << p << ",n=" << structures->nr[NR_E_IDX] << "), ";
						}
					}
					cout << endl;
				}
	}*/
		
	
	/*******************************/
	/* REPRODUCING CODE ON THE CPU */
	/*******************************/
	if (runcomp)
		gold_selection_phase3();
	
	if (runcomp && pdp_out->will_print_rule_selection()) {
		cout << "Checking data from CPU:" << endl;
		for (unsigned int sim=0; sim < options->num_parallel_simulations; sim++)
			for (unsigned int env=0; env<options->num_environments; env++)
				for (unsigned int block=0; block<besize; block++) {
					int rule_ini=structures->ruleblock.rule_idx[block];
					int rule_end=structures->ruleblock.rule_idx[block+1];

					unsigned int N=structures->nb[NB_IDX];

					cout << "Sim=" << sim << ", env=" << env << ", block " << block << ", N=" << N << ":";

					for (unsigned int r = rule_ini; r < rule_end; r++) {
						float p=0.0;

						if (r<rpsize) {
							p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
							cout << "r_" << r-rule_ini << "(p=" << p << ",n=" << structures->nr[NR_P_IDX] << "), ";
						}
						else if (GET_ENVIRONMENT(structures->ruleblock.membrane[block])==env) {
							p=structures->probability[env*rpsize+r];
							cout << "r_" << r-rule_ini << "(p=" << p << ",n=" << structures->nr[NR_E_IDX] << "), ";
						}
					}
					cout << endl;
				}
	}
	
	if (runcomp && pdp_out->will_print_dcba_phase())
		cout << endl << "Time for phase 3: GPU=" << counters.timesp3gpu << " ms, CPU=" << counters.timesp3cpu << " ms, speedup=" << counters.timesp3cpu/counters.timesp3gpu << "x" << endl;
	
	pdp_out->print_end_profiling_dcba_phase();

	return true;
}

/*********************************************/
/***********************/
/* Execution (Phase 4) */
/***********************/

/**********************/
/* Kernel for Phase 4 */
/**********************/
__global__ void kernel_phase4 (PDP_Psystem_REDIX::Rule rule,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Rhs rhs,
			PDP_Psystem_REDIX::NR nr,
			uint rpsize,
			uint resize,
			uint re_chunk,
			struct _options options) {
	
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint r=threadIdx.x;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint rp_chunks=(rpsize + blockDim.x -1)>>CU_LOG_THREADS;
	
	/* Rules of Pi, executed by each environment */
	for (int rchunk=0; rchunk < rp_chunks; rchunk++) {
		r=rchunk*blockDim.x+threadIdx.x;
		
		uint N=0;
		
		if (r < rpsize)
			N=nr[D_NR_P_IDX(r)];
		
		if (N>0) {	
			int o_ini=rule.rhs_idx[r];
			int o_end=rule.rhs_idx[r+1];
			
			for (int o=o_ini; o<o_end; o++) {
				uint obj=rhs.object[o];
				uint mult=rhs.mmultiplicity[o];
				uint membr=GET_MEMBR(mult);
				mult=GET_MULTIPLICITY(mult);

				atomicAdd(&(configuration.multiset[D_MU_IDX(obj,membr)]),N*mult);
			}			
		}
		//__syncthreads();
	}

	/* Communication rules, distributed among the environments */
	uint reini=rpsize+env*re_chunk;
	uint reend=rpsize+(env+1)*re_chunk;
	uint it=0;
	
	r = reini+(it++)*blockDim.x+threadIdx.x;
	
	while ((r<resize) && (r<reend)) {
		int o_ini=rule.rhs_idx[r];
		int o_end=rule.rhs_idx[r+1];

		uint N=nr[D_NR_E_IDX(r)];

		if (N>0)
		for (int o=o_ini; o<o_end; o++) {
			uint obj=rhs.object[o];
			uint denv=rhs.mmultiplicity[o];
			
			obj=sim*options.num_environments*esize+
				denv*esize+obj;
			
			atomicAdd(&(configuration.multiset[obj]),N);
		}
		r = reini+(it++)*blockDim.x+threadIdx.x;
	}
}


int Simulator_gpu_dir::execution() {
		
//	if (options->verbose>0) { cout << endl << "--------------------------" << endl; cout << "Launching GPU code for phase 4" << endl; }
	pdp_out->print_dcba_phase(4);

	pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 4");
	
	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;
	uint re_chunk=0;
	if (d_structures->env_rule_size < 32*options->num_environments) {
		re_chunk=d_structures->env_rule_size;
	}
	else {
		re_chunk=d_structures->env_rule_size/options->num_environments+
		(d_structures->env_rule_size%options->num_environments)==0?0:1;
	}
		
	
	/* Apply kernel for Phase 4 */
	if (runcomp) {
		//	if (options->verbose>0)	cout << endl << "Launching kernel for phase 4: ";
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 4");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}
	
	dim3 dimGrid (cu_blocksx, cu_blocksy);
	dim3 dimBlock (cu_threads);

	kernel_phase4 <<<dimGrid,dimBlock>>> (d_structures->rule,
		d_structures->configuration, d_structures->rhs, 
		d_structures->nr, d_structures->pi_rule_size,
		d_structures->pi_rule_size+d_structures->env_rule_size,
		re_chunk, *options);
	 
	getLastCudaError("kernel for phase 4 launch failure");
	//cutilDeviceSynchronize();	
	cudaDeviceSynchronize(); // Compatible for CUDA < 4.0

	if (runcomp) {
		sdkStopTimer(&counters.timer);
		counters.timesp4gpu=sdkGetTimerValue(&counters.timer);
//		if (options->verbose>0)	cout << counters.timesp4gpu << "ms." << endl;
		pdp_out->print_profiling_dcba_microphase_result(counters.timesp4gpu);
	}


	/**************************************/
	/* PROFILING AND CHECK-OUT PROCEDURES */
	/**************************************/
	
//	if (options->verbose>1) {
//		/* Temporal testing procedure */
//		cout << "Checking data from GPU:" << endl;
//		d_cfg.multiset = new MULTIPLICITY[structures->configuration.multiset_size];
//
//		checkCudaErrors(cudaMemcpy(d_cfg.multiset, d_structures->configuration.multiset, d_structures->configuration.multiset_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
//
//		print_configuration(d_cfg);
//
//		delete []d_cfg.multiset;
//	}
		
	
	/*******************************/
	/* REPRODUCING CODE ON THE CPU */
	/*******************************/
	if (runcomp) 
		gold_execution();
		
//	if (runcomp && options->verbose>1) {
//		cout << "Checking data from CPU:" << endl;
//		print_configuration(structures->configuration);
//	}
	
	if (runcomp && pdp_out->will_print_dcba_phase())
		cout << endl << "Time for phase 4: GPU=" << counters.timesp4gpu << " ms, CPU=" << counters.timesp4cpu << " ms, speedup=" << counters.timesp4cpu/counters.timesp4gpu << "x" << endl;

	pdp_out->print_end_profiling_dcba_phase();

	return 0;
}


/*******************************************/
/* Methods of the GPU wrapper for printing */
/*******************************************/

void PDP_Psystem_redix_out_std_gpuwrapper::retrieve_configuration() {
	checkCudaErrors(cudaMemcpy(structures->configuration.multiset, d_structures->configuration.multiset, d_structures->configuration.multiset_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(structures->configuration.membrane, d_structures->configuration.membrane, d_structures->configuration.membrane_size*sizeof(CHARGE), cudaMemcpyDeviceToHost));
}

void PDP_Psystem_redix_out_std_gpuwrapper::retrieve_block() {
	checkCudaErrors(cudaMemcpy(structures->nb, d_structures->nb, d_structures->nb_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
}

void PDP_Psystem_redix_out_std_gpuwrapper::retrieve_rule() {
	checkCudaErrors(cudaMemcpy(structures->nr, d_structures->nr, d_structures->nr_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
}

void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_table () {
    /* Output profiling information */
	/* Independently of the verbosity level */
	if (runcomp) {
		cout << endl << "-------------------------" << endl;
			cout << "Profiling mode enabled" << endl << "-------------------------" << endl;

		float totalselgpu=counters->timesp1gpu+counters->timesp2gpu+counters->timesp3gpu;
		float totalselcpu=counters->timesp1cpu+counters->timesp2cpu+counters->timesp3cpu;
		float totalexgpu=counters->timesp4gpu;
		float totalexcpu=counters->timesp4cpu;
		float totalgpu=totalselgpu+totalexgpu;
		float totalcpu=totalselcpu+totalexcpu;

		cout << "Time information summary:" << endl << "-------------------------" << endl;
		cout << "* Time in parts"
			<< endl << "Speedup on phases: phase1=" << counters->timesp1cpu/counters->timesp1gpu << "x, phase2=" << counters->timesp2cpu/counters->timesp2gpu << "x, phase3=" <<counters->timesp3cpu/counters->timesp3gpu << "x, phase4=" <<counters->timesp4cpu/counters->timesp4gpu << "x"
			<< endl << "Time of phases on GPU: phase1=" << counters->timesp1gpu << "ms, phase2=" << counters->timesp2gpu << "ms, phase3=" << counters->timesp3gpu << "ms, phase4=" << counters->timesp4gpu << "ms"
			<< endl << "Time of phases on CPU: phase1=" << counters->timesp1cpu << "ms, phase2=" << counters->timesp2cpu << "ms, phase3=" <<counters->timesp3cpu << "ms, phase4=" <<counters->timesp4cpu << "ms"
			<< endl << "Profiling GPU: phase1=" << counters->timesp1gpu*100.0/totalgpu << "%, phase2=" << counters->timesp2gpu*100.0/totalgpu << "%, phase3=" <<counters->timesp3gpu*100.0/totalgpu << "%, phase4=" <<counters->timesp4gpu*100.0/totalgpu << "%"
			<< endl << "Profiling CPU: phase1=" << counters->timesp1cpu*100.0/totalcpu << "%, phase2=" << counters->timesp2cpu*100.0/totalcpu << "%, phase3=" <<counters->timesp3cpu*100.0/totalcpu << "%, phase4=" <<counters->timesp4cpu*100.0/totalcpu << "%"
			<< endl << "Profiling selection GPU: phase1=" << counters->timesp1gpu*100.0/totalselgpu << "%, phase2=" << counters->timesp2gpu*100.0/totalselgpu << "%, phase3=" <<counters->timesp3gpu*100.0/totalselgpu << "%"
			<< endl << "Profiling selection CPU: phase1=" << counters->timesp1cpu*100.0/totalselcpu << "%, phase2=" << counters->timesp2cpu*100.0/totalselcpu << "%, phase3=" <<counters->timesp3cpu*100.0/totalselcpu << "%" << endl;

		cout << "* Total time" << endl
			<< "Total selection time: GPU=" << totalselgpu << "ms, " << "CPU=" << totalselcpu << "ms, "
			<< "Speedup=" << totalselcpu/totalselgpu << "x"
			<< endl << "Total execution time: GPU=" << totalexgpu << "ms, " << "CPU=" << totalexcpu << "ms"
			<< ", Speedup=" << totalexcpu/totalexgpu << "x"
			<< endl << "Total time: GPU=" << totalgpu << ", CPU=" << totalcpu << ", Speedup=" << totalcpu/totalgpu << "x"
			<< endl;
	}
}

void PDP_Psystem_redix_out_std_gpuwrapper::print_temporal_configuration () {
	if (!pdpout->will_print_temporal_configuration()) return;

	retrieve_configuration();

	for (int sim=0;sim<options->num_parallel_simulations;sim++) {
		pdpout->print_simulation(psb+sim);
		pdpout->print_temporal_configuration(sim);
	}
}

void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_dcba_phase (const char * message) {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	cout << endl << "--------------------------" << endl <<
			message << endl;
}

void PDP_Psystem_redix_out_std_gpuwrapper::print_end_profiling_dcba_phase () {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	cout << "--------------------------" << endl<<endl;
}

void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_dcba_microphase_name (const char * message) {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	cout << message << ": ";

	cout.flush();
}

// This function should be called after executing the microphase
void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_dcba_microphase_result (float time) {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	cout << time << "ms." << endl;

	cout.flush();
}

// This function should be called after executing the microphase
void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_dcba_microphase_result (bool result) {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	if (result) cout << "OK." << endl;
	else cout << "FAILED." << endl;

	cout.flush();
}

// This function should be called after executing the microphase
void PDP_Psystem_redix_out_std_gpuwrapper::print_profiling_dcba_microphase_datum (const char* message1, float datum, const char* message2) {
	if (!runcomp && !pdpout->will_print_dcba_phase()) return;

	cout << message1 << " " << datum << " " << message2 << endl;

	cout.flush();
}
