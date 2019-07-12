/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2015  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Author: Miguel Ángel Martínez-del-Amor
    Contributors: Adolfo Gastalver
     	 	 	  Andrés Doncel
    
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
 * roadmap. Natural Computing, 15, 4 (2016), 565-573.
 */

#include "simulator_gpu_dir.h"

#include "curng_binomial.h"
#include "competition.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <timestat.h>
#include <cstdlib>
#include <future>

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

//Using constant memory to load as symbols results in no real gain (nor loss)
__constant__ _options d_options;
__constant__ _computations d_computations;

/***************************************************************************/

/* Constructor of the class */

Simulator_gpu_dir::Simulator_gpu_dir(PDP_Psystem_REDIX* PDPps,int mode,bool accurate, PDP_Psystem_output* PDPout) {
	this->structures=PDPps->structures;
	this->options=PDPps->options;
	runcomp=(mode==1);
	profiling=(mode==3);
	this->mode=mode;
	this->accurate=accurate; // use n/d mode for normalization by default
	error=false;
	init();

	// The real out (binary, csv...)
	this->PDPout=PDPout;

	/* Configure the standard output class (debugging purposes only) */
	/* Let keep it here, after init(). The initial configuration is */
	/* backed up there, and d_structures is initialized */
	pdp_out = new PDP_Psystem_redix_out_std_gpuwrapper(PDPps,this->d_structures,runcomp,&counters,profiling);
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

		if (PDPout)
			PDPout->write_configuration(structures->configuration.multiset,structures->configuration.membrane,psb,0,structures->stringids.id_objects);
		
		if (sim_parallel > options->num_simulations - psb)
			options->num_parallel_simulations=options->num_simulations - psb;
		
		reset(psb);

		pdp_out->print_sim_range(psb,psb+options->num_parallel_simulations);

		auto handle = std::async(std::launch::async,
						&Simulator_gpu_dir::do_nothing,this);

        /* MAIN LOOP */
		for (uint i=0; i<k; i++) {
            pdp_out->print_step(i);

			if (selection(i))
				return false;

			if (execution(i))
				return false;

			//Check consistency and updating errors
			//Each cycle or if its last step
			if ((i+1==k||(i+1)%options->error_cycle==0) && check_step_errors())
				return false;

            pdp_out->print_configuration();
            if (PDPout && ((i+1)%options->cycles==0)) {
            	//Wait for possible previous copy to end
            	cudaStreamSynchronize(copy_stream);

            	//Copy dev2dev
            	retrieve_copy();

            	//We must copy first
            	cudaStreamSynchronize(execution_stream);

            	//Wait for previous write to copy to host
            	handle.wait();

            	//Copy dev2host
            	retrieve_async(psb);

            	handle = std::async(std::launch::async,
            			&Simulator_gpu_dir::write_async,this,psb,i);
            }

		}

	}

    /* Output profiling information */
	pdp_out->print_profiling_table();


	return true;
}
// The function we want to execute on the new thread.
void Simulator_gpu_dir::write_async(int psb,int i)
{
	unsigned int* output_multiset_pointer=structures->configuration.multiset;
	//Wait until the copy to host has finished
	cudaStreamSynchronize(copy_stream);

	//cout << "Writing..." << endl;

	if(options->output_filter!=NULL){

		if(options->GPU_filter){
			output_multiset_pointer=output_multiset;
		}

		//Filtered configuration
		for (uint simu=psb; (simu <psb+sim_parallel) && (simu < options->num_simulations); simu++)
			PDPout->write_configuration_filtered(output_multiset_pointer,structures->configuration.membrane,simu,i+1,structures->stringids.id_objects);

	}else{
		for (uint simu=psb; (simu <psb+sim_parallel) && (simu < options->num_simulations); simu++)
			PDPout->write_configuration(output_multiset_pointer,structures->configuration.membrane,simu,i+1,structures->stringids.id_objects);
	}
	//cout << "Finished writing. Next..." << endl;
}
// Aux function, does nothing
void Simulator_gpu_dir::do_nothing()
{
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

inline size_t memory_human_readable(size_t bytes, char& u) {
	int count=0;
	float div=1;
	char unit[6]={' ','K','M','G','T','P'};
	while ((bytes/div)>1023 && count<3){
		div*=1024;
		count++;
	}

	u = unit[count];
	return bytes/div;
}

inline void print_memory_bytes_human(const char* message, size_t bytes) {
	char unit;
	size_t memh = memory_human_readable(bytes,unit);

	cout << message << bytes << " (" << memh << unit << "B)" << endl;
}

/* in reference parameters start and end writes the starting and ending block idx of the module, if it is for Pi or for Env (according to pirules) */
inline void module_start_end (uint & start, uint & end, int m, bool pirules, Options options) {
	start = 0;
	end = options->num_rule_blocks;

	if (pirules) { // Pi rules
		start = options->modules_pi_index[m];
		if (m < options->modules-1) end = options->modules_pi_index[m+1];
	} else {       // env rules
		start = options->modules_env_index[m] + options->num_rule_blocks;
		if (m < options->modules-1) end = options->modules_env_index[m+1] + options->num_rule_blocks;
		else end+= options->num_blocks_env;
	}
}


//TODO: Make this member to return a boolean value, to check errors
bool Simulator_gpu_dir::init() {

	checkCudaErrors(cudaStreamCreate (&execution_stream));
	checkCudaErrors(cudaStreamCreate (&copy_stream));
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
	// calculate memory footprint depending on number of simulations
	dep_mem=options->num_membranes*options->num_environments*sizeof(CHARGE); //membrane charge
	dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(MULTIPLICITY); //multiset
	dep_mem+=(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sizeof(MULTIPLICITY);//nb (block applications)
	dep_mem+=((structures->pi_rule_size)*options->num_environments+structures->env_rule_size)*sizeof(MULTIPLICITY);//nr (rule applications)
	if (accurate) dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(uint); // numerators
	else dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(float); //addition
	dep_mem+=asize*options->num_environments*sizeof(ABV_T); // ABV activations
	dep_mem+=(1+options->num_membranes*options->num_environments)*sizeof(uint); //data error
	dep_mem+=curng_sizeof_state(CU_THREADS*options->num_environments); //random data

	dep_mem+=options->num_membranes*options->num_environments*sizeof(CHARGE); // membrane for async copy
	dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(MULTIPLICITY); //multiset for async copy
	dep_mem+=options->objects_to_output*sizeof(MULTIPLICITY); // filtered multiset
	if(!options->micro){
		dep_mem+=options->num_environments*sizeof(uint);//d_sizes
		dep_mem+=options->num_environments*besize*sizeof(uint);//d_active_blocks
		dep_mem+=(options->num_environments*structures->pi_rule_size)*sizeof(uint);// d_active_rules
	}
	// TODO Add here new data structures depending on the number of simulations

	// calculate memory footprint not depending on number of simulations
	unsigned int non_dep_mem=0;
	if (accurate) non_dep_mem += options->num_objects*options->num_membranes*options->num_environments*sizeof(uint)*2*options->modules;

	// According to required memory for simulating the PDP system, and the memory available in the GPU, calculate how many simulations can be launched in parallel
	sim_parallel=gsl_min(options->num_simulations,(((unsigned int) max_memory_gpu*0.8)-options->mem-non_dep_mem)/dep_mem);
	options->num_parallel_simulations=sim_parallel;

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

	options->mem+=dep_mem*sim_parallel+non_dep_mem;
	/************************************/
	/* Initialization of GPU structures */
	/************************************/
	
	/* Create initial configuration */
	ini_cfg = structures->configuration;

	structures->configuration.membrane_size=options->num_membranes*options->num_environments*options->num_simulations;
	checkCudaErrors(cudaMallocHost((void**)&structures->configuration.membrane,structures->configuration.membrane_size*sizeof(CHARGE)));

	structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments*options->num_simulations;
	checkCudaErrors(cudaMallocHost((void**)&structures->configuration.multiset, structures->configuration.multiset_size*sizeof( MULTIPLICITY)));

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
	
	/* Initialize aux data on CPU only if necessary */
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
	addition_size=options->num_objects*options->num_membranes*options->num_environments*sim_parallel*options->modules;
	bool finished = false;
	
	while (!finished) {
		/* Use n/d notation for row additions */
		if (accurate) {
			/* Check overflows */
			finished=true;
			bool overflow=false;
			int mesize = options->modules*esize;

			ini_denominator = new uint[mesize];
			ini_numerator = new uint[mesize];
			denominator = ini_denominator;
			numerator = new uint [addition_size];
			addition = NULL;
			
			for (int i=0;i<mesize;i++) {
				ini_denominator[i]=1;
				ini_numerator[i]=0;
			}
			
			if (options->modular) { // when using modules, we will create numerators and denominators per module (so there is no need to substract from blocks not belonging to a module)
				for (int m=0;m<(options->modules)*2;m++) {
					uint start = 0;
					uint end = options->num_rule_blocks;
					module_start_end(start,end,m/2,(m%2)==0,options);

					if (start >= end) continue;

					//if (end == UINT_MAX) continue;

					uint * m_ini_denominator = ini_denominator + (esize*(m/2));
					uint * m_ini_numerator = ini_numerator + (esize*(m/2));

					for (uint block=start; block<end; block++) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							uint a=m_ini_denominator[membr*options->num_objects+obj];
							uint b=mult;

							int multiple1 = a % b;
							int multiple2=1;
							if (multiple1!=0)
								multiple2 = b % a;

							/* If a is multiple of b */
							if (multiple1==0) {
								overflow=safe_u_add(m_ini_numerator[membr*options->num_objects+obj],a/b);
							}
							/* If b is multiple of a */
							else if (multiple2==0) {
								overflow=safe_u_mul(m_ini_numerator[membr*options->num_objects+obj],b/a);
								overflow=overflow||safe_u_add(m_ini_numerator[membr*options->num_objects+obj],1);
								m_ini_denominator[membr*options->num_objects+obj]=b;
							}
							/* If they are no multiple */
							else {
								overflow=safe_u_mul(m_ini_numerator[membr*options->num_objects+obj],b);
								overflow=overflow||safe_u_add(m_ini_numerator[membr*options->num_objects+obj],a);
								overflow=overflow||safe_u_mul(m_ini_denominator[membr*options->num_objects+obj],b);
							}
							if (overflow) break;
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
			} else {
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
		cout << "=> Memory used by the virtual table algorithm on the GPU:" << endl;

		if (accurate) {
			print_memory_bytes_human("\tDenominator+Ini_numerator: ",  esize*sizeof(uint)*2);
			print_memory_bytes_human("\tNumerator: ",  addition_size*sizeof(uint));
		} else
			print_memory_bytes_human("\tAddition: " ,  addition_size*sizeof(float));

		print_memory_bytes_human("\tNb: ", d_structures->nb_size*sizeof(MULTIPLICITY));
		print_memory_bytes_human("\tNr: ", d_structures->nr_size*sizeof(MULTIPLICITY));
		print_memory_bytes_human("\tABV: ",abv_size*sizeof(ABV_T));
		print_memory_bytes_human("\tRNG: ",curng_sizeof_state(CU_THREADS*options->num_environments*sim_parallel));
		print_memory_bytes_human("\tErrors: ",data_error_size*sizeof(uint));
		print_memory_bytes_human("\tMembrane charges (+async): ",d_structures->configuration.membrane_size*sizeof(CHARGE)*2);
		print_memory_bytes_human("\tMultisets (+async): ",d_structures->configuration.multiset_size*sizeof(MULTIPLICITY)*2);

		cout << "\t P system info (Again):" << endl;
		print_memory_bytes_human("\tRule block info: ",(d_structures->ruleblock_size+1)*(sizeof(LHS_IDX)+sizeof(RULE_IDX)+sizeof(MEMBRANE)));
		print_memory_bytes_human("\tLHS: ",d_structures->lhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY)+sizeof(INV_MULTIPLICITY)));
		print_memory_bytes_human("\tRHS: ",d_structures->rhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY)));
		print_memory_bytes_human("\tRule: ",(d_structures->pi_rule_size+d_structures->env_rule_size+1)*sizeof(RHS_IDX));
		print_memory_bytes_human("\tProbability: ",d_structures->probability_size*sizeof(PROBABILITY));
		
		print_memory_bytes_human("\tTOTAL: ",options->mem);
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

	//Allocate Aux Configuration for async copy
	checkCudaErrors(cudaMalloc((void**)&(d_configuration.multiset), d_structures->configuration.multiset_size*sizeof(MULTIPLICITY)));
	checkCudaErrors(cudaMalloc((void**)&(d_configuration.membrane), d_structures->configuration.membrane_size*sizeof(CHARGE)));

	//Allocate filter if any

	if(options->output_filter!=NULL){
		options->GPU_filter=true;
		checkCudaErrors(cudaMalloc((void**)&d_output_filter,options->objects_to_output*sizeof(unsigned int)));

		//Allocate compact multisets
		checkCudaErrors(cudaMalloc((void**)&d_output_multiset,sim_parallel*options->objects_to_output*sizeof(MULTIPLICITY)));
		checkCudaErrors(cudaMallocHost((void**)&output_multiset,sim_parallel*options->objects_to_output*sizeof(MULTIPLICITY)));
	}

	if(!options->micro){
		checkCudaErrors(cudaMalloc(&d_sizes,sim_parallel*options->num_environments*sizeof(uint)));
		checkCudaErrors(cudaMalloc(&d_active_blocks,sim_parallel*options->num_environments*besize*sizeof(uint)));
		checkCudaErrors(cudaMalloc(&d_active_rules,sim_parallel*(options->num_environments*structures->pi_rule_size)*sizeof(uint)));
	}
	// Allocate Additions
	if (!accurate)
		checkCudaErrors(cudaMalloc((void**)&d_addition,addition_size*sizeof(float)));
	else {
		checkCudaErrors(cudaMalloc((void**)&d_denominator,options->modules*esize*sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_ini_numerator,options->modules*esize*sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_numerator,addition_size*sizeof(uint)));
	}

	// Allocate ABV
	checkCudaErrors(cudaMalloc((void**)&d_abv,abv_size*sizeof(ABV_T)));

	// Allocate Errors
	checkCudaErrors(cudaMalloc((void**)&d_data_error,data_error_size*sizeof(uint)));

	// Allocate RNG states
	//Now the kernel is launched in a stream, so it can execute while the rest of structures are copied to memory
	//We must cudaStreamSynchronize after all the memory is set
	curng_binomial_init(dim3(options->num_environments,options->num_parallel_simulations),CU_THREADS,execution_stream,options->fast);

	/* Copies */
	//Now they are async with curng_init!!!

	//If micro-DCBA, make partition
	if(options->micro){
		int* partition=new int[options->num_rule_blocks];
		int* trans_partition=new int[options->num_rule_blocks];
		int* alphabet=new int[options->num_objects*options->num_membranes];

		competition::reset_partition(partition,
				alphabet,
				options->num_rule_blocks,
				options->num_objects*options->num_membranes);

		competition::make_partition_2(partition,
					structures->ruleblock.lhs_idx,
					structures->lhs.object,
					alphabet,
					options->num_rule_blocks,
					options->num_objects,
					options->num_membranes,
					structures->lhs.mmultiplicity,
					structures->lhs_size);
		//Counts the number of different competition blocks
		options-> num_partitions=competition::normalize_partition(partition,trans_partition,options->num_rule_blocks);

		if(options->num_partitions==1){
			cout << "Full competition, micro-DCBA may not improve performance..." << endl;
		}

		options->independent_ruleblocks=competition::initialize_partition_structures(trans_partition,
				options->num_partitions,options->num_rule_blocks,
				&accum_offset,&part_indexes,&ordered_rules,&compacted_blocks,&large_blocks,
				CU_THREADS
				);
		competition::reorder_ruleblocks(structures,ordered_rules,options);

		options->num_partitions-=options->independent_ruleblocks;
		//Ruleblocks that competes with other ruleblocks
		//int dependent_ruleblocks=options->num_rule_blocks-options->independent_ruleblocks;
		//checkCudaErrors(cudaMalloc((void**)&d_partition,dependent_ruleblocks*sizeof(uint)));
		//checkCudaErrors(cudaMemcpyAsync(d_partition, ordered_rules, dependent_ruleblocks*sizeof(uint), cudaMemcpyHostToDevice,copy_stream));
		for (int i = 0; i < NUM_STREAMS; ++i) { cudaStreamCreate(&streams[i]); }

		delete [] partition;
		delete [] trans_partition;
		delete [] alphabet;

		if (options->verbose>1)
			cout << "Using micro-DCBA, so ruleblocks are partitioned:" << endl <<
				"=> Number of partitions: " << options->num_partitions << endl <<
				"=> Number of independent ruleblocks (unit partitions): " << options->independent_ruleblocks << endl <<
                "=> Total partitions: " << options->independent_ruleblocks+options->num_partitions << endl <<
                "=> Ratio ruleblocks / partitions: " << (options->num_rule_blocks+options->num_blocks_env)/(options->num_partitions) << endl << endl;
	}
	else if (options->modular) {
		for (int i = 0; i < NUM_STREAMS; ++i) { cudaStreamCreate(&streams[i]); }
	}


	// Set ABV
	checkCudaErrors(cudaMemsetAsync(d_abv,0xFF,abv_size*sizeof(ABV_T),copy_stream));

	// Set Errors
	checkCudaErrors(cudaMemsetAsync(d_data_error,0,data_error_size*sizeof(uint),copy_stream));

	//Copy filter filter
	if(options->GPU_filter){
		checkCudaErrors(cudaMemcpyAsync(d_output_filter, options->output_filter,options->objects_to_output*sizeof(unsigned int), cudaMemcpyHostToDevice,copy_stream));
	}
	

	// Copy Ruleblock
	checkCudaErrors(cudaMemcpyAsync(d_structures->ruleblock.lhs_idx, structures->ruleblock.lhs_idx, (d_structures->ruleblock_size+1)*sizeof(LHS_IDX), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->ruleblock.rule_idx, structures->ruleblock.rule_idx, (d_structures->ruleblock_size+1)*sizeof(RULE_IDX), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->ruleblock.membrane, structures->ruleblock.membrane, d_structures->ruleblock_size*sizeof(MEMBRANE), cudaMemcpyHostToDevice,copy_stream));

	// Copy LHS
	checkCudaErrors(cudaMemcpyAsync(d_structures->lhs.object, structures->lhs.object, d_structures->lhs_size*sizeof(OBJECT), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->lhs.mmultiplicity, structures->lhs.mmultiplicity, d_structures->lhs_size*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->lhs.imultiplicity, structures->lhs.imultiplicity, d_structures->lhs_size*sizeof(INV_MULTIPLICITY), cudaMemcpyHostToDevice,copy_stream));

	// Copy RHS
	checkCudaErrors(cudaMemcpyAsync(d_structures->rhs.object, structures->rhs.object, d_structures->rhs_size*sizeof(OBJECT), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->rhs.mmultiplicity, structures->rhs.mmultiplicity, d_structures->rhs_size*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice,copy_stream));

	// Copy Rule
	checkCudaErrors(cudaMemcpyAsync(d_structures->rule.rhs_idx, structures->rule.rhs_idx, (d_structures->pi_rule_size+d_structures->env_rule_size+1)*sizeof(RHS_IDX), cudaMemcpyHostToDevice,copy_stream));

	// Copy Probability
	checkCudaErrors(cudaMemcpyAsync(d_structures->probability, structures->probability, d_structures->probability_size*sizeof(PROBABILITY), cudaMemcpyHostToDevice,copy_stream));

	// Copy Additions
	if (accurate) {
		checkCudaErrors(cudaMemcpyAsync(d_denominator, ini_denominator, options->modules*esize*sizeof(uint), cudaMemcpyHostToDevice,copy_stream));
		checkCudaErrors(cudaMemcpyAsync(d_ini_numerator, ini_numerator, options->modules*esize*sizeof(uint), cudaMemcpyHostToDevice,copy_stream));
	}	



	//Using constant memory to load as symbols results in no real gain (nor loss)
	checkCudaErrors(cudaMemcpyToSymbolAsync(d_options, options, sizeof(_options),size_t(0),cudaMemcpyHostToDevice,copy_stream));

	_computations* computations;
	computations=new _computations;
	computations->besize=options->num_blocks_env+options->num_rule_blocks;
	computations->esize=options->num_objects*options->num_membranes;
	computations->msize=options->num_objects;
	computations->asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	computations->block_chunks=(besize + CU_THREADS -1)>>CU_LOG_THREADS;
	computations->rpsize=structures->pi_rule_size;
	computations->resize=structures->pi_rule_size+structures->env_rule_size;

	checkCudaErrors(cudaMemcpyToSymbolAsync(d_computations, computations, sizeof(_computations), size_t(0),cudaMemcpyHostToDevice,copy_stream));


	// Create a timer
	sdkCreateTimer(&counters.timer);

	//Final synchronize
	//cudaStreamSynchronize(execution_stream);


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
	checkCudaErrors(cudaFreeHost(aux.membrane));
	checkCudaErrors(cudaFreeHost(aux.multiset));

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

	checkCudaErrors(cudaFree(d_configuration.multiset));
	checkCudaErrors(cudaFree(d_configuration.membrane));


	//Deallocate filter if any
	if(options->output_filter!=NULL){
		checkCudaErrors(cudaFree(d_output_filter));
		checkCudaErrors(cudaFree(d_output_multiset));
		checkCudaErrors(cudaFreeHost(output_multiset));
	}

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
	

	//Deallocate partition for micro
	if(options->micro){
		//checkCudaErrors(cudaFree(d_partition));
		delete [] accum_offset;
		delete [] ordered_rules;
		delete [] part_indexes;

		for (int i = 0; i < NUM_STREAMS; ++i)
		{
			cudaStreamDestroy(streams[i]);
		}

	} else{
		checkCudaErrors(cudaFree(d_sizes));
		checkCudaErrors(cudaFree(d_active_blocks));
		checkCudaErrors(cudaFree(d_active_rules));

		if (options->modular) {
			for (int i = 0; i < NUM_STREAMS; ++i)
			{
				cudaStreamDestroy(streams[i]);
			}
		}
	}
	checkCudaErrors(cudaStreamDestroy(execution_stream));
	checkCudaErrors(cudaStreamDestroy(copy_stream));
	// Deallocate RNG states
	curng_binomial_free();	

	sdkDeleteTimer(&counters.timer);
	
	cudaDeviceReset(); // cudaThreadExit is deprecated
}

void Simulator_gpu_dir::reset(int sim_ini) {
	checkCudaErrors(cudaMemcpyAsync(d_structures->configuration.membrane, structures->configuration.membrane+sim_ini*options->num_environments*options->num_membranes, options->num_parallel_simulations*options->num_environments*options->num_membranes*sizeof(CHARGE), cudaMemcpyHostToDevice,copy_stream));
	checkCudaErrors(cudaMemcpyAsync(d_structures->configuration.multiset, structures->configuration.multiset+sim_ini*options->num_environments*esize, options->num_parallel_simulations*options->num_environments*esize*sizeof(MULTIPLICITY), cudaMemcpyHostToDevice,copy_stream));
	cudaStreamSynchronize(copy_stream);
}

void Simulator_gpu_dir::cleanup() {
	// not needed anymore, but kept just in case
	checkCudaErrors(cudaMemsetAsync(d_abv,0xFF,abv_size*sizeof(ABV_T),copy_stream));
	//checkCudaErrors(cudaMemsetAsync(d_data_error,0,data_error_size*sizeof(uint),copy_stream));
	//checkCudaErrors(cudaMemsetAsync((d_structures->nr),0, d_structures->nr_size*sizeof(MULTIPLICITY),copy_stream));
	//checkCudaErrors(cudaMemsetAsync((d_structures->nb),0, d_structures->nb_size*sizeof(MULTIPLICITY),copy_stream));
	cudaStreamSynchronize(copy_stream);
}

__global__ void kernel_output_filter(MULTIPLICITY* d_output_multiset,
									MULTIPLICITY *src_multiset,
									unsigned int *d_output_filter,
									int max_objects,
									int sim_size){
	//Calculate id
	uint tidx=threadIdx.x+blockIdx.x*blockDim.x;

	//Only write if we are not out of bounds
	if(tidx<max_objects){
		//Thread tidx will write to position tidx
		//The object_id to be written at position tidx is stored in d_output_filter
		uint obj_id=d_output_filter[tidx];

		//Get object from proper position taking offset into account
		d_output_multiset[max_objects*blockIdx.y+tidx]=src_multiset[sim_size*blockIdx.y+obj_id];
	}

}

void Simulator_gpu_dir::retrieve_copy() {
	if(options->GPU_filter){
		uint cu_threads=CU_THREADS;
		uint cu_blocksx=options->objects_to_output/cu_threads;
		if(cu_blocksx==0){
			//Less objects than max threads per block
			//use one block and one thread per block
			cu_threads=options->objects_to_output;
			cu_blocksx=1;
		}else if(options->objects_to_output%cu_threads!=0){
			//there are some objects that do not fill into a block
			//Use extra block and keep track of position
			cu_blocksx++;
		}
		uint cu_blocksy=options->num_parallel_simulations;

		kernel_output_filter<<<dim3(cu_blocksx,cu_blocksy),
									cu_threads,0,execution_stream>>>
									(d_output_multiset,
									d_structures->configuration.multiset,
									d_output_filter,
									options->objects_to_output,
									options->num_environments*esize);
	}
	//getLastCudaError("Error copying filtered output device to device");
	else{
		checkCudaErrors(cudaMemcpyAsync(d_configuration.membrane, d_structures->configuration.membrane,d_structures->configuration.membrane_size*sizeof(CHARGE), cudaMemcpyDeviceToDevice,execution_stream));
		checkCudaErrors(cudaMemcpyAsync(d_configuration.multiset, d_structures->configuration.multiset,d_structures->configuration.multiset_size*sizeof(MULTIPLICITY), cudaMemcpyDeviceToDevice,execution_stream));
	}
}
void Simulator_gpu_dir::retrieve_async(int sim_ini) {
	if(options->GPU_filter){
		checkCudaErrors(cudaMemcpyAsync(output_multiset, d_output_multiset, options->num_parallel_simulations*options->objects_to_output*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost,copy_stream));
	}
	else
	{
		checkCudaErrors(cudaMemcpyAsync(structures->configuration.membrane+sim_ini*options->num_environments*options->num_membranes, d_configuration.membrane, options->num_parallel_simulations*options->num_environments*options->num_membranes*sizeof(CHARGE), cudaMemcpyDeviceToHost,copy_stream));
		checkCudaErrors(cudaMemcpyAsync(structures->configuration.multiset+sim_ini*options->num_environments*esize, d_configuration.multiset, options->num_parallel_simulations*options->num_environments*esize*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost,copy_stream));
	}
}
//Deprecated
void Simulator_gpu_dir::retrieve(int sim_ini) {

	checkCudaErrors(cudaMemcpy(structures->configuration.membrane+sim_ini*options->num_environments*options->num_membranes, d_structures->configuration.membrane, options->num_parallel_simulations*options->num_environments*options->num_membranes*sizeof(CHARGE), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(structures->configuration.multiset+sim_ini*options->num_environments*esize, d_structures->configuration.multiset, options->num_parallel_simulations*options->num_environments*esize*sizeof(MULTIPLICITY), cudaMemcpyDeviceToHost));
}

/***************************************************************************/
/***************/
/* MAIN PHASES */
/***************/

int Simulator_gpu_dir::selection(uint step=0){

	/* PHASE 1: DISTRIBUTION */
	if (!selection_phase1(step))
		return 1;

	/* PHASE 2: MAXIMALITY */
	if (!selection_phase2(step))
		return 2;

	/* PHASE 3: PROBABILITY */
	if (!selection_phase3(step))
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
/* Global phase for independent ruleblocks on the GPU */
/*****************************************/
__global__ void kernel_micro_dcba_independent(
			PDP_Psystem_REDIX::Rule rule,
			PDP_Psystem_REDIX::Ruleblock ruleblock,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Lhs lhs,
			PDP_Psystem_REDIX::Rhs rhs,
			PDP_Psystem_REDIX::Probability probability,
			uint * d_abv,
			PDP_Psystem_REDIX::NR nr,
		//	uint * d_data_error,
			uint part_init) {
	volatile struct _options options=d_options;
	volatile uint bdim=blockDim.x;
	volatile uint env=blockIdx.x;
	volatile uint sim=blockIdx.y;
	uint block=threadIdx.x;

	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint part_size=besize-part_init;

	uint resize=d_computations.resize;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint rpsize=d_computations.rpsize;
	uint asize=d_computations.asize;
	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	/* Minimum applications */
	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
        uint N=0;

		block=bchunk*blockDim.x+threadIdx.x+part_init;

		uint o_ini=0;
		uint o_end=0;
		uint membr=0;
		if (block < besize){
				o_ini=ruleblock.lhs_idx[block];
				o_end=ruleblock.lhs_idx[block+1];
				membr=ruleblock.membrane[block];
		}
		// If the block is active
		if ((block < besize) &&
				((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
									        >> ((~block)&ABV_DESPL_MASK))
											& 0x1)) {
			N=UINT_MAX;

			for (int o=o_ini; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				uint value = configuration.multiset[D_MU_IDX(obj,membr)]/mult;
				N=(value < N) ? value : N;
				if(N==0) break;
			}

		}
		uint rule_ini=ruleblock.rule_idx[block];
		uint rule_end=ruleblock.rule_idx[block+1];

		//In case its an env_rule, it should be in the proper environment
		N=N*(!IS_ENVIRONMENT(membr)||env==GET_ENVIRONMENT(membr));

		if (N==0) {
			if(block<besize)
				for (uint r = rule_ini; r < rule_end; r++) {
					if (block < options.num_rule_blocks)
						nr[D_NR_P_IDX(r)] = 0;
					else
						nr[D_NR_E_IDX(r)] = 0;
			}
		}
		else{
			/* Consume LHS */
			for (int o=o_ini; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				/* Delete block application and check errors */
				if (atomicSub(configuration.multiset+sim*options.num_environments*esize+env*esize+membr*msize+obj,N*mult)
					< N*mult)
//					if (!update_error) update_error=true;
//					block_upd_error = 1+block;
					printf("error: %u %u %u %u\n",sim,env,block,N);
			}

			//Calculate multinomial and apply to rules
			configuration.membrane[D_CH_IDX(GET_MEMBRANE(membr))]=GET_BETA(membr);

			float cr=0.0f,d=1.0f;
			uint r;
			float p;
			uint val=0;
			//Only n-1 rules, to avoid branching on last
			for (r = rule_ini; r < rule_end-1; r++) {
				val=0;

				if (IS_ENVIRONMENT(membr)) {
					p=probability[options.num_environments*rpsize+(r-rpsize)];
				}
				else {
					p=probability[env*rpsize+r];
				}

				cr = fdividef(p,d);

				if (cr > 0.0f) {
					val=curng_binomial_random (N, cr);
				}


				if (!IS_ENVIRONMENT(membr))
					nr[D_NR_P_IDX(r)] = val;
				else
					nr[D_NR_E_IDX(r)] = val;

				N-=val;
				d*=(1-cr);
			}

			//Last rule, to avoid one branch on the loop
			r=rule_end-1;
			val=0;

			if (IS_ENVIRONMENT(membr)) {
				p=probability[options.num_environments*rpsize+(r-rpsize)];
			}
			else {
				p=probability[env*rpsize+r];
			}

			cr = fdividef(p,d);

			if (cr > 0.0f) {
				val=N;
			}

			if (!IS_ENVIRONMENT(membr))
				nr[D_NR_P_IDX(r)] = val;
			else
				nr[D_NR_E_IDX(r)] = val;

		}
	}

}
/*****************************************/
/* Step 1 (filters) of phase1 on the GPU */
/*****************************************/
__global__ void kernel_phase1_filters(
			PDP_Psystem_REDIX::Ruleblock ruleblock,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Lhs lhs,
			PDP_Psystem_REDIX::NR nb,
		//	struct _options options,
			uint * d_abv,
			uint * d_data_error) {
	volatile _options options=d_options;
	volatile uint env=blockIdx.x;
	volatile uint sim=blockIdx.y;
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
				if (setch!=4 && setch!= chb){
				/*if (m_c_charges[am]==4) {
					m_c_charges[am]=chb;//GET_BETA(membr);
					//printf("S=%d,B=%d,T=%d: (am=%d,beta=%d) -> (charge=%d)\n",sim,block,threadIdx.x,am,chb,m_c_charges[am]);
				}
				else if (m_c_charges[am]!= chb) /*GET_BETA(membr)*/ {
					//printf("!!S=%d,B=%d,T=%d: (am=%d,beta=%d) -> (charge=%d)\n",sim,block,threadIdx.x,am,chb,m_c_charges[am]);
					m_c_conflicts[am] = block;
					c_conflict = true;}
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
		d_data_error[1+sim*options.num_environments*options.num_membranes
		             +env*options.num_membranes+threadIdx.x]=m_c_charges[threadIdx.x];
		d_data_error[1+gridDim.y*options.num_environments*options.num_membranes
		             +sim*options.num_environments*options.num_membranes
		             +env*options.num_membranes+threadIdx.x]=m_c_conflicts[threadIdx.x];

		if (threadIdx.x==0)// && d_data_error[0]!=CONSISTENCY_ERROR)
			d_data_error[0]=CONSISTENCY_ERROR;
	}
}

/*****************************************/
/* Step 1 (filters) of phase1 on the GPU */
/*****************************************/
__global__ void kernel_phase1_filters_v2(
			PDP_Psystem_REDIX::Ruleblock ruleblock,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Lhs lhs,
			PDP_Psystem_REDIX::NR nb,
		//	struct _options options,
			uint * d_abv,
			uint * d_data_error,
			uint part_init,
			uint part_end) {
	uint part_size=part_end-part_init;
	uint part_chunks=((part_size) + blockDim.x - 1)>>CU_LOG_THREADS;
	volatile _options options=d_options;
	volatile uint env=blockIdx.x;
	volatile uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	//uint block_chunks=(besize + blockDim.x -1)>>CU_LOG_THREADS;
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

	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
		block=bchunk*blockDim.x+threadIdx.x+part_init;

		if (threadIdx.x < (blockDim.x>>ABV_LOG_WORD_SIZE)) {
			s_abv[threadIdx.x]=ABV_INIT_WORD;
		}

		__syncthreads();

		if (block < part_end) {
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
		d_data_error[1+sim*options.num_environments*options.num_membranes
		             +env*options.num_membranes+threadIdx.x]=m_c_charges[threadIdx.x];
		d_data_error[1+gridDim.y*options.num_environments*options.num_membranes
		             +sim*options.num_environments*options.num_membranes
		             +env*options.num_membranes+threadIdx.x]=m_c_conflicts[threadIdx.x];

		if (threadIdx.x==0)// && d_data_error[0]!=CONSISTENCY_ERROR)
			d_data_error[0]=CONSISTENCY_ERROR;
	}
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

/************************************************************/
/* Step 2 (normalization and minimums) of phase1 on the GPU */
/* This non-accurate version causes updating errors
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
		if((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
			        >> ((~threadIdx.x)&ABV_DESPL_MASK))
					& 0x1) {
//		if (d_is_active(threadIdx.x,s_abv)) {
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
		if((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
			        >> ((~threadIdx.x)&ABV_DESPL_MASK))
					& 0x1) {
//		if (d_is_active(threadIdx.x,s_abv)) {
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


		//
		// We start by having the total sum and inactive blocks substract their multiplicities
		if ((block < besize) &&
				!((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
					        >> ((~threadIdx.x)&ABV_DESPL_MASK))
							& 0x1)) {
//				!d_is_active(threadIdx.x,s_abv)) {
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
		if ((block < besize) &&
				((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
				        >> ((~threadIdx.x)&ABV_DESPL_MASK))
						& 0x1)) {
//				d_is_active(threadIdx.x,s_abv)) {
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

__global__ void kernel_phase1_object_numerator(uint * d_numerator,
			uint * d_ini_numerator,
			uint obj_chunks,
			bool only_not_null=false){
	// Initialize addition vector
	struct _options options=d_options;
	uint esize=d_computations.esize;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint msize=options.num_objects;

	for (int ochunk=0; ochunk < obj_chunks; ochunk++) {
		uint obj=blockDim.x*blockIdx.z+ochunk*blockDim.x*gridDim.z+threadIdx.x;
		if (only_not_null) obj= ochunk*blockDim.x+threadIdx.x;

		if (obj>=esize) break;

		if (only_not_null && d_ini_numerator[obj]==0) continue; // allows other modules to write their numerators
		d_numerator[D_AD_IDX(obj,0)]=d_ini_numerator[obj];
	}
}

__global__ void kernel_phase1_object_addition(float * d_addition,
			uint obj_chunks){
	// Initialize addition vector
	struct _options options=d_options;
	uint esize=d_computations.esize;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint msize=options.num_objects;

	for (int ochunk=0; ochunk < obj_chunks; ochunk++) {
		uint obj= ochunk*blockDim.x+threadIdx.x;

		if (obj>=esize) break;
		d_addition[D_AD_IDX(obj,0)]=1.0f;
	}
}

template <bool accurate>
__global__ void kernel_phase1_normalization_v2 (
		PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nr,
		uint * d_denominator,
		uint * d_numerator,
		float * d_addition,
		uint * d_abv,
		uint part_init,
		uint part_end) {
	struct _options options=d_options;
	uint part_size=part_end-part_init;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint bdim=blockDim.x;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;

	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	/* Normalization - step 2 *
	 *  calculate the sum of objects in lhs */
	for (int bchunk=0; bchunk < part_chunks; bchunk++) {

		block=bchunk*blockDim.x+threadIdx.x+part_init;
		
		if (accurate) {
			// We start by having the total sum and inactive blocks substract their multiplicities
			if ((block < part_end) &&
					!((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
								>> ((~block)&ABV_DESPL_MASK))
								& 0x1)) {
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
		} else {
			if ((block < part_end) &&
					((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
								>> ((~block)&ABV_DESPL_MASK))
								& 0x1)) {
				uint o_init=ruleblock.lhs_idx[block];
				uint o_end=ruleblock.lhs_idx[block+1];
				for (int o=o_init; o < o_end; o++) {
					uint obj=lhs.object[o];
					uint membr=lhs.mmultiplicity[o];
					membr=GET_MEMBR(membr);

					// TODO: Check if using imultiplicity is more efficient
					float inv=lhs.imultiplicity[o];
					atomicAddf(d_addition+D_AD_IDX(obj,membr),inv);
				}
			}
		}
		//__syncthreads();
	}

	__syncthreads();

	/* Normalization - step 2 *
	 * Column minimum calculation */
	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
        uint min=0;

        block=bchunk*blockDim.x+threadIdx.x+part_init;

		// If the block is active
		if ((block < part_end) &&
				((d_abv[sim*options.num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
							>> ((~block)&ABV_DESPL_MASK))
							& 0x1)) {
			min=UINT_MAX;
            uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];

			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);
				uint value = min;

				if (accurate)
					value = (configuration.multiset[D_MU_IDX(obj,membr)] * d_denominator[membr*options.num_objects+obj]) / (mult*mult*d_numerator[D_AD_IDX(obj,membr)]);
				else
					value = configuration.multiset[D_MU_IDX(obj,membr)]/(mult*mult*(d_addition[D_AD_IDX(obj,membr)]-1.0f));

				min=(value < min) ? value : min;

				if(min==0) break;

				/*if (value==UINT_MAX)
					if (accurate) printf ("block %d, obj %d*%d in env %d, using %d/%d, having %u\n",block,obj,mult,env,d_numerator[D_AD_IDX(obj,membr)],d_denominator[membr*options.num_objects+obj],configuration.multiset[D_MU_IDX(obj,membr)]);
					else printf ("block %d, obj %d*%d in env %d, using %f, having %u\n",block,obj,mult,env,d_addition[D_AD_IDX(obj,membr)],configuration.multiset[D_MU_IDX(obj,membr)]);*/
			}
		}
		__syncthreads();

		if (block < part_end)
			nr[D_NB_IDX(block)]=min;
	}

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
	//__shared__ bool block_sel;
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
            //if (!block_sel) block_sel=true;
			
			/* Consume LHS */
            uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				//if (env==0 && threadIdx.x==0) printf("Updating %d*%d from  %d\n",bapp,mult,configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]);
				/* Delete block application and check errors */
				if (atomicSub(configuration.multiset+sim*options.num_environments*esize+env*esize+membr*msize+obj,bapp*mult)
					< bapp*mult) {
					if (!update_error) update_error=true;
					block_upd_error = 1+block;
					/* Pre-filter: only filter last rules consuming objects */
					//bapp*mult+mult) ;
					//d_deactivate(threadIdx.x,s_abv);
				   // if (threadIdx.x==0 && env==2) printf("Updating %u*%d from \n",bapp,mult);//,configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]);
				}
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
	//Changed: only save error if it was all ok until here (otherwise we would be overwriting, for example, CONSISTENCY_ERROR)
	if (threadIdx.x==0 && update_error && d_data_error[0]==0) {
		d_data_error[1+sim*options.num_environments*options.num_membranes+env*options.num_membranes]=block_upd_error;
		d_data_error[0]=UPDATING_CONFIGURATION_ERROR;
	}
	
}

__global__ void kernel_phase1_update_v2(
		PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		uint * d_abv,
		uint * d_data_error,
		uint part_init,
		uint part_end) {
	struct _options options=d_options;
	uint part_size=part_end-part_init;
	bool update_error=false;
	uint block_upd_error=0;
	uint bdim=blockDim.x;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;
	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	/* Deleting LHS *
	 * Adding block applications */
	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
	
		block=bchunk*blockDim.x+threadIdx.x+part_init;
		if (block >= part_end) break;

		uint bapp=nr[D_NB_IDX(block)];
		nr[D_NB_IDX(block)]=0; // TODO: avoid this line. It is required for modular version.

		if (bapp>0) {

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
					< bapp*mult) {
					if (!update_error)  update_error=true;
					block_upd_error = 1+block;
					//printf("env %d, block %u (%u-%u), caused error, obj %u, mult %u, app %u, there is %u\n",env,block,part_init,part_end,obj,mult,bapp, configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]);
				}
			}

			/* Add applications to block */
			nb[D_NB_IDX(block)]+=bapp;
		}
	}
	__syncthreads();

	/** Filter 2 **/
	//After consuming objects, check which can still be applied, for phase2 to be filtered
	for (int bchunk=0; bchunk < part_chunks; bchunk++) {

		block=bchunk*blockDim.x+threadIdx.x+part_init;
		if (block >= part_end) break;


		if ((d_abv[sim*options.num_environments*asize+
					 env*asize+
					 (block>>ABV_LOG_WORD_SIZE)]
	               >> ((~block)&ABV_DESPL_MASK))
	               & 0x1) {

			// Using new registers avoid memory accesses on the for loop
			uint o_init=ruleblock.lhs_idx[block];
			uint o_end=ruleblock.lhs_idx[block+1];
			for (int o=o_init; o < o_end; o++) {
				uint obj=lhs.object[o];
				uint membr=lhs.mmultiplicity[o];
				uint mult=GET_MULTIPLICITY(membr);
				membr=GET_MEMBR(membr);

				// Check if we have enough objects to apply the block, otherwise deactivate it
				if (configuration.multiset[sim*options.num_environments*esize+env*esize+membr*msize+obj]<mult) {
					atomicAnd((d_abv+sim*options.num_environments*asize+
									 env*asize+
									 (block>>ABV_LOG_WORD_SIZE)), ~(0x1<<((~block)&ABV_DESPL_MASK)));
					break;
				}
			}
		}
	}
	//Changed: only save error if it was all ok until here (otherwise we would be overwriting, for example, CONSISTENCY_ERROR)
	if ( update_error && d_data_error[0]==0) {
		d_data_error[1+sim*options.num_environments*options.num_membranes+env*options.num_membranes]=block_upd_error;
		d_data_error[0]=UPDATING_CONFIGURATION_ERROR;
	}

}


/************************************************/
/* Implementation of Phase 1 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase1(uint step=0) {

    pdp_out->print_dcba_phase(1);

    pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 1");

	/* Create and start timers */
	if (runcomp) {
		counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0;
		counters.timek1cpu = counters.timek2cpu = counters.timek3cpu = 10;
	} else if (profiling) {
		;//counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0;
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
	if (runcomp || profiling) {
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for filters");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}

	if (options->modular) {
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// TODO!!!!!!!!!!! Is it better than just running kernel phase 1??? which datastructures???

   		//cleanup(); // we need to explicitly clean up auxiliary data structures when using modules: rule selection, etc.
		int stepc =step % options->cycles;
		int stream_to_go=0;
   		for (int m=0;m<options->modules*2;m++) {
			bool module_active = stepc >= options->modules_start[m/2] && stepc < options->modules_end[m/2];
			if (!module_active) continue;

			uint start, end;
			module_start_end(start,end,m/2,(m%2)==0,options);

			if (start < end) {
				kernel_phase1_filters_v2 <<<dimGrid,dimBlock,sh_mem,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nb,
							d_abv, d_data_error,start,end);

				stream_to_go=(stream_to_go+1)%NUM_STREAMS;
			}
		}
		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}
		getLastCudaError("kernel for modular filters phase 1 launch failure");
	}
	else {
		kernel_phase1_filters <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->lhs, d_structures->nb,
			d_abv, d_data_error);

		cudaStreamSynchronize(execution_stream);
		getLastCudaError("kernel for filters phase 1 launch failure");
	}

	if (runcomp||profiling) {
		sdkStopTimer(&counters.timer);
		counters.timek1gpu+=sdkGetTimerValue(&counters.timer);
		pdp_out->print_profiling_dcba_microphase_result(counters.timek1gpu);
	}

	for (int a=0; a<options->accuracy; a++) {
		/* Apply kernel for normalization */

		sh_mem=(cu_threads>>ABV_LOG_WORD_SIZE)*sizeof(uint);

		if (runcomp||profiling) {
			pdp_out->print_profiling_dcba_microphase_name("Launching kernel for normalization");
			sdkResetTimer(&counters.timer);
			sdkStartTimer(&counters.timer);
		}
		if (options->modular) {
			int stepc =step % options->cycles;
			int stream_to_go=0;

			if (!accurate) { // once for all modules
				kernel_phase1_object_addition<<<dimGrid,dimBlock,0,streams[stream_to_go]>>>(d_addition,obj_chunks);
				cudaStreamSynchronize(streams[stream_to_go]);
			}

			for (int m=0;m<options->modules*2;m++) {
				bool module_active = stepc >= options->modules_start[m/2] && stepc < options->modules_end[m/2];
				if (!module_active) continue;

				uint start, end;
				module_start_end(start,end,m/2,(m%2)==0,options);

				if (start >= end) continue;
				//cout << "step " << step << ", module " << (m/2) << ", running " << ((m%2==0)?"Pi":"Env") << ". Start=" << start << ",End=" << end << "." << endl;
				if (accurate) {
					// select the corresponding ini_numerator and ini_denominator
					uint* d_module_numerator   = d_ini_numerator + (m/2)*esize;
					uint* d_module_denominator = d_denominator + (m/2)*esize;

					if ((m%2) == 0) { // only once, right before launching Pi blocks. Requires to synchronize because env blocks also need this.
						// Required to reset addition vector to their initial state. Every module should load its numerators
						kernel_phase1_object_numerator<<<dimGrid,dimBlock,0,streams[stream_to_go]>>>( d_numerator,
										d_module_numerator,
										obj_chunks, true);
						cudaStreamSynchronize(streams[stream_to_go]);//required
					}

					kernel_phase1_normalization_v2 <true> <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
								d_structures->configuration, d_structures->lhs, d_structures->nr,
								d_module_denominator,d_numerator,d_addition,d_abv,
								start,
								end);
				}
				else {
					kernel_phase1_normalization_v2 <false> <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
								d_structures->configuration, d_structures->lhs, d_structures->nr,
								d_denominator,d_numerator,d_addition,d_abv,
								start,end);
					//cudaStreamSynchronize(streams[stream_to_go]);
				}

				kernel_phase1_update_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nb,
							d_structures->nr, d_abv, d_data_error,
							start,
							end);
				//cudaStreamSynchronize(streams[stream_to_go]);

				stream_to_go=(stream_to_go+1)%NUM_STREAMS;
			//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
			}

			for(int i=0;i<NUM_STREAMS;i++){
				cudaStreamSynchronize(streams[i]);
			}
			getLastCudaError("kernel for phase 1 modular launch failure");
			if (runcomp||profiling) {
				sdkStopTimer(&counters.timer);
				counters.timek2gpu+=sdkGetTimerValue(&counters.timer);
				pdp_out->print_profiling_dcba_microphase_result(counters.timek2gpu);
			}
		}
		else if(options->micro){
			//Trying to reduce kernel loop by using blocks in z component
			obj_chunks=(esize + cu_threads -1)/(cu_threads);
			if(obj_chunks>=32){
				dimGrid.z=32;
				obj_chunks=(obj_chunks+dimGrid.z-1)/dimGrid.z;
			}

			//Reset addition vector to their initial state
			kernel_phase1_object_numerator<<<dimGrid,dimBlock,0,execution_stream>>>( d_numerator,
							 d_ini_numerator,
							obj_chunks);
			dimGrid.z=1;
			cudaStreamSynchronize(execution_stream);
			getLastCudaError("pre kernel for phase 1 micro launch failure");

			int stream_to_go=0;
			for(int i=0;i<large_blocks;i++){
				//Large blocks separated
				uint start=accum_offset[i];
				uint end=accum_offset[i+1];

				kernel_phase1_normalization_v2<true> <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nr,
							d_denominator,d_numerator,d_addition,d_abv,
							start,
							end);

				kernel_phase1_update_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nb,
							d_structures->nr, d_abv, d_data_error,
							start,
							end);

				stream_to_go=(stream_to_go+1)%NUM_STREAMS;
			//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
			}
			for(int i=0;i<compacted_blocks;i++){
				//Small blocks in chunks
				uint start=accum_offset[part_indexes[i]];
				uint end=accum_offset[part_indexes[i+1]];
				kernel_phase1_normalization_v2<true> <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nr,
							d_denominator,d_numerator,d_addition,d_abv,
							start,
							end);

				kernel_phase1_update_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nb,
							d_structures->nr, d_abv, d_data_error,
							start,
							end);
				stream_to_go++;
				if(stream_to_go==NUM_STREAMS)
					stream_to_go=0;
			//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
			}
			//Independent ruleblocks
//			uint start=accum_offset[options->num_partitions];
//			uint end=start+options->independent_ruleblocks;
//			kernel_phase1_normalization_acu_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
//						d_structures->configuration, d_structures->lhs, d_structures->nr,
//						d_denominator,d_numerator,d_abv,
//						start,
//						end+options->num_blocks_env);
//
//			kernel_phase1_update_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
//						d_structures->configuration, d_structures->lhs, d_structures->nb,
//						d_structures->nr, d_abv, d_data_error,
//						start,
//						end+options->num_blocks_env);

			//std::cout<< "from rules "<< start<<" to "<<end<<std::endl;

			for(int i=0;i<NUM_STREAMS;i++){
				cudaStreamSynchronize(streams[i]);
			}
			getLastCudaError("kernel for phase 1 micro launch failure");
			if (runcomp) {
				sdkStopTimer(&counters.timer);
				counters.timek2gpu+=sdkGetTimerValue(&counters.timer);
				pdp_out->print_profiling_dcba_microphase_result(counters.timek2gpu);
			}

		} else {

			if (! accurate)
			kernel_phase1_normalization <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
				d_structures->configuration, d_structures->lhs, d_structures->nr,
				*options,d_addition,d_abv,obj_chunks);
			else
			kernel_phase1_normalization_acu <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
				d_structures->configuration, d_structures->lhs, d_structures->nr,
				*options,d_denominator,d_numerator,d_ini_numerator,d_abv,obj_chunks);

			cudaStreamSynchronize(execution_stream);

			getLastCudaError("kernel for normalization phase 1 launch failure");

			if (runcomp||profiling) {
				cudaStreamSynchronize(execution_stream);
				getLastCudaError("kernel for phase 1 (normalization) launch failure");

				sdkStopTimer(&counters.timer);
				counters.timek2gpu+=sdkGetTimerValue(&counters.timer);
				pdp_out->print_profiling_dcba_microphase_result(counters.timek2gpu);
			}

			/* Apply kernel for update and filter 2 */

			if (runcomp||profiling) {
				pdp_out->print_profiling_dcba_microphase_name("Launching kernel for updating");
				sdkResetTimer(&counters.timer);
				sdkStartTimer(&counters.timer);
			}

			kernel_phase1_update <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
				d_structures->configuration, d_structures->lhs, d_structures->nb,
				d_structures->nr, *options, d_abv, d_data_error);

			cudaStreamSynchronize(execution_stream);

			getLastCudaError("kernel for update phase 1 launch failure");

			if (runcomp||profiling) {
				cudaStreamSynchronize(execution_stream);
				getLastCudaError("kernel for phase 1 (update) launch failure");

				sdkStopTimer(&counters.timer);
				counters.timek3gpu+=sdkGetTimerValue(&counters.timer);
				pdp_out->print_profiling_dcba_microphase_result(counters.timek3gpu);
			}
		}
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
		gold_selection_phase1_acu(step);
	
		/* Retrieving times */
		counters.timesp1gpu= counters.timek1gpu+counters.timek2gpu+counters.timek3gpu;
	} else if (profiling)
		counters.timesp1gpu+= counters.timek1gpu+counters.timek2gpu+counters.timek3gpu;
	
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

		cout << "Checking ABV: there are " << count_errors << " differences." << endl;

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
		
		
		
		// TODO: First solution (following KISS methodology): 
		//       ThreadIdx.x==0 will update everything, and rest of threads will
		//       compute next s_blocks
	}
}
__global__ void kernel_phase2_v2(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::Lhs lhs,
		PDP_Psystem_REDIX::NR nb,
	//	PDP_Psystem_REDIX::NR nr,
	//	struct _options options,
		uint * d_abv,
		int part_init,
		int part_end) {
	struct _options options=d_options;
	extern __shared__ uint sData[];
	//Next b counts the number of blocks
	uint part_size=part_end-part_init;
	//BDim is num threads
	uint bdim = blockDim.x;
	//Activation bit vectors: useless because only accessed once
	//volatile uint * s_abv = sData;
	//Rule order
	uint * s_blocks = sData;
	//Active blocks per partition
	__shared__ uint s_next;

	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint block=threadIdx.x;

	//Num of ruleblocks and communication rules
	//At most, only num_rule_blocks
	uint besize=options.num_blocks_env+options.num_rule_blocks;
	//Environment size
	uint esize=options.num_objects*options.num_membranes;
	//Membrane size
	uint msize=options.num_objects;
	uint asize=(besize>>ABV_LOG_WORD_SIZE) + 1;

	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	if(threadIdx.x==0){
		s_next=0;
	}

	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
		__syncthreads();

		block=bchunk*bdim+threadIdx.x+part_init;

		//Custom activation index
		if (block < part_end &&
				(d_abv[sim*options.num_environments*asize+
											 env*asize+
											 (block>>ABV_LOG_WORD_SIZE)]
							               >> ((~block)&ABV_DESPL_MASK))
							        & 0x1) {
			s_blocks[atomicInc(&s_next,bdim+2)]=block;
		}
		__syncthreads();

		if(threadIdx.x==0){
			//1. iterate rules in random order as previously calculated
			//2. for each rule, calculate minimum applications
			//3. for each rule, update applications and configurations

			uint o_init,o_end;
			int available_rules=s_next;

			for(int i=0;i<available_rules;i++){
				uint apps=UINT_MAX;

				uint next_block=s_blocks[i];

				//Indexes and lhs lengths
				o_init=ruleblock.lhs_idx[next_block];
				o_end=ruleblock.lhs_idx[next_block+1];

				uint obj;
				uint membr;
				uint rule_mult;

				//Get minimum applications (cannot be done in parallel because it depends
				//on previous applications from the random order)
				for (int o=o_init; o < o_end; o++) {
					obj=lhs.object[o];
					membr=lhs.mmultiplicity[o];
					rule_mult = GET_MULTIPLICITY(membr);
					membr=GET_MEMBR(membr);

					uint conf_mult = configuration.multiset[D_MU_IDX(obj,membr)];//configuration.multiset[D_MU_IDX(GET_OBJECT(obj),0)];

					apps=min(apps,conf_mult/rule_mult);
				}

				//Update applications and configurations
				if(apps==0)continue;

				nb[D_NB_IDX(next_block)]+=apps;

				//printf("Rule %u Applications: %u\n",next_block,apps);
				for (int o=o_init; o < o_end; o++) {
					obj=lhs.object[o];
					membr=lhs.mmultiplicity[o];
					rule_mult = GET_MULTIPLICITY(membr);
					membr=GET_MEMBR(membr);

	//				Check if new multiplicity is valid (>0)
	//				If substracting an uint results in a bigger number, then it was negative
	//				if(configuration.multiset[D_MU_IDX(GET_OBJECT(obj),0)]
	//						  <configuration.multiset[D_MU_IDX(GET_OBJECT(obj),0)]-apps*rule_mult)
	//									printf("error on phase 2 micro-v2: rule %u \n",next_block);

					//configuration.multiset[D_MU_IDX(GET_OBJECT(obj),0)]-=apps*rule_mult;
					configuration.multiset[D_MU_IDX(obj,membr)]-=apps*rule_mult;
				}
			}
		s_next=0;
		}
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
	//	struct _options options,
		uint * d_abv) {
	_options options=d_options;
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

	}
}


/************************************************/
/* Implementation of Phase 2 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase2(uint step=0){
	
//	if (options->verbose>0)	cout << endl << "--------------------------" << endl <<	"Launching GPU code for phase 2" << endl;
	pdp_out->print_dcba_phase(2);

    pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 2");
	
	if (runcomp) {
		//counters.timer = 0;
		counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0.0f;
		counters.timek1cpu = counters.timek2cpu = counters.timek3cpu = 1.0f;
	}// else if (profiling)
		//counters.timek1gpu = counters.timek2gpu = counters.timek3gpu = 0.0f;
	
	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;

	/* Apply kernel for Phase 2 */
	if (runcomp||profiling) {
		//	if (options->verbose>0) cout << endl << "Launching kernel for phase 2: ";
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 2");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}
	if(options->modular){
		//TODO: sort this and preaccumulate partitions
		if (pdp_out->will_print_dcba_phase())
			cout << "(using modular DCBA kernel)"<<endl;

		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads);
		size_t sh_mem=(cu_threads)*sizeof(uint);

		cudaStreamSynchronize(execution_stream);
		getLastCudaError("pre kernel for phase 2 modular launch failure");

		int stream_to_go=0;
		//Trick:If a rule has no competition, then it must have been applied as many times as possible,
		//so there is no point in launching a kernel with it. Hence, no need to launch it for env rules

		int stepc =step % options->cycles;

		for (int m=0;m<options->modules;m++) {
			bool module_active = stepc >= options->modules_start[m] && stepc < options->modules_end[m];
			if (!module_active) continue;

			// for Pi rule blocks within active modules
			uint start, end;
			module_start_end(start,end,m,true,options);

			if (start < end) {
				kernel_phase2_v2 <<<dimGrid,dimBlock,sh_mem,streams[stream_to_go]>>> (d_structures->ruleblock,
											d_structures->configuration, d_structures->lhs, d_structures->nb,
											//d_structures->nr,
											d_abv,
											start,end);
				stream_to_go = (stream_to_go + 1) % NUM_STREAMS;
			}
		}

		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}
		getLastCudaError("post kernel for phase 2 modular launch failure");

	}
	else if(options->micro){
		//TODO: sort this and preaccumulate partitions
		if (pdp_out->will_print_dcba_phase())
			cout << "(using micro DCBA kernel)"<<endl;
	
		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads);
		size_t sh_mem=(cu_threads)*sizeof(uint);
	
		cudaStreamSynchronize(execution_stream);
		getLastCudaError("pre kernel for phase 2 micro launch failure");

		int stream_to_go=0;
		//Trick:If a rule has no competition, then it must have been applied as many times as possible,
		//so there is no point in launching a kernel with it

		for(int i=0;i<large_blocks;i++){
			//Large blocks separated
			uint start=accum_offset[i];
			uint end=accum_offset[i+1];

			kernel_phase2_v2 <<<dimGrid,dimBlock,sh_mem,streams[stream_to_go]>>> (d_structures->ruleblock,
							d_structures->configuration, d_structures->lhs, d_structures->nb,
							//d_structures->nr,
							d_abv,
							start,end);
			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;

		}
		for(int i=0;i<compacted_blocks;i++){
			//Small blocks in chunks
			uint start=accum_offset[part_indexes[i]];
			uint end=accum_offset[part_indexes[i+1]];
			kernel_phase2_v2 <<<dimGrid,dimBlock,sh_mem,streams[stream_to_go]>>> (d_structures->ruleblock,
										d_structures->configuration, d_structures->lhs, d_structures->nb,
										//d_structures->nr,
										d_abv,
										start,end);
			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;


		}

		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}
		getLastCudaError("kernel for phase 2 micro launch failure");

	}else{
		if (mode>=2) {
			if (pdp_out->will_print_dcba_phase())
				cout << "(using basic kernel)"<<endl;

			dim3 dimGrid (cu_blocksx, cu_blocksy);
			dim3 dimBlock (cu_threads+1);
			size_t sh_mem=((cu_threads>>ABV_LOG_WORD_SIZE) + 2*cu_threads)*sizeof(uint);

			kernel_phase2_generic <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
				d_structures->configuration, d_structures->lhs, d_structures->nb,
				d_structures->nr, *options, d_abv);
		}
		else if (mode<2) {
			if (pdp_out->will_print_dcba_phase())
				cout << "(using blhs kernel)"<<endl;

			dim3 dimGrid (cu_blocksx, cu_blocksy);
			dim3 dimBlock (cu_threads);
			size_t sh_mem=((cu_threads>>ABV_LOG_WORD_SIZE) + 2*cu_threads + options->max_lhs*cu_threads)*sizeof(uint);

			kernel_phase2_blhs <<<dimGrid,dimBlock,sh_mem,execution_stream>>> (d_structures->ruleblock,
				d_structures->configuration, d_structures->lhs, d_structures->nb,
				d_structures->nr, d_abv);
		}
		cudaStreamSynchronize(execution_stream);

		getLastCudaError("kernel for phase 2 launch failure");
	}
	if (runcomp||profiling) {
		sdkStopTimer(&counters.timer);
		counters.timesp2gpu+=sdkGetTimerValue(&counters.timer);
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
		gold_selection_phase2(step);
	
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
		PDP_Psystem_REDIX::Probability probability
		) {
	volatile uint env=blockIdx.x;
	volatile uint rpsize=d_computations.rpsize;
	volatile uint resize=d_computations.resize;
	volatile _options options=d_options;
	volatile uint sim=blockIdx.y;
	volatile uint block=threadIdx.x;
	volatile uint besize=d_computations.besize;//options.num_blocks_env+options.num_rule_blocks;
	volatile uint block_chunks=d_computations.block_chunks;//(besize + blockDim.x -1)>>CU_LOG_THREADS;

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
			//Alternative version along with the memset
		//if(N>0){
			// Update charges
			configuration.membrane[D_CH_IDX(GET_MEMBRANE(membr))]=GET_BETA(membr);
			
			float cr=0.0f,d=1.0f;
			uint r;
			float p;
			uint val=0;
			//Only n-1 rules, to avoid branching on last
			for (r = rule_ini; r < rule_end-1; r++) {
				val=0;
				if (IS_ENVIRONMENT(membr)) {
					p=probability[options.num_environments*rpsize+(r-rpsize)];
				}
				else {
					p=probability[env*rpsize+r];
				}

				cr = fdividef(p,d);
				
				if (cr > 0.0f) {
					val=curng_binomial_random (N, cr);
				}

				if (!IS_ENVIRONMENT(membr))
					nr[D_NR_P_IDX(r)] = val;
				else
					nr[D_NR_E_IDX(r)] = val;

				N-=val;
				d*=(1-cr);
			}

			//Last rule, to avoid one branch on the loop
			r=rule_end-1;
			val=0;
			if (IS_ENVIRONMENT(membr)) {
				p=probability[options.num_environments*rpsize+(r-rpsize)];
			}
			else {
				p=probability[env*rpsize+r];
			}

			cr = fdividef(p,d);

			if (cr > 0.0f) {
				val=N;
			}
			if (!IS_ENVIRONMENT(membr))
				nr[D_NR_P_IDX(r)] = val;
			else
				nr[D_NR_E_IDX(r)] = val;


		}
		__syncthreads();
	}
}

__global__ void kernel_phase3_rules(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		PDP_Psystem_REDIX::Probability probability,
		int part_init,
		int part_end
		) {
	volatile uint part_size=part_end-part_init;
	volatile uint env=blockIdx.x;
	volatile uint rpsize=d_computations.rpsize;
	volatile uint resize=d_computations.resize;
	volatile _options options=d_options;
	volatile uint sim=blockIdx.y;
	volatile uint block=threadIdx.x;
	volatile uint besize=d_computations.besize;
	volatile uint bdim = blockDim.x;
	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	for (int bchunk=0; bchunk < part_chunks; bchunk++) {
		block=bchunk*bdim+threadIdx.x+part_init;

		if (block >= part_end) break;

		int rule_ini=ruleblock.rule_idx[block];
		int rule_end=ruleblock.rule_idx[block+1];

		uint N=nb[D_NB_IDX(block)];
		uint membr=ruleblock.membrane[block];

		if (N==0) {
			for (uint r = rule_ini; r < rule_end; r++) {
					nr[D_NR_P_IDX(r)] = 0;
			}
		}
		else {
			//Alternative version along with the memset
		//if(N>0){
			// Update charges
			configuration.membrane[D_CH_IDX(GET_MEMBRANE(membr))]=GET_BETA(membr);

			float cr=0.0f,d=1.0f;
			uint r;
			float p;
			uint val=0;
			//Only n-1 rules, to avoid branching on last
			for (r = rule_ini; r < rule_end-1; r++) {
				val=0;

				p=probability[env*rpsize+r];


				cr = fdividef(p,d);

				if (cr > 0.0f) {
					val=curng_binomial_random (N, cr);
				}


				nr[D_NR_P_IDX(r)] = val;

				N-=val;
				d*=(1-cr);
			}

			//Last rule, to avoid one branch on the loop
			r=rule_end-1;
			val=0;

			p=probability[env*rpsize+r];

			cr = fdividef(p,d);

			if (cr > 0.0f) {
				val=N;
			}

			nr[D_NR_P_IDX(r)] = val;


		}
		__syncthreads();
	}
}

__global__ void kernel_phase3_v2(PDP_Psystem_REDIX::Ruleblock ruleblock,
		PDP_Psystem_REDIX::Configuration configuration,
		PDP_Psystem_REDIX::NR nb,
		PDP_Psystem_REDIX::NR nr,
		PDP_Psystem_REDIX::Probability probability,
		uint part_init,
		uint part_end
		) {
	volatile uint part_size=part_end-part_init;
	volatile uint env=blockIdx.x;
	volatile uint rpsize=d_computations.rpsize;
	volatile uint resize=d_computations.resize;
	volatile _options options=d_options;
	volatile uint sim=blockIdx.y;
	volatile uint block=threadIdx.x;
	volatile uint besize=d_computations.besize;//options.num_blocks_env+options.num_rule_blocks;
	volatile uint block_chunks=d_computations.block_chunks;//(besize + blockDim.x -1)>>CU_LOG_THREADS;
	volatile uint bdim = blockDim.x;
	uint part_chunks=((part_size) + bdim - 1)>>CU_LOG_THREADS;

	for (int bchunk=0; bchunk < block_chunks; bchunk++) {

		block=bchunk*bdim+threadIdx.x+part_init;

		if (block >= part_end) break;

		int rule_ini=ruleblock.rule_idx[block];
		int rule_end=ruleblock.rule_idx[block+1];

		uint N=0;//nb[D_NB_IDX(block)];
		uint membr=ruleblock.membrane[block];

		if (block<options.num_rule_blocks || env==GET_ENVIRONMENT(membr)) {
			N=nb[D_NB_IDX(block)];
		}

		if (N==0) {
			for (uint r = rule_ini; r < rule_end; r++) {
				if (block < options.num_rule_blocks)
					nr[D_NR_P_IDX(r)] = 0;
				else if (env==GET_ENVIRONMENT(membr))
					nr[D_NR_E_IDX(r)] = 0;
			}
		}
		else {
			//Alternative version along with the memset
		//if(N>0){
			// Update charges
			configuration.membrane[D_CH_IDX(GET_MEMBRANE(membr))]=GET_BETA(membr);

			float cr=0.0f,d=1.0f;
			uint r;
			float p;
			uint val=0;
			//Only n-1 rules, to avoid branching on last
			for (r = rule_ini; r < rule_end-1; r++) {
				val=0;
				if (IS_ENVIRONMENT(membr)) {
					p=probability[options.num_environments*rpsize+(r-rpsize)];
				}
				else {
					p=probability[env*rpsize+r];
				}

				cr = fdividef(p,d);

				if (cr > 0.0f) {
					val=curng_binomial_random (N, cr);
				}

				if (!IS_ENVIRONMENT(membr))
					nr[D_NR_P_IDX(r)] = val;
				else
					nr[D_NR_E_IDX(r)] = val;

				N-=val;
				d*=(1-cr);
			}

			//Last rule, to avoid one branch on the loop
			r=rule_end-1;
			val=0;
			if (IS_ENVIRONMENT(membr)) {
				p=probability[options.num_environments*rpsize+(r-rpsize)];
			}
			else {
				p=probability[env*rpsize+r];
			}

			cr = fdividef(p,d);

			if (cr > 0.0f) {
				val=N;
			}
			if (!IS_ENVIRONMENT(membr))
				nr[D_NR_P_IDX(r)] = val;
			else
				nr[D_NR_E_IDX(r)] = val;


		}
		__syncthreads();
	}
}




/************************************************/
/* Implementation of Phase 3 (calls to kernels) */
/************************************************/
bool Simulator_gpu_dir::selection_phase3(uint step=0) {

//	if (options->verbose>0) { cout << endl << "--------------------------" << endl; cout << "Launching GPU code for phase 3" << endl; }
	pdp_out->print_dcba_phase(3);

	pdp_out->print_profiling_dcba_phase("Launching GPU code for phase 3");

	/* USING GPU KERNELS */
	uint cu_threads=CU_THREADS;
	uint cu_blocksx=options->num_environments;
	uint cu_blocksy=options->num_parallel_simulations;

	/* Apply kernel for Phase 3 */

	if (runcomp||profiling) {
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 3");
//		if (options->verbose>0)	cout << endl << "Launching kernel for phase 3: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}

	dim3 dimGrid (cu_blocksx, cu_blocksy);
	dim3 dimBlock (cu_threads);

	if (options->modular) {
		int stream_to_go=0;
		int stepc =step % options->cycles;

		for (int m=0;m<options->modules*2;m++) {
			bool module_active = stepc >= options->modules_start[m/2] && stepc < options->modules_end[m/2];
			if (!module_active) continue;

			uint start, end;
			module_start_end(start,end,m/2,(m%2)==0,options);

			//cout << "step " << step << ", module " << m/2 << ". Start=" << start << ",End=" << end << "." << endl;

			if (start < end) {

				kernel_phase3_v2 <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
													d_structures->configuration, d_structures->nb,
													d_structures->nr, d_structures->probability,
													start, end);

				stream_to_go = (stream_to_go + 1) % NUM_STREAMS;
			}
		}

		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}

		//cudaStreamSynchronize(execution_stream);
		getLastCudaError("post kernel for phase 3 modular launch failure");
	}
	else if(options->micro){

		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads);
		int stream_to_go=0;
		for(int i=0;i<large_blocks;i++){
			//Large blocks separated
			uint start=accum_offset[i];
			uint end=accum_offset[i+1];


			kernel_phase3_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
													d_structures->configuration, d_structures->nb,
													d_structures->nr, d_structures->probability,
													start,
													end
													);

			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;
		//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
		}
		for(int i=0;i<compacted_blocks;i++){
			//Small blocks in chunks
			uint start=accum_offset[part_indexes[i]];
			uint end=accum_offset[part_indexes[i+1]];
			kernel_phase3_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
											d_structures->configuration, d_structures->nb,
											d_structures->nr, d_structures->probability,
											start,
											end
											);

			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;
		//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
		}
		//Independent ruleblocks
//		uint start=accum_offset[options->num_partitions];
//		uint end=start+options->independent_ruleblocks;
//		kernel_phase3_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->ruleblock,
//														d_structures->configuration, d_structures->nb,
//														d_structures->nr, d_structures->probability,
//														start,
//														end
//														);
//
//
//
//
//		kernel_phase3_env <<<dimGrid,dimBlock,0,execution_stream>>> (d_structures->ruleblock,
//				d_structures->configuration, d_structures->nb,
//				d_structures->nr, d_structures->probability
//				);
		uint start=accum_offset[options->num_partitions];
		kernel_micro_dcba_independent<<<dimGrid,dimBlock,0,execution_stream>>>(
				d_structures->rule,
				d_structures->ruleblock,
				d_structures->configuration,
				d_structures->lhs,
				d_structures->rhs,
				d_structures->probability,
				d_abv,
				d_structures->nr,
			//	d_data_error,
				start);

		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}

		cudaStreamSynchronize(execution_stream);
		getLastCudaError("pre kernel for phase 3 micro launch failure");
	}else{
		kernel_phase3 <<<dimGrid,dimBlock,0,execution_stream>>> (d_structures->ruleblock,
			d_structures->configuration, d_structures->nb,
			d_structures->nr, d_structures->probability	);

		cudaStreamSynchronize(execution_stream);

		getLastCudaError("kernel for phase 3 launch failure");
	}
	if (runcomp||profiling) {
		cudaStreamSynchronize(execution_stream);
		getLastCudaError("kernel for phase 3 launch failure");

		sdkStopTimer(&counters.timer);
		counters.timesp3gpu+=sdkGetTimerValue(&counters.timer);
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
					cout << endl
				}
	}*/

	
	/*******************************/
	/* REPRODUCING CODE ON THE CPU */
	/*******************************/
	if (runcomp)
		gold_selection_phase3(step);
	
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


}


__global__ void kernel_phase4_rules (PDP_Psystem_REDIX::Rule rule,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Rhs rhs,
			PDP_Psystem_REDIX::NR nr,
			int part_init,
			int part_end,
			bool modular=false) {
	_options options=d_options;
	uint partsize = part_end-part_init;
	uint resize=d_computations.resize;
	uint rpsize=(!modular)?partsize:d_computations.rpsize;
	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint r=threadIdx.x;
	uint esize=options.num_objects*options.num_membranes;
	uint msize=options.num_objects;
	uint rp_chunks=(partsize + blockDim.x -1)>>CU_LOG_THREADS;

	/* Rules of Pi, executed by each environment */
	for (int rchunk=0; rchunk < rp_chunks; rchunk++) {
		r=rchunk*blockDim.x+threadIdx.x+part_init;

		uint N=0;

		if (r < part_end)
			N=nr[D_NR_P_IDX(r)];

		if (N>0) {
			//printf("env %d, sim %d, rule %d, %d times\n",env,sim,r,N);
			nr[D_NR_P_IDX(r)]=0; // reset NR
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

}
__global__ void kernel_phase4_env(PDP_Psystem_REDIX::Rule rule,
			PDP_Psystem_REDIX::Configuration configuration,
			PDP_Psystem_REDIX::Rhs rhs,
			PDP_Psystem_REDIX::NR nr,
			uint re_chunk, // standard mode
			bool modular=false, uint part_init=0, uint part_end=0) {  // modular mode
	_options options=d_options;
	uint resize=d_computations.resize;
	uint rpsize=d_computations.rpsize;

	uint env=blockIdx.x;
	uint sim=blockIdx.y;
	uint r=threadIdx.x;
	uint esize=options.num_objects*options.num_membranes;
	uint reini=rpsize+env*re_chunk;
	uint reend=rpsize+(env+1)*re_chunk;
	uint it=0;
	if (modular) {
		reini = part_init;
		reend = part_end;
	}

	r = reini+(it++)*blockDim.x+threadIdx.x;

	/* Communication rules, distributed among the environments */
	//Done in a kernel apart for keeping focus, besides both kernels can be run in parallel
	while ((r<resize) && (r<reend)) {

		uint N=nr[D_NR_E_IDX(r)];

		if (N>0){
			nr[D_NR_E_IDX(r)]=0; // reset NR
			int o_ini=rule.rhs_idx[r];
			int o_end=rule.rhs_idx[r+1];
			for (int o=o_ini; o<o_end; o++) {
				uint obj=rhs.object[o];
				uint denv=rhs.mmultiplicity[o];

				obj=sim*options.num_environments*esize+
					denv*esize+obj;

				atomicAdd(&(configuration.multiset[obj]),N);
			}
		}
		r = reini+(it++)*blockDim.x+threadIdx.x;
	}
}

int Simulator_gpu_dir::execution(uint step=0) {
		
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
	if (runcomp||profiling) {
		//	if (options->verbose>0)	cout << endl << "Launching kernel for phase 4: ";
		pdp_out->print_profiling_dcba_microphase_name("Launching kernel for phase 4");
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);
	}
	
	dim3 dimGrid (cu_blocksx, cu_blocksy);
	dim3 dimBlock (cu_threads);

	if (options->modular) {
		int stream_to_go=0;
		int stepc =step % options->cycles;

		for (int m=0;m<options->modules;m++) {
			bool module_active = stepc >= options->modules_start[m] && stepc < options->modules_end[m];
			if (!module_active) continue;

			// for Pi rule blocks within active modules
			uint start, end;
			module_start_end(start,end,m,true,options);

			if (start < end) {
				kernel_phase4_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->rule,
																				d_structures->configuration, d_structures->rhs,
																				d_structures->nr,
																				structures->ruleblock.rule_idx[start],
																				structures->ruleblock.rule_idx[end],true);

				stream_to_go = (stream_to_go + 1)%NUM_STREAMS;
			}

			// for env rule blocks within active modules
			module_start_end(start,end,m,false,options);

			if (start < end) {
				// for env rule blocks within active modules
				kernel_phase4_env <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->rule,
					d_structures->configuration, d_structures->rhs,
					d_structures->nr, re_chunk, true, start, end);

				stream_to_go = (stream_to_go + 1)%NUM_STREAMS;
			}
		}

		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}
		//cudaStreamSynchronize(execution_stream);
		getLastCudaError("kernel for phase 4 modular launch failure");
	}
	else if(options->micro){
		dim3 dimGrid (cu_blocksx, cu_blocksy);
		dim3 dimBlock (cu_threads);


		cudaStreamSynchronize(execution_stream);
		getLastCudaError("pre kernel for phase 4 micro launch failure");

		int stream_to_go=0;
		for(int i=0;i<large_blocks;i++){
			//Large blocks separated
			uint start=accum_offset[i];
			uint end=accum_offset[i+1];

			kernel_phase4_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->rule,
																			d_structures->configuration, d_structures->rhs,
																			d_structures->nr,
																			structures->ruleblock.rule_idx[start],
																			structures->ruleblock.rule_idx[end]);


			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;
		//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
		}
		for(int i=0;i<compacted_blocks;i++){
			//Small blocks in chunks
			uint start=accum_offset[part_indexes[i]];
			uint end=accum_offset[part_indexes[i+1]];
			kernel_phase4_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->rule,
																			d_structures->configuration, d_structures->rhs,
																			d_structures->nr,
																			structures->ruleblock.rule_idx[start],
																			structures->ruleblock.rule_idx[end]);


			stream_to_go++;
			if(stream_to_go==NUM_STREAMS)
				stream_to_go=0;
		//	std::cout<< "from rules "<< start<<" to "<<end<<std::endl;
		}
		//Independent ruleblocks
		uint start=accum_offset[options->num_partitions];
		uint end=start+options->independent_ruleblocks;
		kernel_phase4_rules <<<dimGrid,dimBlock,0,streams[stream_to_go]>>> (d_structures->rule,
													d_structures->configuration, d_structures->rhs,
													d_structures->nr,
													structures->ruleblock.rule_idx[start],
													structures->ruleblock.rule_idx[end]);

		kernel_phase4_env <<<dimGrid,dimBlock,0,execution_stream>>> (d_structures->rule,
					d_structures->configuration, d_structures->rhs,
					d_structures->nr, re_chunk);



		for(int i=0;i<NUM_STREAMS;i++){
			cudaStreamSynchronize(streams[i]);
		}
		cudaStreamSynchronize(execution_stream);

		getLastCudaError("kernel for phase 4 micro launch failure");

	}else{

		kernel_phase4 <<<dimGrid,dimBlock,0,execution_stream>>> (d_structures->rule,
			d_structures->configuration, d_structures->rhs,
			d_structures->nr, d_structures->pi_rule_size,
			d_structures->pi_rule_size+d_structures->env_rule_size,
			re_chunk, *options);

		kernel_phase4_env <<<dimGrid,dimBlock,0,execution_stream>>> (d_structures->rule,
							d_structures->configuration, d_structures->rhs,
							d_structures->nr, re_chunk);
		cudaStreamSynchronize(execution_stream);

		getLastCudaError("kernel for phase 4 launch failure");
	}

	if (runcomp||profiling) {
		//cudaStreamSynchronize(execution_stream);
		//getLastCudaError("kernel for phase 4 launch failure");

		sdkStopTimer(&counters.timer);
		counters.timesp4gpu+=sdkGetTimerValue(&counters.timer);
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
		gold_execution(step);
		
//	if (runcomp && options->verbose>1) {
//		cout << "Checking data from CPU:" << endl;
//		print_configuration(structures->configuration);
//	}
	
	if (runcomp && pdp_out->will_print_dcba_phase())
		cout << endl << "Time for phase 4: GPU=" << counters.timesp4gpu << " ms, CPU=" << counters.timesp4cpu << " ms, speedup=" << counters.timesp4cpu/counters.timesp4gpu << "x" << endl;

	pdp_out->print_end_profiling_dcba_phase();

	return 0;
}


bool Simulator_gpu_dir::check_step_errors(){

	checkCudaErrors(cudaMemcpyAsync(data_error, d_data_error, data_error_size*sizeof(uint), cudaMemcpyDeviceToHost,execution_stream));

	cudaStreamSynchronize(execution_stream);

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

		return true;
	}
	else
		pdp_out->print_profiling_dcba_microphase_result(true);


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
		return true;
	}
	else pdp_out->print_profiling_dcba_microphase_result(true);

	return false;

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
	} else if (profiling) {
		float totalselgpu=counters->timesp1gpu+counters->timesp2gpu+counters->timesp3gpu;
		float totalexgpu=counters->timesp4gpu;
		float totalgpu=totalselgpu+totalexgpu;
		cout << "Total ms) \t Selection (ms) \t Execution (ms) \t Phase 1 (ms) \t Phase 2 (ms) \t Phase 3 (ms) \t Phase 4 (ms) \t Filters (ms) \t Norm (ms) \t Update (ms)" << endl;
		cout << totalgpu << "\t" << totalselgpu << "\t" << totalexgpu << "\t" << counters->timesp1gpu << "\t" << counters->timesp2gpu << "\t" << counters->timesp3gpu << "\t" << counters->timesp4gpu << "\t" << counters->timek1gpu << "\t" << counters->timek2gpu << "\t" << counters->timek3gpu << endl;
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
// This function should be called after executing the microphase
void PDP_Psystem_redix_out_std_gpuwrapper::print_block_competition(int competing_block,bool env_blocks) {
	if (!runcomp && !pdpout->will_print_configuration()) return;
	pdpout->print_block_competition(competing_block,env_blocks);

}
