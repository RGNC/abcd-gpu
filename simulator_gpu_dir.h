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

#ifndef __GPU_SIM__DIR__
#define __GPU_SIM__DIR__

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_math.h>
// We have moved to the timer implementation in CUDA SDK 5.5. It's nice since
// it works both for windows and linux. 
// TODO: Provide our own library based on it, in case CUDA is not in the system.
#include <helper_timer.h>
#include "pdp_psystem_redix.h"
#include "pdp_psystem_output.h"
#include "simulator.h"

/* ERROR VALUES */
#define NO_ERROR 0
#define CONSISTENCY_ERROR 1
#define UPDATING_CONFIGURATION_ERROR 2
// Add new type of errors here
#define UNDEFINED_ERROR 255

/* ABV VALUES */
#define ABV_T uint
#define ABV_WORD_SIZE 32
#define ABV_LOG_WORD_SIZE 5
#define ABV_DESPL_MASK 0x1F
#define ABV_INIT_WORD 0xFFFFFFFF

/* A class for printing PDP systems from GPU */
/* Forwarded declared, go to the end of the file */

class PDP_Psystem_redix_out_std_gpuwrapper;


/* Simulator of PDP systems on the GPU*/

class Simulator_gpu_dir: public Simulator {
	friend class PDP_Psystem_redix_out_std_gpuwrapper;

public:

	Simulator_gpu_dir(PDP_Psystem_REDIX* PDPps,int mode=0,bool accurate=true, PDP_Psystem_output* PDPout=NULL);

	~Simulator_gpu_dir() {
		del();
	}

	bool run();
	bool step(int k);
	
	/* Methods for querying data */
	unsigned int get_object_value(  int obj, 
					int membr, 
					int env,
					int sim) {
		return structures->configuration.multiset[sim*options->num_environments*options->num_membranes*options->num_objects+env*options->num_membranes*options->num_objects+membr*options->num_objects+obj];
	}

	char get_membrane_charge(  int membr,
				   int env,
				   int sim) {
		return structures->configuration.membrane[sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+membr];
	}

private:
	/*****************************/
	/* Error handling structures */
	/*****************************/
	uint data_error_size;
	uint * data_error, *d_data_error;

	/*****************************/
	/* Auxiliary data structures */
	/*****************************/
	PDP_Psystem_REDIX::Structures structures, d_structures;
	Options options;

	MULTIPLICITY *d_nb;
	float *addition, *d_addition;
	uint * ini_denominator, * ini_numerator;
	uint * denominator, * numerator, *d_denominator, *d_numerator, *d_ini_numerator;
	PDP_Psystem_REDIX::Configuration ini_cfg, d_cfg;
	unsigned int msize,esize,bpsize,besize,rpsize,resize;
	unsigned int addition_size;
	
	/* Options for the simulator */
	bool runcomp;
	int mode;
	bool accurate;
	bool error;

	/* Variables for parallelization */
	int max_threads;
	int env_parallel;
	int sim_parallel;
	cudaDeviceProp dev_property;

	/*******************************/
	PDP_Psystem_output * PDPout;
	bool will_out;

	/*******************************/
	/* Only for debugging purposes */

    PDP_Psystem_redix_out_std_gpuwrapper* pdp_out;
    //PDP_Psystem_redix_out_std* pdp_out;
	// TODO: delete after their replacement 
	void print_configuration(PDP_Psystem_REDIX::Configuration conf);
	void print_block_applications(uint * nb);
        /************************************************/
        
        
	/* Initialization */
	bool init();
	void del();
	void reset(int sim_ini);
	void retrieve(int sim_ini);

	/* Micro phases executed on GPU */
	bool selection_phase1();
	bool selection_phase2();
	bool selection_phase3();

	/* Micro phases executed on CPU */
	void gold_selection_phase1();
	void gold_selection_phase2();
	void gold_selection_phase3();
	
	void gold_selection_phase1_acu();
	
	unsigned int gold_execution();

	/* Main phases */
	int selection();
	int execution();
	
	/***************************************************************/
	/* The following defines how to use activation bit vectors ABV */
	/***************************************************************/
	unsigned int asize,abv_size;
	ABV_T * abv, *d_abv;


	void initialize_abv () {
		abv_size=asize*options->num_environments*options->num_parallel_simulations;
		abv = new unsigned int [abv_size];
	}

	bool is_active (uint block, uint env, uint sim) {
		return (abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
		               >> ((~block)&ABV_DESPL_MASK))
		        & 0x1;
	}

	void set_activation (bool value, uint block, uint env, uint sim) {
		abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)] =
				(abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
				      & ~(0x1<<((~block)&ABV_DESPL_MASK)))
				|(value << ((~block)&ABV_DESPL_MASK));
	}

	void deactivate(uint block, uint env, uint sim) {
		abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)] &=
				~(0x1<<((~block)&ABV_DESPL_MASK));
	}

	void reset_activations () {
		for (unsigned int i=0; i<asize*options->num_environments*options->num_parallel_simulations; i++)
			abv[i]=ABV_INIT_WORD;
	}
	
	/****************************/
	/* Time counters structures */
	/****************************/

	struct _time_values {
		StopWatchInterface* timer;
		
		float timesp1gpu;
		float timesp1cpu;		
		
		float timek1gpu;
		float timek2gpu;
		float timek3gpu;
		float timek1cpu;
		float timek2cpu; 
		float timek3cpu;
		
		float timesp2gpu;
		float timesp2cpu;
		
		float timesp3gpu;
		float timesp3cpu;
		
		float timesp4gpu;
		float timesp4cpu;
	} counters;	
	
	typedef struct _time_values * Time_values;

	/* Safely add and mult uint numbers, returns true if overflow */
	inline bool safe_u_add(uint& op1, uint op2){
		uint c_test=op1+op2;
		if (c_test-op2 != op1) 
			return true;
		else op1=c_test;

		return false;
	}

	inline bool safe_u_mul(uint& op1, uint op2) {
		uint c_test=op1*op2;
		if (c_test/op2 != op1) 
			return true;
		else op1=c_test;	

		return false;
	}
};



/********************************************************************/
/* A new class for printing PDP systems, the idea is to automatically
 * retrieve the required data from the GPU only when necessary. */

class PDP_Psystem_redix_out_std_gpuwrapper {
private:
	PDP_Psystem_redix_out_std * pdpout;
	PDP_Psystem_REDIX::Structures d_structures;
	PDP_Psystem_REDIX::Structures structures,original_structures;
	Options options;
	Simulator_gpu_dir::Time_values counters;
	bool runcomp;
	// Parallel Simulation Block
	int psb;

	void retrieve_configuration();
	void retrieve_block();
	void retrieve_rule();

public:
	// Create the object only when the simulator has been initialized (both in GPU and in CPU)
	// The structures from pdp object will be used to retrieve data from GPU, be sure that you have a copy beforehand
	// The structures on the GPU should be created. It can be assigned later instead, but be sure to have them initialized
	//  before using print methods at your own risk!
    PDP_Psystem_redix_out_std_gpuwrapper (PDP_Psystem_REDIX* pdp, PDP_Psystem_REDIX::Structures d_structures=NULL, bool runcomp=false, Simulator_gpu_dir::Time_values counters=NULL) {
    	pdpout=new PDP_Psystem_redix_out_std(pdp);
    	options=pdp->options;
    	this->d_structures=d_structures;
    	this->counters=counters;
    	this->runcomp=runcomp;
    	this->psb=0;


    	// My own copy, since nb and nr might not be initialized in pdp
    	original_structures=pdpout->structures;
    	structures = new PDP_Psystem_REDIX::_structures;
    	pdpout->structures=structures;

    	if (pdpout->will_print_configuration()) {
    		structures->configuration.membrane_size=d_structures->configuration.membrane_size;
    		structures->configuration.membrane = new CHARGE[structures->configuration.membrane_size];

    		structures->configuration.multiset_size = d_structures->configuration.multiset_size;
    		structures->configuration.multiset = new MULTIPLICITY[structures->configuration.multiset_size];

    		structures->stringids=pdp->structures->stringids;
    	}

    	if (pdpout->will_print_block_selection()) {
    		structures->nb_size = d_structures->nb_size;
    		structures->nb = new MULTIPLICITY [structures->nb_size];

    		structures->lhs_size=pdp->structures->lhs_size;
    		structures->lhs=pdp->structures->lhs;

    		structures->ruleblock=pdp->structures->ruleblock;
    		structures->ruleblock_size=pdp->structures->ruleblock_size;
    	}

    	if (pdpout->will_print_rule_selection()) {
    		structures->nr_size = d_structures->nr_size;
    		structures->nr = new MULTIPLICITY [structures->nr_size];

    		structures->probability=pdp->structures->probability;
    		structures->probability_size=pdp->structures->probability_size;

    		structures->rhs_size=pdp->structures->rhs_size;
    		structures->rhs=pdp->structures->rhs;

    		structures->rule=pdp->structures->rule;

    		structures->env_rule_size=pdp->structures->env_rule_size;
    		structures->pi_rule_size=pdp->structures->pi_rule_size;
    	}
    }

    ~PDP_Psystem_redix_out_std_gpuwrapper() {
    	if (pdpout->will_print_configuration()) {
    		delete [] structures->configuration.membrane;
    		delete [] structures->configuration.multiset;
    	}
    	if (pdpout->will_print_block_selection())
    		delete [] structures->nb;
    	if (pdpout->will_print_rule_selection())
    		delete [] structures->nr;

    	delete structures;

    	pdpout->structures=original_structures;
    }

    void set_device_structures(PDP_Psystem_REDIX::Structures d_structures) {
        this->d_structures=d_structures;
    }

    // Next methods are overridden, we need first to retrieve data from GPU!
    void print_simulator_process(const char * message) {
    	pdpout->print_simulator_process(message);
    }

    void print_simulation(int simulation) {
    	pdpout->print_simulation(simulation);
    }

    void print_step(int step) {
    	pdpout->print_step(step);
    }

    void print_sim_step(int simulation, int step) {
    	pdpout->print_sim_step(simulation,step);
    }

    void print_sim_thread(int simulation, int stid) {
    	pdpout->print_sim_thread(simulation,stid);
    }

    // Only by printing the range can you change the psb
    void print_sim_range (int inisim, int endsim) {
    	psb=inisim;
    	pdpout->print_sim_range(inisim,endsim);
    }

    void print_dcba_phase(int phase) {
    	pdpout->print_dcba_phase(phase);
    }

    void print_configuration() {
    	if (!pdpout->will_print_configuration()) return;

    	// TODO: retrieve only the portion of configuration for range of simulations
    	retrieve_configuration();

    	for (int sim=0;sim<options->num_parallel_simulations;sim++) {
    		pdpout->print_simulation(psb+sim);
    		pdpout->print_configuration(sim);
    	}
    }

    void print_block_selection() {
    	if (!pdpout->will_print_block_selection()) return;

    	retrieve_block();

    	for (int sim=0;sim<options->num_parallel_simulations;sim++) {
    		pdpout->print_simulation(psb+sim);
    		pdpout->print_block_selection(sim);
    	}
    }

    void print_rule_selection() {
    	if (!pdpout->will_print_rule_selection()) return;

    	retrieve_rule();
    	//retrieve_block();

    	for (int sim=0;sim<options->num_parallel_simulations;sim++) {
    		pdpout->print_simulation(psb+sim);
    		pdpout->print_rule_selection(sim);
    	}
    }

    void print_temporal_configuration();

    /* Concerning run comparison between GPU and CPU, and profiling */
    void print_profiling_table ();

    void print_profiling_dcba_phase (const char * message);
    void print_end_profiling_dcba_phase ();

    // This function should be called before executing the microphase
    void print_profiling_dcba_microphase_name (const char * message);
    // This function should be called after executing the microphase
    void print_profiling_dcba_microphase_result (float time);
    // This function should be called after executing the microphase
    void print_profiling_dcba_microphase_result (bool result);

    void print_profiling_dcba_microphase_datum (const char * message1, float datum, const char * message2);

    // Will prints
    inline bool will_print_simulator_process() {return pdpout->will_print_simulator_process();}
    inline bool will_print_simulation() {return pdpout->will_print_simulation();}
    inline bool will_print_step() {return pdpout->will_print_step();}
    inline bool will_print_sim_step() {return pdpout->will_print_sim_step();}
    inline bool will_print_sim_thread() {return pdpout->will_print_sim_thread();}
    inline bool will_print_sim_range() {return pdpout->will_print_sim_range();}
    inline bool will_print_dcba_phase() {return pdpout->will_print_dcba_phase();}
    inline bool will_print_configuration() {return pdpout->will_print_configuration();}
    inline bool will_print_block_selection() {return pdpout->will_print_block_selection();}
    inline bool will_print_rule_selection() {return pdpout->will_print_rule_selection();}
    inline bool will_print_temporal_configuration() {return pdpout->will_print_temporal_configuration();}

};

#endif
