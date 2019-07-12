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

#ifndef __PDP_P_SYSTEM_REDIX__
#define __PDP_P_SYSTEM_REDIX__

#include "pdp_psystem.h"

/* Some extra macros */

#define COMPOSE_MEMBRANE(membrane,alpha,beta) ((0xA<<28) | (membrane << 4) | ((beta&0x3) << 2) | (alpha&0x3))
#define GET_MEMBRANE(membrane) ((membrane>>4)&0xFFFFFF)
#define GET_ALPHA(membrane) (membrane&0x3)
#define GET_BETA(membrane) ((membrane>>2)&0x3)
#define IS_MEMBRANE(membrane) ((membrane>>28)==0xA)

#define COMPOSE_ENVIRONMENT(environment) ((0xF << 28) | (environment&0x0FFFFFFF))
#define GET_ENVIRONMENT(env) (env&0x0FFFFFFF)
#define IS_ENVIRONMENT(env) ((env>>28)==0xF)

#define COMPOSE_MMULTIPLICITY(mult,memb) ((mult<<16) | (memb&0xFFFF))
#define GET_MULTIPLICITY(mult) (mult>>16)
#define GET_MEMBR(memb) (memb&0xFFFF)

#define CHARGE_TO_CHAR(c) ((c==0)?"0":((c==1)?"+":((c==2)?"-":"W")))

/*
 * Definitions for data implementation
 */

typedef unsigned int LHS_IDX;
typedef unsigned int RHS_IDX;
typedef unsigned int RULE_IDX;
typedef unsigned int MEMBRANE;
typedef unsigned int OBJECT;
typedef unsigned int MULTIPLICITY;
typedef float INV_MULTIPLICITY;
typedef char CHARGE;
typedef float PROBABILITY;


/* Class of P system structures based on struct of arrays, reduced by storing indexes */
class PDP_Psystem_REDIX: public PDP_Psystem {
    //friend class PDP_Psystem_redix_out_std;
public:
	PDP_Psystem_REDIX(PDP_Psystem_source* PDPps_source);
	void print();

	/*
	 * Definition file for structures
	 */

	struct _ruleblock {
		LHS_IDX *lhs_idx;
		RULE_IDX *rule_idx;
		MEMBRANE *membrane;
	};

	typedef struct _ruleblock Ruleblock;

	struct _rule_object_lhs {
		OBJECT *object;
		MULTIPLICITY *mmultiplicity;
		INV_MULTIPLICITY * imultiplicity;
	};

	struct _rule_object_rhs {
		OBJECT *object;
		MULTIPLICITY *mmultiplicity;
	};

	typedef struct _rule_object_lhs Lhs;

	struct _rule {
		RHS_IDX *rhs_idx;
	};

	typedef struct _rule Rule;

	typedef PROBABILITY *Probability;

	typedef struct _rule_object_rhs Rhs;

	typedef MULTIPLICITY *NR;

	struct _configuration {
		MULTIPLICITY * multiset;
		unsigned int multiset_size;
		CHARGE * membrane;
		unsigned int membrane_size;
	};

	typedef struct _configuration Configuration;

	struct _stringids {
	    char** id_objects;
		char** id_environments;
		char** id_membranes;
	};
	
	typedef struct _stringids Stringids;
	/*
	 * General structure
	 */

	struct _structures {
		Ruleblock ruleblock;
		unsigned int ruleblock_size;
		Lhs lhs;
		unsigned int lhs_size;
		Rule rule;
		unsigned int pi_rule_size,env_rule_size;
		Probability probability;
		unsigned int probability_size;
		Rhs rhs;
		unsigned int rhs_size;
		NR nr;
		unsigned int nr_size;
		//***************************
		// TODO: Delete
		NR nb;
		unsigned int nb_size;
		//***************************
		Configuration configuration;
		Stringids stringids;
	};

	typedef struct _structures* Structures;
	typedef struct _structures struct_structures;

	Structures structures;
};

/* The class for PDP system outputs through the standard output (can be the
 * screen, a file, etc., depending if the binary abcd-gpu was called forwarding
 * the standard output "abcd-gpu > output.txt" or "abcd-gpu | less"
 * */

class PDP_Psystem_redix_out_std {
    friend class PDP_Psystem_REDIX;
    friend class PDP_Psystem_redix_out_std_gpuwrapper;
private:
    Options options;
    PDP_Psystem_REDIX* pdp;
    PDP_Psystem_REDIX::Structures structures;
    int verbosity;
    unsigned int msize,esize,bpsize,besize,rpsize,resize;
    
public:
    PDP_Psystem_redix_out_std(PDP_Psystem_REDIX* pdpsys);
    ~PDP_Psystem_redix_out_std();
    
    /* Next functions will print different stuff *
     * with functions "will_*" you can tune when */
    
    /* Verbosity 1 */
    inline bool will_print_simulator_process() {return options->verbose>=1;}
    void print_simulator_process(const char * message);
    
    /* Verbosity 2 */
    inline bool will_print_simulation() {return options->verbose>=2;}
    void print_simulation(int simulation);
    
    inline bool will_print_step() {return options->verbose>=2;}
    void print_step(int step);
    
    inline bool will_print_sim_step() {return options->verbose>=2;}
    void print_sim_step(int simulation, int step);
    
    inline bool will_print_sim_thread() {return options->verbose>=2;}
    void print_sim_thread(int simulation, int stid);
    
    inline bool will_print_sim_range() {return options->verbose>=2;}
    void print_sim_range (int inisim, int endsim);
    
    /* Verbosity 3 */
    inline bool will_print_dcba_phase() {return options->verbose>=3;}
    void print_dcba_phase(int phase);
    
    /* Verbosity 4 */
    inline bool will_print_configuration() {return options->verbose>=4;}
    void print_configuration(int sim);
    
    /* Verbosity 5 */
    inline bool will_print_block_selection() {return options->verbose>=5;}
    void print_block_selection(int sim);
    
    inline bool will_print_rule_selection() {return options->verbose>=5;}
    void print_rule_selection(int sim);
    
    /* Verbosity 6 */
    inline bool will_print_temporal_configuration() {return options->verbose>=6;}
    void print_temporal_configuration(int sim);
};

#endif
