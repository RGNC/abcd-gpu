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

#include "pdp_psystem_sab.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

/* Random generator of a PDP P System */
PDP_Psystem_SAB::PDP_Psystem_SAB(PDP_Psystem_source * source) {
	unsigned int rule_ini=0;
	unsigned int lhs_ini=0;

	options=source->options;

	if (options==NULL || !source->start()) {
		cerr<< "Error in the parameters or in the source" << endl;
		exit(1);
	}

	structures = new struct _structures;

	/* Create empty data for ruleblocks */
	structures->ruleblock_size = options->num_rule_blocks+options->num_blocks_env;
	structures->ruleblock.lhs_ini = new LHS_INI [structures->ruleblock_size];
	structures->ruleblock.lhs_length = new LHS_LENGTH [structures->ruleblock_size];
	structures->ruleblock.rule_ini = new RULE_INI [structures->ruleblock_size];
	structures->ruleblock.rule_length = new RULE_LENGTH [structures->ruleblock_size];
	structures->ruleblock.active_membrane = new MEMBRANE [structures->ruleblock_size];
	structures->ruleblock.lhs_charge = new CHARGE [structures->ruleblock_size];
	structures->ruleblock.rhs_charge = new CHARGE [structures->ruleblock_size];

	/* For each ruleblock in Pi, calculate number of rules and lhs */
	source->pi_loop_rule_blocks();
	unsigned int block=0;

	do {
		unsigned int rule_length = source->pi_loop_rules();
		unsigned int lhs_length = source->pi_loop_lhs();
		unsigned int membr = source->pi_lhs_membrane();

		/* Choose number of rules in the block */
		structures->ruleblock.rule_length[block] = rule_length;
		structures->ruleblock.rule_ini[block] = rule_ini;
		rule_ini += rule_length;

		/* Choose active membrane */
		structures->ruleblock.active_membrane[block] = membr;
		structures->ruleblock.lhs_charge[block] = source->pi_lhs_charge();
		structures->ruleblock.rhs_charge[block] = source->pi_rhs_charge();

		/* Choose length for left hand side */
		structures->ruleblock.lhs_length[block] = lhs_length;
		structures->ruleblock.lhs_ini[block] = lhs_ini;
		lhs_ini += structures->ruleblock.lhs_length[block];

		block++;
	} while (source->pi_next_rule_block());

	/* For each ruleblock of communication, calculate number of rules */
	unsigned int env_rule_ini=rule_ini;
	source->env_loop_rule_blocks();
	do {
		unsigned int rule_length = source->env_loop_rules();
		unsigned int lhs = source->env_get_object_lhs();
		unsigned int membr = source->env_get_environment();

		/* Choose number of rules in the block */
		structures->ruleblock.rule_length[block] = rule_length;
		structures->ruleblock.rule_ini[block] = env_rule_ini;
		env_rule_ini += rule_length;

		/* Choose active membrane */
		structures->ruleblock.active_membrane[block] = membr;
		structures->ruleblock.lhs_charge[block] = 'e';
		structures->ruleblock.rhs_charge[block] = 'n';

		/* Choose length for left hand side */
		structures->ruleblock.lhs_length[block] = 1;
		structures->ruleblock.lhs_ini[block] = lhs;

		block++;
	} while (source->env_next_rule_block());

	/* Create empty data for rules */
	structures->rule_size=rule_ini;
	structures->env_rule_size=env_rule_ini-rule_ini;
	structures->rule.rhs_ini = new RHS_INI [structures->rule_size+structures->env_rule_size];
	structures->rule.rhs_length = new RHS_LENGTH [structures->rule_size+structures->env_rule_size];

	/* Create empty data for LHS */
	structures->lhs_size=lhs_ini;
	structures->lhs.object = new OBJECT [structures->lhs_size];
	structures->lhs.multiplicity = new MULTIPLICITY [structures->lhs_size];
	structures->lhs.membrane = new MEMBRANE [structures->lhs_size];

	/* Create empty probabilities */
	structures->probability_size = structures->rule_size*options->num_environments+structures->env_rule_size;
	structures->probability = new PROBABILITY [structures->probability_size];

	int rhs_ini=0;

	/* For each ruleblock in Pi: create rhs, probabilities and lhs */
    source->pi_loop_rule_blocks();
    block=0;

	do {
		source->pi_loop_rules();
		unsigned int i=structures->ruleblock.rule_ini[block];

		/* Create RHS, init probabilities */
		do {
			structures->rule.rhs_ini[i] = rhs_ini;
			structures->rule.rhs_length[i] = source->pi_loop_rhs();

			rhs_ini += structures->rule.rhs_length[i];

			float *prob = source->pi_rule_probabilities();

			for (int env=0; env<options->num_environments; env++) {
				structures->probability[env*structures->rule_size+i]=prob[env];
			}

			i++;
		}while(source->pi_next_rule());
		
		/* Init LHS */
		OBJECT obj=0;
		MULTIPLICITY mult=0;

		i=structures->ruleblock.lhs_ini[block];

		source->pi_loop_lhs();
		source->pi_lhs_loop_U();

		while (source->pi_lhs_next_object(obj,mult)) {
			structures->lhs.object[i] = obj;
			structures->lhs.multiplicity[i] = mult;
			structures->lhs.membrane[i++] = source->pi_lhs_parent_membrane();
		}

		source->pi_lhs_loop_V();

		while (source->pi_lhs_next_object(obj,mult)) {
			structures->lhs.object[i] = obj;
			structures->lhs.multiplicity[i] = mult;
			structures->lhs.membrane[i++] = source->pi_lhs_membrane();
		}

		block++;
	} while (source->pi_next_rule_block());

	/* For each ruleblock in communication, create RHS and prob */
	source->env_loop_rule_blocks();

	do {
		source->env_loop_rules();
		unsigned int i=structures->ruleblock.rule_ini[block];

		/* Create RHS, init probabilities */
		do {

			structures->rule.rhs_ini[i] = rhs_ini;
			structures->rule.rhs_length[i] = source->env_loop_rhs();
			rhs_ini += structures->rule.rhs_length[i];

			float prob = source->env_get_probability();

			structures->probability[options->num_environments*structures->rule_size+(i-structures->rule_size)]=prob;

			i++;
		}while(source->env_next_rule());
		block++;

	} while (source->env_next_rule_block());

	structures->rhs_size = rhs_ini;
	structures->rhs.object = new OBJECT [structures->rhs_size];
	structures->rhs.multiplicity = new MULTIPLICITY [structures->rhs_size];
	structures->rhs.membrane = new MEMBRANE [structures->rhs_size];

	/* For each ruleblock in Pi, initiate RHS */
	source->pi_loop_rule_blocks();
	block=0;

    do {
    	source->pi_loop_rules();
    	/* For each rule in the block */
    	unsigned int r=structures->ruleblock.rule_ini[block];

    	do {
    		/* Initialize RHS */
    		source->pi_loop_rhs();
			source->pi_rhs_loop_U();
			int o=structures->rule.rhs_ini[r];
			unsigned int obj=0,mult=0;

			while (source->pi_rhs_next_object(obj,mult)) {
				structures->rhs.object[o] = obj;
				structures->rhs.multiplicity[o] = mult;
				structures->rhs.membrane[o++] = source->pi_lhs_parent_membrane();
			}

			source->pi_rhs_loop_V();

			while (source->pi_rhs_next_object(obj,mult)) {
				structures->rhs.object[o] = obj;
				structures->rhs.multiplicity[o] = mult;
				structures->rhs.membrane[o++] = source->pi_lhs_membrane();
			}

    		r++;
		} while (source->pi_next_rule());

    	block++;
	} while (source->pi_next_rule_block());

    /* For each ruleblock for communication, initiate RHS */
    /* The wonderful thing of this solution must be improved */
	source->env_loop_rule_blocks();

	do {
		source->env_loop_rules();
		/* For each rule in the block */
		unsigned int r=structures->ruleblock.rule_ini[block];

		do {
			/* Initialize RHS */
			source->env_loop_rhs();

			int o=structures->rule.rhs_ini[r];
			unsigned int obj=0,mult=0;

			while (source->env_next_object(obj,mult)) {
				structures->rhs.object[o] = obj;
				structures->rhs.multiplicity[o] = mult;
				structures->rhs.membrane[o++] = mult;
			}

			r++;
		} while (source->env_next_rule());

		block++;
	} while (source->env_next_rule_block());

    /* Initialize Nb */
	//structures->nb_size= ((options->num_rule_blocks)*options->num_environments+options->num_blocks_env);//*options->num_parallel_simulations;
	//structures->nb = new MULTIPLICITY [structures->nb_size];

	/* Initialize Nr */
	//structures->nr_size= ((structures->rule_size)*options->num_environments+structures->env_rule_size);//*options->num_parallel_simulations;
	//structures->nr = new MULTIPLICITY [structures->nr_size];

	/* Initialize Configuration */
	//structures->configuration; // = new Configuration;

	structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments;//*options->num_parallel_simulations;
	structures->configuration.multiset = new MULTIPLICITY [structures->configuration.multiset_size];

	structures->configuration.membrane_size = options->num_membranes*options->num_environments;
	structures->configuration.membrane = new CHARGE [structures->configuration.membrane_size];

	source->conf_loop();

	source->conf_loop_environments();
	unsigned int env=0;

	do {
		unsigned int memb=0;
		source->conf_loop_membranes();

		do {
			structures->configuration.membrane[env*options->num_membranes+memb]=source->conf_charge_membrane();

			source->conf_loop_objects();
			unsigned int object=0;
			unsigned int multiplicity=0;

			while (source->conf_next_object(object,multiplicity)) {
				for (int s=0;s<options->num_parallel_simulations;s++)
					structures->configuration.multiset[s*options->num_environments*options->num_membranes*options->num_objects
					                                   +env*options->num_membranes*options->num_objects+memb*options->num_objects
					                                   +object]=multiplicity;
			}

			memb++;
		}while (source->conf_next_membrane());
		env++;
	} while (source->conf_next_environment());

	/* Finally, process end, and printing system */
	print();
}

void PDP_Psystem_SAB::print() {
	//int bsize = options->num_parallel_simulations*options->num_environments;

	unsigned int brmem = (structures->ruleblock_size) * (sizeof(LHS_INI)+sizeof(LHS_LENGTH)+sizeof(RULE_INI)+sizeof(RULE_LENGTH)+sizeof(MEMBRANE)+sizeof(CHARGE));
	unsigned int lhsmem = structures->lhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY)+sizeof(MEMBRANE));
	unsigned int rmem = (structures->rule_size+structures->env_rule_size) * (sizeof(RHS_INI)+sizeof(RHS_LENGTH));
	unsigned int pmem = structures->probability_size * sizeof(float);
	unsigned int rhsmem = structures->rhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY)+sizeof(MEMBRANE));
	unsigned int multmem = structures->configuration.multiset_size * sizeof(MULTIPLICITY);
	unsigned int mbmem = structures->configuration.membrane_size * (2*sizeof(CHARGE)+sizeof(MEMBRANE));

	options->mem = brmem+lhsmem+rmem+pmem+rhsmem+multmem+mbmem;

	if (options->verbose > 0) {
		cout << "The P system has been successfully initiated" << endl << endl << "--------------------------" << endl ;

		cout << "SIZES: " << endl;

		cout << "\t Num rule blocks for Pi: " << options->num_rule_blocks << endl;
		cout << "\t Num rule blocks for envs: " << options->num_blocks_env << endl;
		cout << "\t Num simulations: " << options->num_simulations << endl;
		cout << "\t Num parallel simulations: " << options->num_parallel_simulations << endl;
		cout << "\t Num environments: " << options->num_environments << endl;
		cout << "\t Num objects: " << options->num_objects << endl;
		cout << "\t Num membranes: " << options->num_membranes << endl;
		cout << "\t Max num rules: " << options->max_num_rules << endl;
		cout << "\t Max lhs: " << options->max_lhs << endl;
		cout << "\t Max rhs: " << options->max_rhs << endl << endl << "--------------------------" << endl << endl;


		if (options->verbose > 1 ) {

		cout << "RULE BLOCKS: " << structures->ruleblock_size << endl << endl ;

		for (int block=0; block < options->num_rule_blocks; block++) {
			cout << "Block " << block << endl;
			cout << "\t Info: " << endl;
			cout << "\t\t*rule_ini = " << structures->ruleblock.rule_ini[block] << endl;
			cout << "\t\t*rule_length = " << structures->ruleblock.rule_length[block] << endl;
			cout << "\t\t*lhs_ini = " << structures->ruleblock.lhs_ini[block] << endl;
			cout << "\t\t*lhs_length = " << structures->ruleblock.lhs_length[block] << endl;
			cout << "\t\t*membrane = " << structures->ruleblock.active_membrane[block] << endl;
			cout << "\t\t*lhs_charge = " << structures->ruleblock.lhs_charge[block] << endl;
			cout << "\t\t*rhs_charge = " << structures->ruleblock.rhs_charge[block] << endl << endl;

			cout << "\t Left-hand side = " ;
			for (int j=structures->ruleblock.lhs_ini[block]; j<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; j++){
				cout << "<obj=" << structures->lhs.object[j] << ",mult=" << structures->lhs.multiplicity[j] << ",membr=" << structures->lhs.membrane[j]<<">, ";
			}
			cout << endl << endl;

			cout << "\t Rules: " << endl;
			for (int j=structures->ruleblock.rule_ini[block]; j<structures->ruleblock.rule_ini[block]+structures->ruleblock.rule_length[block]; j++){
				cout << "\t\t Rule " << j << ":" << endl;
				cout << "\t\t\t Probabilities (by envs): " ;

				for (int k=0; k<options->num_environments; k++) {
					cout << structures->probability[k*structures->rule_size+j] << ", ";
				}
			
				cout << endl << "\t\t\t Right-hand side: ";
			
				for (int k=structures->rule.rhs_ini[j]; k<structures->rule.rhs_ini[j]+structures->rule.rhs_length[j]; k++) {
					cout << "<obj=" << structures->rhs.object[k] << ",mult=" << structures->rhs.multiplicity[k] << ",membr=" << structures->rhs.membrane[k] << ">, ";
				}
				cout << endl;
			}
			cout << endl << "---" << endl << endl;
		}
	
		for (int block = options->num_rule_blocks; block < options->num_blocks_env+options->num_rule_blocks; block++) {
			cout << "Block " << block << endl;
			cout << "\t Info: " << endl;
			cout << "\t\t*rule_ini = " << structures->ruleblock.rule_ini[block] << endl;
			cout << "\t\t*rule_length = " << structures->ruleblock.rule_length[block] << endl;
			cout << "\t\t*lhs_ini (object) = " << structures->ruleblock.lhs_ini[block] << endl;
			cout << "\t\t*lhs_length = " << structures->ruleblock.lhs_length[block] << endl;
			cout << "\t\t*environment = " << structures->ruleblock.active_membrane[block] << endl;
			cout << "\t\t*lhs_charge = " << structures->ruleblock.lhs_charge[block] << endl;
			cout << "\t\t*rhs_charge = " << structures->ruleblock.rhs_charge[block] << endl << endl;

			cout << "\t Rules: " << endl;
			for (int j=structures->ruleblock.rule_ini[block]; j<structures->ruleblock.rule_ini[block]+structures->ruleblock.rule_length[block]; j++){
				cout << "\t\t Rule " << j << ":" << endl;
				cout << "\t\t\t Probability: " << structures->probability[options->num_environments*structures->rule_size+(j-structures->rule_size)];

				cout << endl << "\t\t\t Right-hand side: ";

				for (int k=structures->rule.rhs_ini[j]; k<structures->rule.rhs_ini[j]+structures->rule.rhs_length[j]; k++) {
					cout << "<obj=" << structures->rhs.object[k] << ",mult=" << structures->rhs.multiplicity[k] << ",env=" << structures->rhs.membrane[k] << ">, ";
				}
				cout << endl;
			}
			cout << endl << "---" << endl << endl;
		}

		cout << "--------------------------" << endl << "CONFIGURATION" << endl << endl;

		cout << "Multiset" << endl;
	
		for (int sim=0; sim<options->num_parallel_simulations; sim++) {
			for (int env=0; env<options->num_environments; env++) {
				cout << "Simulation " << sim << ", environment " << env << ":";

				for (int memb=0; memb<options->num_membranes; memb++) {
					cout << endl << "\t\tMembrane " << memb << ": ";
					for (int obj=0; obj<options->num_objects; obj++) {
						cout << "Obj " << obj
								<< "*" << structures->configuration.multiset[sim*options->num_objects*options->num_membranes*options->num_environments+env*options->num_objects*options->num_membranes+memb*options->num_objects+obj] << ", ";
					}
				}

				cout << endl;
			}
		}

		cout << endl << endl;
		} // Verbose 1

        cout << "--------------------------" << endl << "MEMORY REQUIREMENTS" << endl << endl;

        cout << "Ruleblock: " << brmem << " (" << brmem/1024 << "KB)" << endl;
        cout << "Lhs: " << lhsmem << " (" << lhsmem/1024 << "KB)" << endl;
        cout << "Rules:" << rmem << " (" << rmem/1024 << "KB)" << endl;
        cout << "Probabilities: " << pmem <<  " (" << pmem/1024 << "KB)" << endl;
        cout << "Rhs: " << rhsmem << " (" << rhsmem/1024 << "KB)" << endl;
        cout << "Multisets: " << multmem << " (" << multmem/1024 << "KB)" << endl;
        cout << "Membrane: " << mbmem << " (" << mbmem/1024 << "KB)" << endl;

        unsigned long int memb = options->mem = brmem+lhsmem+rmem+pmem+rhsmem+multmem+mbmem;
        int count=0;
        float div=1;
        char unit[6]={' ','K','M','G','T','P'};
        while ((memb/div)>1023 && count<3){
        	div*=1024;
        	count++;
        }
        cout << "TOTAL: " << memb << " (" << memb/div << " " << unit[count] << "B)" << endl << endl;

        cout << "Printing system finished" << endl << endl;


	} // Verbose 0
}

