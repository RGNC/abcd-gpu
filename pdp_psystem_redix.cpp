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

#include "pdp_psystem_redix.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

/* Random generator of a PDP P System */
PDP_Psystem_REDIX::PDP_Psystem_REDIX(PDP_Psystem_source * source) {
	unsigned int rule_ini=0;
	unsigned int lhs_ini=0;

	//if (options==NULL || !source->start()) {
	if (!source->start()) {
		cerr<< "Error reading from source. Halting Excecution" << endl;
		exit(1);
	}

	options=source->options;

	structures = new struct _structures;

	/* Create empty data for ruleblocks */
	structures->ruleblock_size = options->num_rule_blocks+options->num_blocks_env;
	structures->ruleblock.lhs_idx = new LHS_IDX [structures->ruleblock_size+1];
	structures->ruleblock.rule_idx = new RULE_IDX [structures->ruleblock_size+1];
	structures->ruleblock.membrane = new MEMBRANE [structures->ruleblock_size];

	/* For each ruleblock in Pi, calculate number of rules and lhs */
	source->pi_loop_rule_blocks();
	unsigned int block=0;

	if (options->num_rule_blocks>0)
	do {
		unsigned int rule_length = source->pi_loop_rules();
		unsigned int lhs_length = source->pi_loop_lhs();
		unsigned int membr = source->pi_lhs_membrane();

		/* Choose number of rules in the block */
		structures->ruleblock.rule_idx[block] = rule_ini;
		rule_ini += rule_length;

		/* Choose active membrane */
		structures->ruleblock.membrane[block] = COMPOSE_MEMBRANE(membr,source->pi_lhs_charge(),source->pi_rhs_charge());

		/* Choose length for left hand side */
		structures->ruleblock.lhs_idx[block] = lhs_ini;
		lhs_ini += lhs_length;

		block++;
	} while (source->pi_next_rule_block());

	/* For each ruleblock of communication, calculate number of rules */
	unsigned int env_rule_ini=rule_ini;
	source->env_loop_rule_blocks();
	
	if (options->num_blocks_env>0)
	do {
		unsigned int rule_length = source->env_loop_rules();
		//unsigned int lhs = source->env_get_object_lhs();
		unsigned int env = source->env_get_environment();

		/* Choose number of rules in the block */
		structures->ruleblock.rule_idx[block] = env_rule_ini;
		env_rule_ini += rule_length;

		/* Choose active membrane */
		structures->ruleblock.membrane[block] = COMPOSE_ENVIRONMENT(env);

		/* Choose length for left hand side */
		structures->ruleblock.lhs_idx[block] = lhs_ini;
		lhs_ini += 1;

		block++;
	} while (source->env_next_rule_block());

	// The last one stores a virtual index
	structures->ruleblock.lhs_idx[block] = lhs_ini;
	structures->ruleblock.rule_idx[block] = env_rule_ini;

	/* Create empty data for rules */
	structures->pi_rule_size=rule_ini;
	structures->env_rule_size=env_rule_ini-rule_ini;
	structures->rule.rhs_idx = new RHS_IDX [structures->pi_rule_size+structures->env_rule_size+1];

	/* Create empty data for LHS */
	structures->lhs_size=lhs_ini;
	structures->lhs.object = new OBJECT [structures->lhs_size];
	structures->lhs.mmultiplicity = new MULTIPLICITY [structures->lhs_size];
	structures->lhs.imultiplicity = new INV_MULTIPLICITY [structures->lhs_size];

	/* Create empty probabilities */
	structures->probability_size = structures->pi_rule_size*options->num_environments+structures->env_rule_size;
	structures->probability = new PROBABILITY [structures->probability_size];

	int rhs_ini=0;

	/* For each ruleblock in Pi: create rhs, probabilities and lhs */
	source->pi_loop_rule_blocks();
	block=0;

	if (options->num_rule_blocks>0)
	do {
		source->pi_loop_rules();
		unsigned int r=structures->ruleblock.rule_idx[block];

		/* Create RHS, init probabilities */
		do {
			structures->rule.rhs_idx[r] = rhs_ini;

			rhs_ini += source->pi_loop_rhs();

			float *prob = source->pi_rule_probabilities();

			for (int env=0; env<options->num_environments; env++) {
				structures->probability[env*structures->pi_rule_size+r]=prob[env];
			}

			r++;
		} while(source->pi_next_rule());
		
		/* Init LHS */
		OBJECT obj=0;
		MULTIPLICITY mult=0;

		unsigned int o=structures->ruleblock.lhs_idx[block];

		source->pi_loop_lhs();
		source->pi_lhs_loop_U();

		while (source->pi_lhs_next_object(obj,mult)) {
			structures->lhs.object[o] = obj;
			structures->lhs.mmultiplicity[o] = COMPOSE_MMULTIPLICITY(mult,source->pi_lhs_parent_membrane());
			structures->lhs.imultiplicity[o++] = 1.0f/mult;
		}

		source->pi_lhs_loop_V();

		while (source->pi_lhs_next_object(obj,mult)) {
			structures->lhs.object[o] = obj;
			structures->lhs.mmultiplicity[o] = COMPOSE_MMULTIPLICITY(mult,source->pi_lhs_membrane());
			structures->lhs.imultiplicity[o++] = 1.0f/mult;
		}

		block++;
	} while (source->pi_next_rule_block());

	/* For each ruleblock in communication, create RHS and prob */
	source->env_loop_rule_blocks();

	if (options->num_blocks_env>0)
	do {
		source->env_loop_rules();
		unsigned int r=structures->ruleblock.rule_idx[block];

		/* Adding lhs */
		unsigned int o=structures->ruleblock.lhs_idx[block];
		structures->lhs.object[o] = source->env_get_object_lhs();
		structures->lhs.mmultiplicity[o] = COMPOSE_MMULTIPLICITY(1,0);//GET_ENVIRONMENT(structures->ruleblock.membrane[block]);
		structures->lhs.imultiplicity[o] = 1.0f;

		/* Create RHS, init probabilities */
		do {

			structures->rule.rhs_idx[r] = rhs_ini;
			rhs_ini += source->env_loop_rhs();

			float prob = source->env_get_probability();

			structures->probability[options->num_environments*structures->pi_rule_size+(r-structures->pi_rule_size)]=prob;

			r++;
		}while(source->env_next_rule());
		block++;

	} while (source->env_next_rule_block());

	structures->rule.rhs_idx[structures->pi_rule_size+structures->env_rule_size] = rhs_ini;

	structures->rhs_size = rhs_ini;
	structures->rhs.object = new OBJECT [structures->rhs_size];
	structures->rhs.mmultiplicity = new MULTIPLICITY [structures->rhs_size];

	/* For each ruleblock in Pi, initiate RHS */
	source->pi_loop_rule_blocks();
	block=0;

	if (options->num_rule_blocks>0)
	do {
		source->pi_loop_rules();
		/* For each rule in the block */
		unsigned int r=structures->ruleblock.rule_idx[block];

		do {
			/* Initialize RHS */
			source->pi_loop_rhs();
				source->pi_rhs_loop_U();
				int o=structures->rule.rhs_idx[r];
				unsigned int obj=0,mult=0;

				while (source->pi_rhs_next_object(obj,mult)) {
					structures->rhs.object[o] = obj;
					structures->rhs.mmultiplicity[o++] = COMPOSE_MMULTIPLICITY(mult,source->pi_lhs_parent_membrane());
				}

				source->pi_rhs_loop_V();

				while (source->pi_rhs_next_object(obj,mult)) {
					structures->rhs.object[o] = obj;
					structures->rhs.mmultiplicity[o++] = COMPOSE_MMULTIPLICITY(mult,source->pi_lhs_membrane());
				}

			r++;
			} while (source->pi_next_rule());

		block++;
	} while (source->pi_next_rule_block());

	/* For each ruleblock for communication, initiate RHS */
	/* The wonderful thing of this solution must be improved */
	source->env_loop_rule_blocks();

	if (options->num_blocks_env>0)
	do {
		source->env_loop_rules();
		/* For each rule in the block */
		unsigned int r=structures->ruleblock.rule_idx[block];

		do {
			/* Initialize RHS */
			source->env_loop_rhs();

			int o=structures->rule.rhs_idx[r];
			unsigned int obj=0,dest_env=0;

			while (source->env_next_object(obj,dest_env)) {
				structures->rhs.object[o] = obj;
				structures->rhs.mmultiplicity[o++] = dest_env;
			}

			r++;
		} while (source->env_next_rule());

		block++;
	} while (source->env_next_rule_block());

	/* Initialize the initial configuration */

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
		} while (source->conf_next_membrane());
		env++;
	} while (source->conf_next_environment());

	structures->stringids.id_objects=source->get_objects_ids();
	structures->stringids.id_membranes=source->get_membranes_ids();
	structures->stringids.id_environments=source->get_environments_ids();
	
	/* Finally, process end, and printing system */
	print();
}

void PDP_Psystem_REDIX::print() {
	//int bsize = options->num_parallel_simulations*options->num_environments;

	unsigned int brmem = (structures->ruleblock_size) * (sizeof(LHS_IDX)+sizeof(RULE_IDX)+sizeof(MEMBRANE));
	unsigned int lhsmem = structures->lhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY));
	unsigned int rmem = (structures->pi_rule_size+structures->env_rule_size) * (sizeof(RHS_IDX));
	unsigned int pmem = structures->probability_size * sizeof(float);
	unsigned int rhsmem = structures->rhs_size * (sizeof(OBJECT)+sizeof(MULTIPLICITY));
	unsigned int multmem = structures->configuration.multiset_size * sizeof(MULTIPLICITY);
	unsigned int mbmem = structures->configuration.membrane_size * sizeof(CHARGE);

	options->mem = brmem+lhsmem+rmem+pmem+rhsmem+multmem+mbmem;

	if (options->verbose > 1) {
		cout << "=> The P system has been successfully initiated" << endl << endl << "--------------------------" << endl ;

		cout << "==> SIZES: " << endl;

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


		if (options->verbose>4 && structures->stringids.id_objects) {
			cout <<"==> ALPHABET: " << options->num_objects << endl << endl;
			for (int obj=0; obj < options->num_objects; obj++) {
				cout << structures->stringids.id_objects[obj] << " = ID " << obj << endl;
			}
			cout << endl << "---" << endl << endl;
		}

		if (options->verbose > 4 ) {

		cout << "==> RULE BLOCKS: " << structures->ruleblock_size << endl << endl ;

		for (int block=0; block < options->num_rule_blocks; block++) {
			cout << "Block " << block << endl;
			cout << "\t Info: " << endl;
			cout << "\t\t*rule_idx = " << structures->ruleblock.rule_idx[block] << endl;
			cout << "\t\t*lhs_idx = " << structures->ruleblock.lhs_idx[block] << endl;
			cout << "\t\t*membrane = " << GET_MEMBRANE(structures->ruleblock.membrane[block]) << endl;
			cout << "\t\t*lhs_charge = " << GET_ALPHA(structures->ruleblock.membrane[block]) << endl;
			cout << "\t\t*rhs_charge = " << GET_BETA(structures->ruleblock.membrane[block]) << endl << endl;

			cout << "\t Left-hand side = " ;
			for (unsigned int j=structures->ruleblock.lhs_idx[block]; j<structures->ruleblock.lhs_idx[block+1]; j++){
				cout << "<obj=";
				if (structures->stringids.id_objects)
					cout << structures->stringids.id_objects[structures->lhs.object[j]] << "|";
				cout << structures->lhs.object[j] << ",mult=" << GET_MULTIPLICITY(structures->lhs.mmultiplicity[j]) << ",membr=" << GET_MEMBR(structures->lhs.mmultiplicity[j]) << ">, ";
			}
			cout << endl << endl;

			cout << "\t Rules: " << endl;
			for (unsigned int j=structures->ruleblock.rule_idx[block]; j<structures->ruleblock.rule_idx[block+1]; j++){
				cout << "\t\t Rule " << j << ":" << endl;
				cout << "\t\t\t Probabilities (by envs): " ;

				for (int k=0; k<options->num_environments; k++) {
					cout << structures->probability[k*structures->pi_rule_size+j] << ", ";
				}
			
				cout << endl << "\t\t\t Right-hand side: ";
			
				for (unsigned int k=structures->rule.rhs_idx[j]; k<structures->rule.rhs_idx[j+1]; k++) {
					cout << "<obj=";
					if (structures->stringids.id_objects)
						cout << structures->stringids.id_objects[structures->rhs.object[k]] << "|";
					cout << structures->rhs.object[k] << ",mult=" << GET_MULTIPLICITY(structures->rhs.mmultiplicity[k]) << ",membr=" << GET_MEMBR(structures->rhs.mmultiplicity[k]) << ">, ";
				}
				cout << endl;
			}
			cout << endl << "---" << endl << endl;
		}
	
		for (int block = options->num_rule_blocks; block < options->num_blocks_env+options->num_rule_blocks; block++) {
			cout << "Block " << block << endl;
			cout << "\t Info: " << endl;
			cout << "\t\t*rule_idx = " << structures->ruleblock.rule_idx[block] << endl;
			cout << "\t\t*lhs_idx = " << structures->ruleblock.lhs_idx[block] << endl;
			cout << "\t\t*environment = " << GET_ENVIRONMENT(structures->ruleblock.membrane[block]) << endl << endl;

			cout << "\t Left-hand side = " ;
			for (unsigned int j=structures->ruleblock.lhs_idx[block]; j<structures->ruleblock.lhs_idx[block+1]; j++){
				cout << "<obj=";
				if (structures->stringids.id_objects)
					cout << structures->stringids.id_objects[structures->lhs.object[j]] << "|";
				cout  << structures->lhs.object[j] << ",mult=" << GET_MULTIPLICITY(structures->lhs.mmultiplicity[j]) << ",membr=" << GET_MEMBR(structures->lhs.mmultiplicity[j]) << ">, ";
			}
			cout << endl << endl;


			cout << "\t Rules: " << endl;
			for (unsigned int j=structures->ruleblock.rule_idx[block]; j<structures->ruleblock.rule_idx[block+1]; j++){
				cout << "\t\t Rule " << j << ":" << endl;
				cout << "\t\t\t Probability: " << structures->probability[options->num_environments*structures->pi_rule_size+(j-structures->pi_rule_size)];

				cout << endl << "\t\t\t Right-hand side: ";

				for (unsigned int k=structures->rule.rhs_idx[j]; k<structures->rule.rhs_idx[j+1]; k++) {
					cout << "<obj=";
					if (structures->stringids.id_objects)
						cout << structures->stringids.id_objects[structures->rhs.object[k]] << "|";
					cout << structures->rhs.object[k] << ",env=" << structures->rhs.mmultiplicity[k] << ">, ";
				}
				cout << endl;
			}
			cout << endl << "---" << endl << endl;
		}

		cout << "--------------------------" << endl << "==> INITIAL CONFIGURATION" << endl << endl;

		cout << "Multiset" << endl;
	
		for (int env=0; env<options->num_environments; env++) {
			cout << "Environment " << env << ":";

			for (int memb=0; memb<options->num_membranes; memb++) {
				cout << endl << "\t\tMembrane " << memb << ": ";
				for (int obj=0; obj<options->num_objects; obj++) {
					if (structures->stringids.id_objects)
						cout << structures->stringids.id_objects[obj];
					else
						cout << "Obj " << obj;
					cout << "*" << structures->configuration.multiset[env*options->num_objects*options->num_membranes+memb*options->num_objects+obj] << ", ";
				}
			}

			cout << endl;
		}

		cout << endl << endl;
		} // Verbose > 3

        cout << "--------------------------" << endl << "=> MEMORY REQUIREMENTS" << endl << endl;

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
        cout << "==> TOTAL: " << memb << " (" << memb/div << " " << unit[count] << "B)" << endl << endl;

        cout << "Printing system finished." << endl << endl;


	} // Verbose 1
}



/*****************************************/
/* PDP_Psystem_redix_out_std definitions */
/*****************************************/


PDP_Psystem_redix_out_std::PDP_Psystem_redix_out_std(PDP_Psystem_REDIX* pdpsys) {
	pdp=pdpsys;
	this->structures=pdpsys->structures;
	this->options=pdpsys->options;
	
	/* Auxiliary functions for debugging */
	/* In case we need to print objects */
	if (options->verbose>3) {
            if (structures->stringids.id_objects==NULL) {
                structures->stringids.id_objects=new char*[options->num_objects];
                for (int i=0; i<options->num_objects; i++) {
                    structures->stringids.id_objects[i]=new char[12];
                    sprintf(structures->stringids.id_objects[i],"obj_%d",i);
                }
            }
            if (structures->stringids.id_membranes==NULL) {
                structures->stringids.id_membranes=new char*[options->num_membranes];
                for (int i=0; i<options->num_membranes; i++) {
                    structures->stringids.id_membranes[i]=new char[8];
                    sprintf(structures->stringids.id_membranes[i],"%d",i);
                }
            }
            if (structures->stringids.id_environments==NULL) {
                structures->stringids.id_environments=new char*[options->num_environments];
                for (int i=0; i<options->num_environments; i++) {
                    structures->stringids.id_environments[i]=new char[8];
                    sprintf(structures->stringids.id_environments[i],"%d",i);
                }	
            }
        }
        
        esize=options->num_objects*options->num_membranes;
	msize=options->num_objects;
	bpsize=options->num_rule_blocks;
	besize=options->num_blocks_env+options->num_rule_blocks;
	rpsize=structures->pi_rule_size;
	resize=structures->pi_rule_size+structures->env_rule_size;
	//asize=((besize>>ABV_LOG_WORD_SIZE)+1);
}

PDP_Psystem_redix_out_std::~PDP_Psystem_redix_out_std(){
    // TODO: Delete ids for objects, membranes and envs
}


    /* Verbosity 1 */
void PDP_Psystem_redix_out_std::print_simulator_process(const char * message) {
	if (will_print_simulator_process())
	    cout << message<<endl;
}
    
    /* Verbosity 2 */
void PDP_Psystem_redix_out_std::print_simulation(int simulation) {
	if (will_print_simulation()) {
	    cout << "==> Simulation " << simulation << endl;
	}
}
    
void PDP_Psystem_redix_out_std::print_step(int step) {
	if (will_print_step()) {
	    cout << "====> Step  " << step << endl;
	}
}
    
void PDP_Psystem_redix_out_std::print_sim_step(int simulation, int step) {
	if (will_print_sim_step()) {
	    cout << "==> Simulation " << simulation << ", step  " << step << endl;
	}
}
    
void PDP_Psystem_redix_out_std::print_sim_thread(int simulation, int stid) {
	if (will_print_sim_thread()) {
	    cout << "==> Simulation " << simulation << ", executed by thread " << stid << endl;
	}
}
    
void PDP_Psystem_redix_out_std::print_sim_range (int inisim, int endsim) {
	if (will_print_sim_range()) {
	    cout << "==> Simulations from " << inisim << " to " << endsim << endl;
	}
}
    
    /* Verbosity 3 */
void PDP_Psystem_redix_out_std::print_dcba_phase(int phase) {
	if (will_print_dcba_phase()) {
	    cout << "======> Phase  " << phase << endl;
	}
}
    
    /* Verbosity 4 */
void PDP_Psystem_redix_out_std::print_configuration(int sim) {

	if (!will_print_configuration()) return;

	cout << "========> Configuration " << endl;

	for (int env=0; env<options->num_environments; env++) {
		cout << "==========> Environment " << structures->stringids.id_environments[env] << ":";

		for (int memb=0; memb<options->num_membranes; memb++) {
			cout << endl << "============> Membrane " << structures->stringids.id_membranes[memb] << "(charge " << CHARGE_TO_CHAR(structures->configuration.membrane[sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+(memb)]) << "): ";
			for (int obj=0; obj<options->num_objects; obj++) {
				if (structures->stringids.id_objects)
					cout << structures->stringids.id_objects[obj];
				else
					cout << "Obj " << obj;
				cout //<< structures->stringids.id_objects[obj]
					<< "*" << structures->configuration.multiset[sim*options->num_environments*esize+env*esize+memb*msize+obj] << ", ";
			}
		}
		cout << endl;
	}
}
    
    /* Verbosity 5 */
void PDP_Psystem_redix_out_std::print_block_selection(int sim) {
	if (!will_print_block_selection()) return;

	cout << "========> Blocks applications " << endl;

	for (int env=0; env<options->num_environments; env++) {
		cout << "==========> Environment " << structures->stringids.id_environments[env] << ": ";
		for (unsigned int block=0; block<besize; block++) {
			cout << "B" << block << "*" << structures->nb[sim*options->num_environments*besize+env*besize+block] << ", ";
		}
		cout << endl;
	}
}
    
void PDP_Psystem_redix_out_std::print_rule_selection(int sim){
	if (!will_print_rule_selection()) return;
	
	cout << "========> Rule applications " << endl;

	for (int env=0; env<options->num_environments; env++) {
		cout << "==========> Environment " << structures->stringids.id_environments[env] << ": "<<endl;
		for (unsigned int block=0; block<besize; block++) {
			int rule_ini=structures->ruleblock.rule_idx[block];
			int rule_end=structures->ruleblock.rule_idx[block+1];

			unsigned int N=0;

			if (block<bpsize)
				N=structures->nb[sim*options->num_environments*besize+env*besize+block];
			else if (env==GET_ENVIRONMENT(structures->ruleblock.membrane[block]))
				N=structures->nb[sim*options->num_environments*besize+env*besize+block];

			cout << "============> Block " << block << ", N=" << N << ":= ";

			int check_subtotal=0;

			for (unsigned int r = rule_ini; r < rule_end; r++) {
				float p=0.0;
				unsigned int val=0;

				if (r>=rpsize)
					p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
				else 	p=structures->probability[env*rpsize+r];

				if (r<rpsize)
					val=structures->nr[sim*(options->num_environments*rpsize+(resize-rpsize))+env*rpsize+r];
				else
					val=structures->nr[sim*(options->num_environments*rpsize+(resize-rpsize))+options->num_environments*rpsize+(r-rpsize)];

				cout << "R" << r-rule_ini << "[" << r << "]:(p=" << p << ",n=" << val <<"), ";

				check_subtotal += val;
			}

			if (N != check_subtotal)
				cout << "ERROR: execution of the block is not maximal (block selection=" << N << " vs total rules applications=" << check_subtotal<<")";
			cout << endl;
		}
	}
}
    
    /* Verbosity 6 */
void PDP_Psystem_redix_out_std::print_temporal_configuration(int sim) {
	if (!will_print_temporal_configuration()) return;
	
	cout << "========> Temporal configuration:" << endl;
	print_configuration(sim);
}
