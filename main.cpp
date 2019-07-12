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

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pdp_psystem.h"
#include "pdp_psystem_source_random.h"
#include "pdp_psystem_source_binary.h"
#include "pdp_psystem_output_binary.h"
#include "pdp_psystem_output_csv.h"

#include "simulator.h"
#include "simulator_seq_table.h"
#include "simulator_seq_dir.h"
#include "simulator_omp_dir.h"
#include "simulator_gpu_dir.h"
#include "simulator_omp_redir.h"

#include <iostream>
#include <time.h>
#include <string>
#include <timestat.h>

using namespace std;

int main (int argc, char* argv[]) {
	//Structures* structures = new Structures;
	Options options = new struct _options;
	char c='\0';
	int mode=0;
	int par=0;
	int example=0;
	bool accu=true;
	bool random_system=true;
	int outputtype=0;
	string input_file="small.bin";

	/* Default */
	options->num_rule_blocks = 384;
	options->num_blocks_env = 128;
	options->max_num_rules = 3;
	options->num_simulations = 3;//10;
	options->num_parallel_simulations = 1;
	options->num_environments = 4;
	options->num_objects = 7024;
	options->num_membranes=3;
	options->max_lhs=4;
	options->max_rhs=4;
	options->time=1;
	options->cycles=1;
	options->accuracy=2;
	options->verbose=0;
	options->debug = false;
	
	// TODO: use getopt_long function to support arguments of type "--blocks"
	while ((c = getopt (argc, argv, "Rb:r:c:l:e:O:o:q:a:t:s:v:f:g:I:M:FX:h")) != -1) {
		switch (c) {
		/* For randomly generated system */
		case 'R':
			random_system=true;
			break;
		case 'b':
			options->num_rule_blocks = atoi(optarg);
			options->num_blocks_env = options->num_rule_blocks/4;
			break;
		case 'r':
			options->max_num_rules = atoi(optarg);
			break;
		case 'l':
			options->max_lhs=options->max_rhs=atoi(optarg);
			break;
		case 'e':
			options->num_environments = atoi(optarg);
			break;
		case 'o':
			options->num_objects = atoi(optarg);
			break;
		case 'q':
			options->num_membranes = atoi(optarg);
			break;
		/* Options for the desired simulation */
		case 'a':
			options->accuracy = atoi(optarg);
			break;
		case 't':
			options->time = atoi(optarg);
			break;
		case 'c':
			options->cycles = atoi(optarg);
			break;
		case 's':
			options->num_simulations = atoi(optarg);
			break;		
		case 'v':
			options->verbose = atoi(optarg);
			break;
		case 'g':
			options->debug = atoi(optarg);
			break;
		case 'f':
			random_system=false;
			input_file = optarg;
			break;
		case 'O':
			outputtype=atoi(optarg);
			break;
		case 'I':
			mode=atoi(optarg);
			break;
			/*if (mode >= 0 && mode <= 5)
				break;
			cout << "Incorrect mode: " << mode << endl;*/
		case 'M':
			par=atoi(optarg);
			break;
		case 'F':
			accu=false;
			break;
		/* Preconfigured examples */
		case 'X':
			example = atoi(optarg);
			break;
		case 'h':
		case '?':
			cout << "Copyright (C) 2015, RGNC, University of Seville" << endl <<
				"This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions" << endl << endl;
			cout << "A simulator for Population Dynamics P systems using DCBA. Version 1.0 beta" << endl<<endl;
			cout << "Usage: ./abcd <options>" <<endl<<
				        "Options for the simulation:"<<endl<<
					" -f: input file in binary format (revision 16-09-2014 supported)" << endl <<
					" -s: number of simulations"<< endl <<
					" -a: accuracy in the algorithms" << endl <<
					" -t: time steps" << endl <<
					" -c: steps per cycle of model" << endl <<
					" -I: select the implementation: " << endl <<
					"     -> 0 for OpenMP simulator (default). For a multicore parallel execution, execute 'export OMP_NUM_THREADS=X' on the terminal (X should be the number of processors - 1 for best performance)," << endl <<
					"     -> 1 for GPU simulator. It requires to have a CUDA capable GPU on the system, with compute capability 1.2 or later." << endl <<
					"     -> (deprecated simulators): 10 for table based simulator, 11 for sequential simulator, 12 for parallel OpenMP simulator" << endl <<
					" -M: select an execution mode:" << endl <<
					"     -> if I=1, M sets the behaviour of the GPU simulator: 0 runs GPU simulator (default), 1 (profiling) runs also the CPU gold code, 2 (profiling) runs with CPU and phase2 basic kernel" << endl <<
					"     -> if I=12, M is the parallelism level: 0 for environments, 1 for simulations, 2 for hybrid-2s and 3 for hybrid-2e" << endl <<
					" -O: output format type" << endl <<
					"     -> 0 for csv (comma-separated values) file (default)" << endl <<
					"     -> 1 for binary file (experimental, not working yet)" << endl <<
					" -F: unset accurate mode (demotes to float row additions)" << endl <<
					" -v: verbosity level (from 0 to 6), as follows:"<< endl<<
					"     -> 0 (default): just show execution time and required simulation memory" << endl <<
					"     -> 1: verbosity 0 + simulator stage" << endl <<
					"     -> 2: verbosity 1 + simulations and transitons steps process" << endl <<
					"     -> 3: verbosity 2 + DCBA phases per transition step" << endl <<
					"     -> 4: verbosity 3 + print PDP system configuration after each transition step" << endl <<
					"     -> 5: verbosity 4 + print blocks and rules selection after each DCBA phase" << endl <<
					"     -> 6: verbosity 5 + print PDP system configuration after each DCBA phase" << endl <<
					"Special options for the random system generator:" << endl <<
					" -R: activate the random system generator (by default, in case a input file is not provided)" << endl <<
					" -o: number of objects" << endl <<
					" -q: number of membranes" << endl <<
					" -b: number of rule blocks" << endl <<
					" -r: maximum number of rules per block" << endl <<
					" -l: maximum number of objects in the LHS/RHS" << endl <<
					" -e: number of environments" << endl <<
					" -X: select one of the prefixed examples: 0 (default: configured through parameters), 1 (small), 2 (medium), 3 (large)" << endl <<
					"Finally, the simulator always outputs the total simulation time in ms and the total memory required in Bytes."<< endl;
			return 0;
		}
	}
	
	if (mode==20) {
		cout << "Glad to see you master, here you have your sample binary file..." << endl;
		PDP_Psystem_source_binary* source=new PDP_Psystem_source_binary(options);
		source->write_test_binary();
		delete source;
		cout << "Wrote the binary file..."<<endl;
		return 0;
	}	

	PDP_Psystem_source* source=NULL;
	PDP_Psystem_output* output=NULL;
	PDP_Psystem* PDPps=NULL;
	Simulator *simulator=NULL;
	
	/* Read the input */	
	if (!random_system) {
	    source=new PDP_Psystem_source_binary(input_file.c_str(),options);
	}
	else {
		// Never output in a random system
		options->cycles=options->time+2;
		input_file="";

		// Pre-configured random systems
	    if (example == 3) {
		    options->num_rule_blocks = 100000;
		    options->num_blocks_env = options->num_rule_blocks/4;
		    options->max_num_rules = 5;
		    options->num_simulations = 50;
		    options->num_parallel_simulations = 1;
		    options->num_parallel_environments = 1;
		    options->num_environments= 15;
		    options->num_objects=7000;
		    options->num_membranes=4;
		    options->max_lhs=5;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=1;
		    //options->verbose=1;
	    }
	    else if (example == 2) {
		    options->num_rule_blocks = 384;
		    options->num_blocks_env = 128;
		    options->max_num_rules = 3;
		    options->num_simulations = 3;//10;
		    options->num_parallel_simulations = 1;
		    options->num_environments = 4;
		    options->num_objects = 1024;
		    options->num_membranes=3;
		    options->max_lhs=4;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=2;
		    //options->verbose=1;
	    }
	    else if (example == 1) {
		    options->num_rule_blocks = 4;
		    options->num_blocks_env = options->num_rule_blocks/4;
		    options->max_num_rules = 3;
		    options->num_simulations = 1;
		    options->num_parallel_simulations = 1;
		    options->num_environments= 2;
		    options->num_objects=6;
		    options->num_membranes=2;
		    options->max_lhs=5;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=1;
		    //options->verbose=1;
	    }
	    
	    source=new PDP_Psystem_source_random(options);
	}


	/* Initiate the simulator */
	if (mode==0) {
		PDPps = new PDP_Psystem_REDIX(source);

		if (outputtype==0)
			output = new PDP_Psystem_output_csv(input_file.c_str(),options);
		else
			output = new PDP_Psystem_output_binary(input_file.c_str(),options);

		simulator = new Simulator_omp_redir((PDP_Psystem_REDIX*)PDPps,options->num_simulations,accu,output);
	}
	else if (mode==1) {
		PDPps = new PDP_Psystem_REDIX(source);

		if (outputtype==0)
			output = new PDP_Psystem_output_csv(input_file.c_str(),options);
		else
			output = new PDP_Psystem_output_binary(input_file.c_str(),options);

		simulator = new Simulator_gpu_dir((PDP_Psystem_REDIX*)PDPps,par,accu,output);
	}
	
	/* Deprecated simulators */
	else if (mode == 10) {
		/* Create the structures for the simulator */
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_seq_table((PDP_Psystem_SAB*)PDPps);
	}
	else if (mode==11) {
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_seq_dir((PDP_Psystem_SAB*)PDPps);
	}
	else if (mode==12) {
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_omp_dir((PDP_Psystem_SAB*)PDPps,par);
	}	
	

	delete source;
	
	start_timer();
	
	/* Run the main loop of the simulator */
	if (simulator && simulator->run()){
		if (options->verbose>0) cout << endl << "Successfully finished." << endl;
	}
	else {
		if (!simulator)
			cout << "No simulator associated to this mode, use -h option" << endl;
		else
			cout << "Problems encountered in the simulator. Halted with errors" << endl;
	}

	double d=end_timer();
	cout << d << " " << options->mem << endl;
	
	delete PDPps;

	delete output;

	delete simulator;

	return 0;

}
