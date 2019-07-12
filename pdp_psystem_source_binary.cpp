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

#include "pdp_psystem_source_binary.h"
#include <string.h>

using namespace std;

/**********************************************************/
/* CONSTRUCTORS:                                          */
/*                                                        */
/* Binary reader of a PDP System: maravillosos procedures */
/**********************************************************/

PDP_Psystem_source_binary::PDP_Psystem_source_binary(const char * filename, Options options) {
	this->options=options;

	// Next is not used yet
	order[0]=GENERAL_INFO;
	order[1]=BLOCKS_PI;
	order[2]=BLOCKS_ENV;
	order[3]=CONFIGURATION;
	
	// INITIALIZE POINTERS
	delete_id_objects=delete_id_environments=delete_id_membranes=true;
	id_objects=NULL;
	id_environments=NULL;
	id_membranes=NULL;	
	membrane_hierarchy=NULL;	
	lengthU=NULL;
	lengthV=NULL;
	active_membrane=NULL;
	rules=NULL;
	lengthUp=NULL;
	lengthVp=NULL;
	obj_lhs=obj_lhs_m=obj_rhs=obj_rhs_m=NULL;
	prob=NULL;	
	block_precision=charge=NULL;	
	env_lengthU=NULL;	
	ini_multiset=NULL;
	ini_info=NULL;
	ini_charge=NULL;

	if (options->verbose>0)
		cout << "[1] STARTING READING THE PDP SYSTEM FROM " << filename << endl;
	
	/************************************/
	/* Open the input file as binary */
	is.open (filename, ios::binary);

	/* If couldn't open it, error*/
	check_file_error(is.fail(), string("Couldn't open binary file ") + filename, is);
}

PDP_Psystem_source_binary::PDP_Psystem_source_binary(Options options) {
	this->options=options;
}

PDP_Psystem_source_binary::~PDP_Psystem_source_binary() {

	if (!is.fail() && is.good() && is.is_open())
		is.close();
	
	if (delete_id_objects&&id_objects) {
	    for (int i=0;i<options->num_objects;i++) {
		if (id_objects[i]) delete [] id_objects[i];
	    }
	    delete[]id_objects;
	}
	if (delete_id_environments&&id_environments) {
	    for (int i=0;i<options->num_environments;i++) {
		if (id_environments[i]) delete [] id_environments[i];
	    }
	    delete[]id_environments;
	}
	if (delete_id_membranes&&id_membranes) {
	    for (int i=0;i<options->num_membranes;i++) {
		if (id_membranes[i]) delete [] id_membranes[i];
	    }
	    delete[]id_membranes;	    
	}	
	if (membrane_hierarchy) delete[]membrane_hierarchy;	
	if (lengthU) delete []lengthU;
	if (lengthV) delete []lengthV;
	if (lengthUp) delete []lengthUp;
	if (lengthVp) delete []lengthVp;
	if (active_membrane) delete []active_membrane;
	if (rules) delete []rules;
	if (prob) delete []prob;
	if (obj_lhs) delete []obj_lhs;
	if (obj_rhs) delete []obj_rhs;
	if (obj_lhs_m) delete []obj_lhs_m;
	if (obj_rhs_m) delete []obj_rhs_m;
	if (block_precision) delete []block_precision;
	if (charge) delete []charge;	
	if (env_lengthU) delete []env_lengthU;	
	if (ini_info) delete []ini_info;
	if (ini_multiset) {
	    for (int i=0;i<options->num_membranes*options->num_environments;i++) {
	    	if (ini_multiset[i]) delete [] ini_multiset[i];
	    }
	    delete [] ini_multiset;
	}
	if (ini_charge) delete []ini_charge;
}


/********************************************/
/* AUXILIARY FUNCTIONS:                     */
/*                                          */
/* Read each section of a binary file       */
/********************************************/

/* Auxiliary function:
    If condition is true, there is an error: close the binary file, 
    print the message and throws an exception. */
void PDP_Psystem_source_binary::check_file_exception(bool condition, string message) {
    if (condition) {
	//is.close();
	throw FileException("Error while reading file: "+message);
	//cerr << "Error while reading file: " << message << endl;	
    }
}

bool PDP_Psystem_source_binary::read_header() {
#ifdef BIN_DEBUG
    cout << "READING HEAD:" << endl;
#endif
    
    	/************************/
	/* Read the file header */
	unsigned int recbytes = read_bytes(is,HEADER_SIZE);

	if ((recbytes & HEADER_BITMASK) != HEADER) {
		cout << "Error, the file is not a good binary file" << endl;
		return false;
	}
	else if ((recbytes & (HEADER_VERSION_BITMASK|HEADER_VARIANT_BITMASK)) != 0X21) {
		cout << "Error, the file version is not compatible" << endl;
		return false;	
	}

#ifdef BIN_DEBUG	
	cout << "Using binary file version " << (recbytes & HEADER_VARIANT_BITMASK) << endl;
#endif
	/************************/
	
	/***************************/
	/* Read the file subheader */
	recbytes = read_bytes(is,SUBHEADER_SIZE);
	
#ifdef BIN_DEBUG	
	cout << "recbytes=" << recbytes << endl;
#endif	
	
	precision.objects = (1 << ((recbytes&0xC00000)>>22));
	precision.environments = (1 << ((recbytes&0x300000)>>20));
	precision.membranes = (1 << ((recbytes&0xC0000)>>18));
	precision.pi_rules = (1 << ((recbytes&0x30000)>>16));
	precision.env_rules = (1 << ((recbytes&0xC000)>>14));
	precision.multisets = (1 << ((recbytes&0x3000)>>12));
	precision.ini_num_objects = (1 << ((recbytes&0xC00)>>10));
	precision.ini_mult = (1 << ((recbytes&0x300)>>8));
	
	id_strings.objects = ((recbytes & 0x4)>>2);
	id_strings.environments = ((recbytes & 0x2)>>1);
	id_strings.membranes = (recbytes & 0x1);
	
#ifdef BIN_DEBUG	
	cout << "#objects="<<precision.objects << endl;
	cout << "#environments="<<precision.environments << endl;
	cout << "#membranes="<<precision.membranes << endl;
	cout << "#pi_rules="<<precision.pi_rules << endl;
	cout << "#env_rules="<<precision.env_rules << endl;
	cout << "#multisets="<<precision.multisets << endl;
	cout << "#mult_rules="<<precision.ini_num_objects << endl;
	cout << "#mult_multisets="<<precision.ini_mult << endl;
	cout << "#showobjects="<< id_strings.objects << endl;
	cout << "#showenvironments="<< id_strings.environments << endl;
	cout << "#showmembranes="<< id_strings.membranes << endl;
	/***************************/
#endif	
	return true;
}

bool PDP_Psystem_source_binary::read_global_sizes() {
#ifdef BIN_DEBUG
    cout << "READING GLOBAL SIZES:" <<endl;
#endif
    	char text_buffer [MAX_TEXT_SIZE];
	
    	/*********************/
	/* Read global sizes */
	number_objects=read_bytes(is,precision.objects);
	
	check_file_exception(number_objects==0,"No objects defined");
	
	if (id_strings.objects) {
		id_objects=new char*[number_objects];
		for (int i=0;i<number_objects;i++){
			int len=read_text(is,text_buffer,MAX_TEXT_SIZE,END_OF_TEXT);
			check_file_exception(len<=0, "Text size is not correct.");
			id_objects[i]=new char[len+1];
			strcpy(id_objects[i],text_buffer);
		}
	}
	else id_objects=NULL;
	
	number_environments=read_bytes(is,precision.environments);
		
	check_file_exception(number_environments==0,"No environments defined");
	
	if (id_strings.environments) {
		id_environments=new char*[number_environments];
		for (int i=0;i<number_environments;i++){
			int len=read_text(is,text_buffer,MAX_TEXT_SIZE,END_OF_TEXT);
			check_file_exception(len<=0, "Text size is not correct.");
			id_environments[i]=new char[len+1];
			strcpy(id_environments[i],text_buffer);
		}
	}
	else id_environments=NULL;
	
	number_membranes=read_bytes(is,precision.membranes);
		
	check_file_exception(number_membranes==0,"No membranes defined");
	
	membrane_hierarchy=new unsigned int[number_membranes];
		
	if (id_strings.membranes)
		id_membranes=new char*[number_membranes];
	else id_membranes=NULL;	
		
	for (int i=0;i<number_membranes;i++){
		unsigned int parent=read_bytes(is,precision.membranes);
		membrane_hierarchy[i]=parent;
		
		if (id_strings.membranes) {
			int len=read_text(is,text_buffer,MAX_TEXT_SIZE,END_OF_TEXT);
			check_file_exception(len<=0, "Text size is not correct.");
			id_membranes[i]=new char[len+1];
			strcpy(id_membranes[i],text_buffer);
		}
	}

	number_rule_blocks_pi=read_bytes(is,precision.pi_rules);
	
	number_rule_blocks_env=read_bytes(is,precision.env_rules);	
	
#ifdef BIN_DEBUG	
	cout << "number objects=" << number_objects << endl;
	if (id_strings.objects && id_objects!=NULL) {
		cout << "Object IDs:" << endl;
		for (int i=0;i<number_objects;i++) cout << id_objects[i] << ", ";
		cout << endl;
	}
	cout << "number membranes=" << number_membranes << endl << "Membrane hierarchy:" << endl;
	
	for (int i=0;i<number_membranes;i++) cout << "parent[" << i << "]=" << membrane_hierarchy[i] << endl;
	
	if (id_strings.membranes && id_membranes!=NULL) {
		cout << "Memb IDs:" << endl;
		for (int i=0;i<number_membranes;i++) cout << id_membranes[i] << ", ";
		cout << endl;
	}
	cout << "number environments=" << number_environments << endl;
	if (id_strings.environments && id_environments!=NULL) {
		cout << "Env IDs:" << endl;
		for (int i=0;i<number_environments;i++) cout << id_environments[i] << ", ";
		cout << endl;
	}
	cout << "number pi rules=" << number_rule_blocks_pi << endl;
	cout << "number env rules=" << number_rule_blocks_env << endl;
#endif
	
	options->num_blocks_env=number_rule_blocks_env;
	options->num_rule_blocks=number_rule_blocks_pi;
	options->num_objects=number_objects;
	options->num_membranes=number_membranes;
	options->num_environments=number_environments;	
	/*********************/
		
	/**********************************/
	/* Initialize aux data structures */
	lengthU= new short int[options->num_rule_blocks];

	lengthV= new short int[options->num_rule_blocks];

	active_membrane= new unsigned int[options->num_rule_blocks+options->num_blocks_env];
	
	block_precision= new char[options->num_rule_blocks+options->num_blocks_env];
	
	rules = new unsigned int[options->num_rule_blocks+options->num_blocks_env+1];
	
	charge = new char[options->num_rule_blocks];
	/**********************************/
	
	return true;
}

/* Required macros for extracting information from rule blocks information byte
 */

#define PREC_MULTIPLICITY(byte) (1<<(((byte)&0xC0)>>6))
#define PREC_NUM_OBJS_LHS(byte) (1<<(((byte)&0x20)>>5))
#define PREC_NUM_OBJS_RHS(byte) (1<<(((byte)&0x10)>>4))
#define PREC_NUM_RULES(byte) (1<<(((byte)&0xC)>>2))
#define DIF_PROB_ENV(byte) !(((byte)&0x2)>>1)
#define SHOW_PARENT_MEMBRANE(byte) ((byte)&0x1)
#define CHARGE_ALPHA(charge) ((charge&0xC0)>>6)
#define CHARGE_BETA(charge) ((charge&0x30)>>4)


bool PDP_Psystem_source_binary::read_inf_1() {
#ifdef BIN_DEBUG
    cout<< "ENTERING FOR INFORMATION 1"<<endl;
#endif
    
    unsigned int objects_in_lhs=0;
    number_rules_pi=number_rules_env=number_rules=0;
    prob_length=0;
    
    rules[0]=0;
    options->max_lhs=0;
    options->max_num_rules=0;
    
    for (unsigned int i=0;i<options->num_rule_blocks;i++) {
	int nr=0;
	block_precision[i]=read_bytes(is,BLOCK_INF_BYTE_SIZE);
	nr=read_bytes(is,PREC_NUM_RULES(block_precision[i]));
	rules[i+1]=rules[i]+nr;
	lengthU[i]=read_bytes(is,PREC_NUM_OBJS_LHS(block_precision[i]));
	active_membrane[i]=read_bytes(is,precision.membranes);
	if (SHOW_PARENT_MEMBRANE(block_precision[i]))
	    read_bytes(is,precision.membranes);
	charge[i]=read_bytes(is,CHARGE_SIZE);
	
	objects_in_lhs+=lengthU[i];
	number_rules_pi+=nr;
	//prob_length+=(DIF_PROB_ENV(block_precision[i]))?nr:nr*options->num_environments;
	prob_length+=nr*options->num_environments;
	// Update max lhs
	if (options->max_lhs<lengthU[i]) options->max_lhs=lengthU[i];
	if (options->max_num_rules<nr) options->max_num_rules=nr;

#ifdef BIN_DEBUG
	cout << "PI Block " << i << ":"<< endl;
	cout << "\tPrecision byte=" << hex << (short int) block_precision[i] << dec << endl;
	cout << "\tNo Rules="<<rules[i+1]-rules[i]<<endl;
	cout << "\tLength LHS="<<lengthU[i]<<endl;
	cout << "\tAMembrane="<<active_membrane[i]<<endl;
	cout << "\tCharges Alf="<<CHARGE_ALPHA(charge[i])<<endl;
	cout << "\tCharges Bet="<<CHARGE_BETA(charge[i])<<endl;
#endif
    }
    
    for (unsigned int i=0;i<options->num_blocks_env;i++) {
	int nr=0;
	unsigned int bid=i+options->num_rule_blocks;
	block_precision[bid]=read_bytes(is,BLOCK_INF_BYTE_SIZE);
	nr=read_bytes(is,PREC_NUM_RULES(block_precision[bid]));
	rules[bid+1]=nr+rules[bid];
	active_membrane[bid]=read_bytes(is,precision.environments);
	
	number_rules_env+=nr;
	//prob_length+=(DIF_PROB_ENV(block_precision[i]))?nr:nr*options->num_environments;
	prob_length+=nr;
	if (options->max_num_rules<nr) options->max_num_rules=nr;

#ifdef BIN_DEBUG	
	cout << "ENV Block " << i << " (" << bid <<"):"<< endl;
	cout << "\tNo Rules="<<nr<<endl;
	cout << "\tEnv="<<active_membrane[i]<<endl;
#endif
    }

    number_rules=number_rules_pi+number_rules_env;

#ifdef BIN_DEBUG    
    cout << "Total rules Pi="<<number_rules_pi<<endl;
    cout << "Total rules Env="<<number_rules_env<<endl;
    cout << "Total rules="<<number_rules<<endl;
    cout << "Total length probs="<<prob_length<<endl;    
#endif
    
    obj_lhs=new unsigned int[objects_in_lhs+options->num_blocks_env];
    obj_lhs_m=new unsigned int[objects_in_lhs];
    lengthUp=new short int[number_rules];
    lengthVp=new short int[number_rules];
    prob=new float[prob_length];
    
    num_obj_lhs_blocks_pi=objects_in_lhs;
    
    return true;
}

bool PDP_Psystem_source_binary::read_inf_2() {
#ifdef BIN_DEBUG
    cout<< "ENTERING FOR INFORMATION 2"<<endl;
#endif
    unsigned int objects_in_rhs=0;
    unsigned int object_pointer=0; // global pointer for objects
    options->max_rhs=0;
    
    for (unsigned int i=0;i<options->num_rule_blocks;i++) {
		for (int r=rules[i]; r<rules[i+1];r++) {

			lengthUp[r]=read_bytes(is,PREC_NUM_OBJS_RHS(block_precision[i]));
			objects_in_rhs+=lengthUp[r];

			if (DIF_PROB_ENV(block_precision[i])) {
			for (int e=0;e<options->num_environments;e++) {
				int pr=read_bytes(is,PROB_SIZE);
				prob[r*options->num_environments+e]=pr/PROB_PREC;
			}
			}
			else {
			int pr=read_bytes(is,PROB_SIZE);
			for (int e=0;e<options->num_environments;e++)
				prob[r*options->num_environments+e]=pr/PROB_PREC;
			}

			if (options->max_rhs<lengthUp[r]) options->max_rhs=lengthUp[r];
		}

		short int lhsl=lengthU[i];
		lengthU[i]=read_bytes(is,PREC_NUM_OBJS_LHS(block_precision[i]));

		for (int o=0; o<lengthU[i]; o++) {
			obj_lhs[object_pointer]=read_bytes(is,precision.objects);
			obj_lhs_m[object_pointer++]=read_bytes(is,PREC_MULTIPLICITY(block_precision[i]));//,precision.multisets);
		}

		lengthV[i]=read_bytes(is,PREC_NUM_OBJS_LHS(block_precision[i]));

		//cout << lengthU[i] << "+" << lengthV[i] <<"="<< lengthU[i]+lengthV[i] << "!=" << lhsl << endl;
	
		check_file_exception((lengthU[i]+lengthV[i])!=lhsl,"Different lengths of LHS in info 1 vs 2 ");

		for (int o=0; o<lengthV[i]; o++) {
			obj_lhs[object_pointer]=read_bytes(is,precision.objects);
			obj_lhs_m[object_pointer++]=read_bytes(is,PREC_MULTIPLICITY(block_precision[i]));//,precision.multisets);
		}
	
	#ifdef BIN_DEBUG
		cout << "PI Block " << i << ":"<< endl;
		for (int r=rules[i]; r<rules[i+1];r++)
			cout << "\tLength RHS="<<lengthUp[r]<<endl <<
				"\tProb=" << prob[r*options->num_environments]<<endl;
		cout << "\tLHS=";
		int o_it=object_pointer-lhsl;
		for (int o=0; o<lengthU[i]; o++) {
			cout << id_objects[obj_lhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_lhs_m[o_it] << " ";o_it++;
		}
		cout <<"[ ";
		for (int o=0; o<lengthV[i]; o++) {
			cout << id_objects[obj_lhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_lhs_m[o_it] << " ";o_it++;
		}
		cout <<"]^" << CHARGE_ALPHA(charge[i]) << "_" << active_membrane[i] << endl;
	#endif
    }
    
    num_obj_rhs_rules_pi=objects_in_rhs;
    
    for (unsigned int i=0;i<options->num_blocks_env;i++) {
	obj_lhs[object_pointer++]=read_bytes(is,precision.objects);
	
	unsigned int bid=i+options->num_rule_blocks;	
	for (int r=rules[bid]; r<rules[bid+1];r++) {
	    lengthUp[r]=read_bytes(is,PREC_NUM_OBJS_RHS(block_precision[bid]));
	    objects_in_rhs+=lengthUp[r];
	    int prob_offset=rules[options->num_rule_blocks]*options->num_environments+
				r-rules[options->num_rule_blocks];
	    int pr=read_bytes(is,PROB_SIZE);
	    prob[prob_offset]=pr/PROB_PREC;
	    if (options->max_rhs<lengthUp[r]) options->max_rhs=lengthUp[r];
	}

#ifdef BIN_DEBUG	
	cout << "ENV Block " << i << ":"<< endl;	
	for (int r=rules[bid]; r<rules[bid+1];r++)
		cout << "\tLength RHS="<<lengthUp[r]<<endl <<
			"\tProb=" << prob[r]<<endl;
	cout << "\tLHS=(";
	
	cout << id_objects[obj_lhs[object_pointer-1]] << ")" << endl;
#endif
    }
    
    obj_rhs=new unsigned int[objects_in_rhs];
    obj_rhs_m=new unsigned int[objects_in_rhs];
    
    
    return true;
}

bool PDP_Psystem_source_binary::read_inf_3() {
#ifdef BIN_DEBUG
    cout<< "ENTERING FOR INFORMATION 3"<<endl;
#endif
    unsigned int object_pointer=0; // global pointer for objects
    
    for (unsigned int i=0;i<options->num_rule_blocks;i++) {
#ifdef BIN_DEBUG
	cout << "PI Block " << i << ":"<< endl;	
#endif	
	for (int r=rules[i]; r<rules[i+1];r++) {
	    int rhsl=lengthUp[r];
	    lengthUp[r]=read_bytes(is,PREC_NUM_OBJS_RHS(block_precision[i]));
	
	    for (int o=0; o<lengthUp[r]; o++) {
		obj_rhs[object_pointer]=read_bytes(is,precision.objects);
		obj_rhs_m[object_pointer++]=read_bytes(is,PREC_MULTIPLICITY(block_precision[i]));//,precision.multisets);
	    }

	    lengthVp[r]=read_bytes(is,PREC_NUM_OBJS_LHS(block_precision[i]));

	    check_file_exception((lengthUp[r]+lengthVp[r])!=rhsl,"Different lengths of RHS in rule "+(r-rules[i]));

	    for (int o=0; o<lengthVp[r]; o++) {
		obj_rhs[object_pointer]=read_bytes(is,precision.objects);
		obj_rhs_m[object_pointer++]=read_bytes(is,PREC_MULTIPLICITY(block_precision[i]));//,precision.multisets);
	    }

#ifdef BIN_DEBUG	    
	    cout << "\tRule " << r-rules[i] << ", RHS=";
	    int o_it=object_pointer-rhsl;
	    for (int o=0; o<lengthUp[r]; o++) {
		cout << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_rhs_m[o_it] << " ";o_it++;
	    }
	    cout <<"[ ";
	    for (int o=0; o<lengthVp[r]; o++) {
		cout << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_rhs_m[o_it] << " ";o_it++;
	    }
	    cout <<"]^" << CHARGE_BETA(charge[i]) << "_" << active_membrane[i] << endl;
#endif
	}
    }
    
    for (unsigned int i=0;i<options->num_blocks_env;i++) {
	unsigned int bid=i+options->num_rule_blocks;
#ifdef BIN_DEBUG	
	cout << "Env Block " << bid << ":"<< endl;
#endif	
	for (int r=rules[bid]; r<rules[bid+1];r++) {	
	    for (int o=0; o<lengthUp[r]; o++) {
		obj_rhs[object_pointer]=read_bytes(is,precision.objects);
		obj_rhs_m[object_pointer++]=read_bytes(is,precision.environments);
	    }
#ifdef BIN_DEBUG
	    cout << "\tRule " << r-rules[i] << ", RHS=";
	    int o_it=object_pointer-lengthUp[r];
	    for (int o=0; o<lengthUp[r]; o++) {
		cout << "(" << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << ")_" << obj_rhs_m[o_it] << " ";o_it++;
	    }
#endif
	}
    }
    
    return true;
}

// Not necessary for version 34
//#define INITIAL_CHARGE(info) ((info>>(6+(precision.ini_num_objects-1)*8))&0x3)
//#define INITIAL_NUMBER_OBJECTS(info) (info&(0xFFFFFFFF^(0xC0<<(precision.ini_num_objects-1)*8)))
#define INI_OFFSET(e,q) (e*options->num_membranes+q)

bool PDP_Psystem_source_binary::read_multisets() {
    
    ini_multiset = new unsigned int* [options->num_environments*options->num_membranes];
    ini_info = new unsigned int [options->num_environments*options->num_membranes];
    ini_charge = new char [options->num_environments*options->num_membranes];
    
    for (int e=0; e<options->num_environments; e++) {
	for (int q=0; q<options->num_membranes; q++) {
		ini_charge[INI_OFFSET(e,q)] = read_bytes(is,1);
	    unsigned int num_objs = ini_info[INI_OFFSET(e,q)] = read_bytes(is,precision.ini_num_objects);
	    //int num_objs=INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET(e,q)]);
	    
	    if (num_objs<=0) ini_multiset[INI_OFFSET(e,q)]=NULL;
	    else ini_multiset[INI_OFFSET(e,q)]=new unsigned int [num_objs*2];
	    
	    for (int o=0;o<num_objs;o++) {
		int obj=read_bytes(is,precision.objects);
		int mult=read_bytes(is,precision.ini_mult);
		ini_multiset[INI_OFFSET(e,q)][o*2]=obj;
		ini_multiset[INI_OFFSET(e,q)][o*2+1]=mult;
	    }	    
	}
    }
    
#ifdef BIN_DEBUG
    for (int e=0; e<options->num_environments; e++) {
	cout << "Env "<<e<<endl;
	for (int q=0; q<options->num_membranes; q++) {
	    //cout << "\tMembr "<<q<<" ("<<INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET(e,q)])<<" #objects):";
		cout << "\tMembr "<<q<<" (charge: " << CHARGE_ALPHA(ini_charge[INI_OFFSET(e,q)]) << ", "<<ini_info[INI_OFFSET(e,q)]<<" #objects):";
	    //for (int o=0;o<(INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET(e,q)]))*2;o+=2) {
		for (int o=0;o<(ini_info[INI_OFFSET(e,q)])*2;o+=2) {
		cout << id_objects[ini_multiset[INI_OFFSET(e,q)][o]] <<"*"
			<< ini_multiset[INI_OFFSET(e,q)][o+1]<< " ";
	    }
	    cout << endl;
	}	
    }
#endif
    
    
    
    return true;
}



/*******************************************************/
/* Public methods inherited from pdp_system_source */
/*******************************************************/

bool PDP_Psystem_source_binary::start() {
	
    try {
	read_header();
	
	read_global_sizes();
	
	read_inf_1();
	
	read_inf_2();
	
	read_inf_3();
	
	read_multisets();
	
    } catch (FileException fe) {
	cerr << fe.getMessage() << endl;
	return false;
    }   
    
    return true;
}

/*********************/
/* Procedures for Pi */

unsigned int PDP_Psystem_source_binary::pi_loop_rule_blocks(){
    rule_block_it=0;
    lhs_it=rhs_it=0;
    return options->num_rule_blocks;
}

bool PDP_Psystem_source_binary::pi_next_rule_block() {
    if (rule_block_it < options->num_rule_blocks-1) {
	lhs_it+=lengthU[rule_block_it]+lengthV[rule_block_it];
	for (int r=rules[rule_block_it];r<rules[rule_block_it+1];r++)
	    rhs_it+=lengthUp[r]+lengthVp[r];
	rule_block_it++;
	return true;
    }
    return false;
}

unsigned int PDP_Psystem_source_binary::pi_loop_lhs() {
    return lengthV[rule_block_it]+lengthU[rule_block_it];
}

char PDP_Psystem_source_binary::pi_lhs_charge() {
    return CHARGE_ALPHA(this->charge[rule_block_it]);
}

unsigned int PDP_Psystem_source_binary::pi_lhs_membrane() {
    return active_membrane[rule_block_it];
}

unsigned int PDP_Psystem_source_binary::pi_lhs_parent_membrane() {
    return this->membrane_hierarchy[active_membrane[rule_block_it]];
}

unsigned int PDP_Psystem_source_binary::pi_lhs_loop_U() {
    U_it=0; V_it=lengthV[rule_block_it]+2;
    return lengthU[rule_block_it];
}

unsigned int PDP_Psystem_source_binary::pi_lhs_loop_V() {
    V_it=0; U_it=lengthU[rule_block_it]+2;
    return lengthV[rule_block_it];
}

bool PDP_Psystem_source_binary::pi_lhs_next_object(unsigned int & object, unsigned int & multiplicity) {
    int oidx=0;
    if (U_it<lengthU[rule_block_it]) {
	oidx=lhs_it+U_it++;
    }
    else if (V_it<lengthV[rule_block_it]) {
	oidx=lhs_it+lengthU[rule_block_it]+V_it++;
    }
    else {
	return false;
    }

    unsigned int obj=random() % options->num_objects;

    object = obj_lhs[oidx];
    multiplicity = obj_lhs_m[oidx];

    return true;
}

unsigned int PDP_Psystem_source_binary::pi_loop_rules() {
    rule_it=rules[rule_block_it];
    
    rhs_it_internal=rhs_it;

    return rules[rule_block_it+1]-rules[rule_block_it];
}

bool PDP_Psystem_source_binary::pi_next_rule() {
    if (rule_it<rules[rule_block_it+1]-1) {
	rhs_it_internal+=lengthUp[rule_it]+lengthVp[rule_it];
	rule_it++;
	return true;
    }
    return false;
}

float* PDP_Psystem_source_binary::pi_rule_probabilities() {
    return &(prob[rule_it*options->num_environments]);
}

unsigned int PDP_Psystem_source_binary::pi_loop_rhs() {
    return lengthVp[rule_it]+lengthUp[rule_it];
}

char PDP_Psystem_source_binary::pi_rhs_charge() {
    return CHARGE_BETA(this->charge[rule_block_it]);
}

unsigned int PDP_Psystem_source_binary::pi_rhs_membrane() {
    return active_membrane[rule_block_it];
}

unsigned int PDP_Psystem_source_binary::pi_rhs_loop_U() {
    Up_it=0; Vp_it=lengthVp[rule_it]+2;
    return lengthUp[rule_it];
}

unsigned int PDP_Psystem_source_binary::pi_rhs_loop_V() {
    Vp_it=0; Up_it=lengthUp[rule_it]+2;
    return lengthVp[rule_it];
}

bool PDP_Psystem_source_binary::pi_rhs_next_object(unsigned int & object, unsigned int & multiplicity) {
    int oidx=0;
    if (Up_it<lengthUp[rule_it]) {
	oidx=rhs_it_internal+Up_it++;
    }
    else if (Vp_it<lengthVp[rule_it]) {
	oidx=rhs_it_internal+lengthUp[rule_it]+Vp_it++;
    }
    else {
	return false;
    }

    object = obj_rhs[oidx];
    multiplicity = obj_rhs_m[oidx];

    return true;
}


/****************************************************/
/* Corresponding to rule blocks of the environments */

unsigned int PDP_Psystem_source_binary::env_loop_rule_blocks() {
    block_env_it=0;
    rule_env_it=rules[options->num_rule_blocks];
    rhs_it=num_obj_rhs_rules_pi;

    return options->num_blocks_env;
}

bool PDP_Psystem_source_binary::env_next_rule_block() {
    if (block_env_it < options->num_blocks_env-1) {
	for (int r=rules[block_env_it];r<rules[block_env_it+1];r++)
	    rhs_it+=lengthUp[r];
	block_env_it++;
	return true;
    }
    return false;
}

unsigned int PDP_Psystem_source_binary::env_get_object_lhs() {
	return obj_lhs[num_obj_lhs_blocks_pi+block_env_it];
}

unsigned int PDP_Psystem_source_binary::env_get_environment() {
	return this->active_membrane[options->num_rule_blocks+block_env_it];
}

unsigned int PDP_Psystem_source_binary::env_loop_rules() {
    
    int benv=options->num_rule_blocks+block_env_it;
    
    rule_env_it=rules[benv];
    
    rhs_it_internal=rhs_it;

    return rules[benv+1]-rules[benv];

}

bool PDP_Psystem_source_binary::env_next_rule() {
    int benv=options->num_rule_blocks+block_env_it;
    if (rule_env_it < rules[benv+1] -1) {
	rhs_it_internal+=lengthUp[rule_env_it];
	rule_env_it++;
	return true;
    }
    return false;
}

float PDP_Psystem_source_binary::env_get_probability() {
    int prob_offset=rules[options->num_rule_blocks]*options->num_environments+
				rule_env_it-rules[options->num_rule_blocks];
    return prob[prob_offset];
}

unsigned int PDP_Psystem_source_binary::env_loop_rhs() {
    Up_it=0;
    return lengthUp[rule_env_it];
}

bool PDP_Psystem_source_binary::env_next_object(unsigned int & object, unsigned int & environment) {
    int oidx=0;
    if (Up_it<lengthUp[rule_env_it]) {
	oidx=rhs_it_internal+Up_it++;
    }
    else {
	return false;
    }

    object = obj_rhs[oidx];
    environment = obj_rhs_m[oidx];

    return true;
}

/*************************************/
/* Corresponding to the configuration*/

bool PDP_Psystem_source_binary::conf_loop() {
    return true;
}

unsigned int PDP_Psystem_source_binary::conf_loop_environments() {
    env_it=0;
    return options->num_environments;
}

bool PDP_Psystem_source_binary::conf_next_environment() {
    if (env_it<options->num_environments-1) {
	env_it++;
	return true;
    }
    return false;
}

unsigned int PDP_Psystem_source_binary::conf_loop_membranes() {
    memb_it=0;
    return options->num_membranes;
}

bool PDP_Psystem_source_binary::conf_next_membrane() {
    if (memb_it<options->num_membranes-1) {
	memb_it++;
	return true;
    }
    return false;
}

unsigned int PDP_Psystem_source_binary::conf_parent_membrane() {
    return this->membrane_hierarchy[memb_it];
}

char PDP_Psystem_source_binary::conf_charge_membrane() {
    //return INITIAL_CHARGE(ini_info[INI_OFFSET(env_it,memb_it)]);
	return CHARGE_ALPHA(ini_charge[INI_OFFSET(env_it,memb_it)]);
}

unsigned int PDP_Psystem_source_binary::conf_loop_objects() {
    obj_it=0;
    return options->num_objects;
}

bool PDP_Psystem_source_binary::conf_next_object(unsigned int & object, unsigned int & multiplicity) {
    //if (obj_it<INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET(env_it,memb_it)])*2-1) {
	if (ini_info[INI_OFFSET(env_it,memb_it)]==NULL) return false;

	if (obj_it<ini_info[INI_OFFSET(env_it,memb_it)]*2-1) {
		object=ini_multiset[INI_OFFSET(env_it,memb_it)][obj_it];
		multiplicity=ini_multiset[INI_OFFSET(env_it,memb_it)][obj_it+1];
		obj_it+=2;
		return true;
    }

    return false;
}


char ** PDP_Psystem_source_binary::get_objects_ids() {
    delete_id_objects=false;
	return id_objects;
}
	    
char ** PDP_Psystem_source_binary::get_environments_ids() {
    delete_id_environments=false;
    return id_environments;
}
	
char ** PDP_Psystem_source_binary::get_membranes_ids() {
    delete_id_membranes=false;
    return id_membranes;
}

/*****************************************************************/
/* Writing the debug file from P-Lingua testing examples (no. 2) */

void PDP_Psystem_source_binary::write_test_binary() {
	
    ofstream outfile ("small.bin",ofstream::binary);
    char buffer [8];

// Header
    buffer[0]=0xAF;
    buffer[1]=0x12;
    buffer[2]=0xFA;
    buffer[3]=0x21; // 2 = PDP systems
    outfile.write(buffer,4);
    
// Subheader
    buffer[0]=0;
    buffer[1]=0;
    buffer[2]=0x04; // Show object labels 
    outfile.write(buffer,3);
    
// Number of objects
    buffer[0]=0x09; // 8 objects plus the empty object #
    outfile.write(buffer,1);
    
// Objects ids    
    buffer[0]='#';
    buffer[1]='\0';
    outfile.write(buffer,2);
    buffer[0]='a';
    outfile.write(buffer,2);
    buffer[0]='b';
    outfile.write(buffer,2);
    buffer[0]='c';
    outfile.write(buffer,2);
    buffer[0]='d';
    outfile.write(buffer,2);
    buffer[0]='e';
    outfile.write(buffer,2);
    buffer[0]='f';
    outfile.write(buffer,2);
    buffer[0]='g';
    outfile.write(buffer,2);
    buffer[0]='h';
    outfile.write(buffer,2);
        
// Number of Environments
    buffer[0]=0x01;
    outfile.write(buffer,1);
    
// Number of membranes
    buffer[0]=0x03; // 2 membranes plus the environment as a region
    outfile.write(buffer,1);
    // Membrane 0
    buffer[0]=0;
    outfile.write(buffer,1); // Id father
    // Membrane 1
    buffer[0]=0;
    outfile.write(buffer,1); // Id father
    // Membrane 2
    buffer[0]=0x01;
    outfile.write(buffer,1); // Id father
    
// Number of rule blocks of Pi
    buffer[0]=0x06;
    outfile.write(buffer,1);
// Number of rule blocks of Environment
    buffer[0]=0x00;
    outfile.write(buffer,1);
    
// INFORMATION RULE BLOCKS PART 1
    
// Rule block 0
    buffer[0]=0; // Information Byte
    buffer[1]=0x03; // Num rules
    buffer[2]=0x03; // Size LHS
    buffer[3]=0x02; // Act Membrane
    buffer[4]=0; // Charges
    outfile.write(buffer,5);
    
// Rule block 1
    buffer[0]=0; // Information Byte
    buffer[1]=0x01; // Num rules
    buffer[2]=0x02; // Size LHS
    buffer[3]=0x02; // Act Membrane
    buffer[4]=0; // Charges
    outfile.write(buffer,5);

// Rule block 2
    buffer[0]=0; // Information Byte
    buffer[1]=0x01; // Num rules
    buffer[2]=0x02; // Size LHS
    buffer[3]=0x02; // Act Membrane
    buffer[4]=0; // Charges
    outfile.write(buffer,5);
    
// Rule block 3
    buffer[0]=0; // Information Byte
    buffer[1]=0x01; // Num rules
    buffer[2]=0x02; // Size LHS
    buffer[3]=0x01; // Act Membrane
    buffer[4]=0xA0; // Charges
    outfile.write(buffer,5);
    
// Rule block 4
    buffer[0]=0; // Information Byte
    buffer[1]=0x01; // Num rules
    buffer[2]=0x01; // Size LHS
    buffer[3]=0x02; // Act Membrane
    buffer[4]=0; // Charges
    outfile.write(buffer,5);
    
// Rule block 5
    buffer[0]=0; // Information Byte
    buffer[1]=0x01; // Num rules
    buffer[2]=0x02; // Size LHS
    buffer[3]=0x02; // Act Membrane
    buffer[4]=0x20; // Charges
    outfile.write(buffer,5);

// INFORMATION RULE BLOCKS PART 2
    
// Rule block 0
    // Rule 1
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x1B; // Probability 7000
    buffer[2]=0x58; // Probability 7000
    // Rule 2
    buffer[3]=0x01; // Size RHS
    buffer[4]=0x7; // Probability 2000
    buffer[5]=0xD0; // Probability 2000
    outfile.write(buffer,6);
    // Rule 3
    buffer[0]=0x02; // Size RHS
    buffer[1]=0x3; // Probability 1000
    buffer[2]=0xE8; // Probability 1000
    outfile.write(buffer,3);
    
    // LHS
    buffer[0]=0x00; // LHS U
    buffer[1]=0x03; // LHS V
    outfile.write(buffer,2);
    
    buffer[0]=0x01; // a
    buffer[1]=0x04; // 4
    buffer[2]=0x02; // b
    buffer[3]=0x04; // 4
    buffer[4]=0x03; // c
    buffer[5]=0x02; // 2
    outfile.write(buffer,6);
    
// Rule block 1
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x27; // Probability 10000
    buffer[2]=0x10; // Probability 10000

    // LHS
    buffer[3]=0x00; // LHS U
    buffer[4]=0x02; // LHS V
    outfile.write(buffer,5);
    
    buffer[0]=0x01; // a
    buffer[1]=0x04; // 4
    buffer[2]=0x04; // d
    buffer[3]=0x01; // 1    
    outfile.write(buffer,4);
    
// Rule block 2
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x27; // Probability 10000
    buffer[2]=0x10; // Probability 10000

    // LHS
    buffer[3]=0x00; // LHS U
    buffer[4]=0x02; // LHS V
    outfile.write(buffer,5);
    
    buffer[0]=0x02; // b
    buffer[1]=0x05; // 5
    buffer[2]=0x04; // d
    buffer[3]=0x02; // 2
    outfile.write(buffer,4);    
    
// Rule block 3
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x27; // Probability 10000
    buffer[2]=0x10; // Probability 10000
    outfile.write(buffer,3);

    // LHS
    buffer[0]=0x01; // LHS U
    buffer[1]=0x02; // b
    buffer[2]=0x01; // 1
    
    buffer[3]=0x01; // LHS V
    buffer[4]=0x01; // a
    buffer[5]=0x07; // 7    
    outfile.write(buffer,6);    

// Rule block 4
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x27; // Probability 10000
    buffer[2]=0x10; // Probability 10000

    // LHS
    buffer[3]=0x01; // LHS U
    buffer[4]=0x01; // a
    buffer[5]=0x03; // 3
    buffer[6]=0x00; // LHS V
    outfile.write(buffer,7);
    
// Rule block 5
    buffer[0]=0x01; // Size RHS
    buffer[1]=0x27; // Probability 10000
    buffer[2]=0x10; // Probability 10000
    outfile.write(buffer,3);  

    // LHS
    buffer[0]=0x02; // LHS U
    buffer[1]=0x01; // a
    buffer[2]=0x01; // 1
    buffer[3]=0x02; // b
    buffer[4]=0x01; // 1
    buffer[5]=0x00; // LHS V
    outfile.write(buffer,6);
    
    
// INFORMATION RULE BLOCKS PART 3
    
// Rule block 0
    // RHS rule 1
    buffer[0]=0x01; // RHS U'
    buffer[1]=0x05; // e
    buffer[2]=0x02; // 2
    buffer[3]=0x00; // RHS V'
    outfile.write(buffer,4);
    // RHS rule 2
    buffer[0]=0x00; // RHS U'    
    buffer[1]=0x01; // RHS V'
    buffer[2]=0x05; // e
    buffer[3]=0x02; // 2
    outfile.write(buffer,4);
    // RHS rule 3
    buffer[0]=0x00; // RHS U'    
    buffer[1]=0x02; // RHS V'
    buffer[2]=0x05; // e
    buffer[3]=0x01; // 1
    buffer[4]=0x06; // f
    buffer[5]=0x01; // 1
    outfile.write(buffer,6);
    
// Rule block 1
    // RHS
    buffer[0]=0x01; // RHS U'
    buffer[1]=0x06; // f
    buffer[2]=0x02; // 2
    buffer[3]=0x00; // RHS V'
    outfile.write(buffer,4);
    
// Rule block 2
    // RHS
    buffer[0]=0x01; // RHS U'
    buffer[1]=0x07; // g
    buffer[2]=0x02; // 2
    buffer[3]=0x00; // RHS V'
    outfile.write(buffer,4);

// Rule block 3
    // RHS
    buffer[0]=0x00; // RHS U'
    buffer[1]=0x01; // RHS V'
    buffer[2]=0x08; // h
    buffer[3]=100; // 100    
    outfile.write(buffer,4);
    
// Rule block 4
    // RHS
    buffer[0]=0x00; // RHS U'
    buffer[1]=0x01; // RHS V'
    buffer[2]=0x05; // e
    buffer[3]=0x03; // 3
    outfile.write(buffer,4);
    
// Rule block 5
    // RHS
    buffer[0]=0x00; // RHS U'
    buffer[1]=0x01; // RHS V'
    buffer[2]=0x07; // g
    buffer[3]=0x03; // 3
    outfile.write(buffer,4);
    
    
// Initial multisets
    // Environment
    buffer[0]=0x01; // Information
    buffer[1]=0x02; // b
    buffer[2]=0x01; // 1
    outfile.write(buffer,3);
    // Membrane 1
    buffer[0]=0x01; // Information
    buffer[1]=0x01; // a
    buffer[2]=60; // 60
    outfile.write(buffer,3);
    // Membrane 2
    buffer[0]=0x04; // Information
    outfile.write(buffer,1);
    buffer[0]=0x01; // a
    buffer[1]=90; // 90
    buffer[2]=0x02; // b
    buffer[3]=72; // 72
    buffer[4]=0x03; // c
    buffer[5]=66; // 66
    buffer[6]=0x04; // d
    buffer[7]=30; // 30
    outfile.write(buffer,8);
    
    outfile.close();    
    
    /* THIS IS EXAMPLE 2 OF P-LINGUA
     * EXPECTING:
     *	PHASE 1: B1*10, B2*11, B3*5, B5*20
     *	PHASE 2: B2*1
     */
}
