/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2019  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Authors: Miguel Ángel Martínez-del-Amor
             Ignacio Pérez-Hurtado
    
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

#include "pdp_psystem_source_pl5.h"
//#include <string.h>
#include <plingua/serialization.hpp>

using namespace std;
using namespace plingua;

/**********************************************************/
/* CONSTRUCTORS:                                          */
/*                                                        */
/* P-Lingua5 reader of a PDP System                       */
/**********************************************************/

PDP_Psystem_source_plingua5::PDP_Psystem_source_plingua5(const char * filename, Options options, string& filter_file): PDP_Psystem_source_binary(filename,options,filter_file) {}

PDP_Psystem_source_plingua5::PDP_Psystem_source_plingua5(Options options): PDP_Psystem_source_binary(options) { }

PDP_Psystem_source_plingua5::~PDP_Psystem_source_plingua5() {} 


/********************************************/
/* AUXILIARY FUNCTIONS:                     */
/*                                          */
/********************************************/

// Infer the number of membranes inside the provided one
int calculate_number_membranes (Membrane & memb)
{
	int num = 1; // count the membrane itself	

	for (int m = 0; m < memb.data.size(); m++)
		num += calculate_number_membranes(memb.data[m]);
	return num;
}

// Go through a tree of membranes, collect the labels in a map, annotate the parents.
void traverse_membranes (Membrane & memb, std::map<LabelString,uint>& map_labels, uint* parents, uint id_parent) {

	// calculate id, add it to the map and set the parent id
	uint id = map_labels.size() + 1;
	map_labels[memb.label[0]] = id;
	parents[id] = id_parent;

	// continue recursively to children membranes
	for (int m = 0; m < memb.data.size(); m++)
		traverse_membranes(memb.data[m],map_labels,parents,id);
}

// translate the charge defined in plingua5 to the charge used in abcdgpu
inline int transform_charge (int charge_pl5) {
	if (charge_pl5 == -1)
		return 2;
	return charge_pl5;
}

// Traverse a membrane tree in an environment to collect the charges
void traverse_charges_membranes (const Membrane & memb, const std::map<LabelString,uint>& map_labels,char* charges) {

	// calculate id, add it to the map and set the parent id
	uint id = map_labels.at(memb.label[0]);
	charges[id] = transform_charge(memb.charge);	

	// continue recursively to children membranes
	for (int m = 0; m < memb.data.size(); m++)
		traverse_charges_membranes(memb.data[m],map_labels,charges);
}

#define INI_OFFSET_PL5(e,q) (e*options->num_membranes+q)

/*******************************************************/
/* Public methods inherited from pdp_system_source     */
/*******************************************************/
bool PDP_Psystem_source_plingua5::start() {

	if (options->verbose>0)
		cout << "Reading binary file using P-Lingua 5 format" << endl;

#ifdef BIN_DEBUG
    cout << "EATING THE CEREALS FROM P-LINGUA 5: " << endl;
#endif

    /************************************************/
    /* I. Reading and parsing P system              */
    File f;
    loadFromBinaryFile(filename, f);
    Psystem &ps = f.psystem;

#ifdef BIN_DEBUG
    cout << "This is the PDP system being recovered: " << endl << ps << endl << endl;
#endif

	
	// Auxiliary std maps for classifying the rules into blocks, etc.
    // for membrane traversing	
	map<LabelString,uint> membrane_labels_id; // map, String Label of a membrane -> membrane ID
	map<LabelString,uint> environment_labels_id; // map, String Label of a membrane -> membrane ID
	// for rules
	map<pair<uint,LHR>,map<Rule,vector<double>>> rule_blocks; //map LHS with a map of rules and vector of probabilities per environment.
	map<tuple<uint,LabelString,ObjectString>,vector<Rule>> environ_blocks; // map a pair object,env_label to a vector of rules
	// for modules
	map<string,uint> module_ids;
	map<string,uint> module_steps;
	map<string,string> module_precedences; // assumes it is synchronized, so it doesn't matter who is the precedent

	// Fetch global sizes
	number_objects = ALPHABET.getObjectAlphabetSize();
	number_environments=ps.structure.data.size();	
	
	// Fetch environments labels	
	for (int e = 0; e < number_environments; e++) {		
		environment_labels_id[ps.structure.data[e].label[1]] = e;
		//cout << ps.structure.data[e].label[1] << endl;
	}

	// Fetch membranes  (env0 is ps.structure.data[0], skin is .data[0])
	Membrane &mem = ps.structure.data[0].data[0];

	number_membranes = calculate_number_membranes(mem)+1; // count the environment as a membrane itself containing the skin
 	membrane_hierarchy = new uint[number_membranes];
	traverse_membranes(mem,membrane_labels_id,membrane_hierarchy,0);	

	// Reading and parsing modules feature (if available)
	string modules;
	if (options->modular) { // it must be enable beforhand by using the corresponding parameter
		try {
			modules = ps.features.at(FeatureString("modules")).as_string();
			options->modular = true;
		} catch(std::out_of_range& e) {
			options->modular = false;
		}
	}

	if (options->modular) {
		// clean the string: delete spaces and line breaks
		modules.erase(remove(modules.begin(),modules.end(),' '),modules.end());
		modules.erase(remove(modules.begin(),modules.end(),'\n'),modules.end());
		modules.erase(remove(modules.begin(),modules.end(),'\t'),modules.end());

		//options->modules = count(modules.begin(),modules.end(),';') + 1;
		string mod;
		istringstream modulesStream(modules);
		uint moduleId=0;

		// at this point, modules = (mod_number,mod_steps,{mod_suc1,mod_suc2,...});(mod_number,mod_steps,{mod_suc1,mod_suc2,...});...
		while (std::getline(modulesStream, mod, ';')) {
			// go to the brackets and cut the substring, so that we can skip empty spaces
			int open = mod.find("(")+1;
			int close = mod.find(")");
			mod = mod.substr(open,close-open);
			// at this point, mod = mod_number,mod_steps,{mod_suc1,mod_suc2,...}
			istringstream modElemsStream(mod);
			try {
				string modElem;
				string modSId;
				// read module ID
				if (std::getline(modElemsStream, modElem, ',')) {
					modSId = modElem;
					module_ids[modSId] = moduleId;
				} else {
					cout << "Error parsing modules" << endl; // TODO: throw error
					return false;
				}
				// read module length in steps
				if (std::getline(modElemsStream, modElem, ',')) {
					//options->modules_end[moduleId] = stoi(modElem) + options->modules_start[moduleId];
					module_steps[modSId] = stoi(modElem);
				} else {
					cout << "Error parsing modules" << endl; // TODO: throw error
					return false;
				}
				// Now let jump to the successors of a module
				open = mod.find("{")+1;
				close = mod.find("}");
				mod = mod.substr(open,close-open);

				istringstream modSucStream(mod);
				string nmod;
				while (std::getline(modSucStream, nmod, ',')) {
					if (nmod.size()>0)
						module_precedences[nmod] = modSId;
				}
			} catch (std::invalid_argument & e) {
				cout << "Error parsing modules: modules lengths are defined using only positive integers" << endl;
				return false;
			}
			moduleId++;
		}

		options->modules = moduleId;
		options->modules_start = new uint[options->modules+1];
		options->modules_end = new uint[options->modules+1];
		options->modules_pi_index = new uint[options->modules+1];
		options->modules_env_index = new uint[options->modules+1];
		memset(options->modules_start,0,sizeof(uint)*(options->modules+1));
		memset(options->modules_end,0,sizeof(uint)*(options->modules+1));
		memset(options->modules_pi_index,0/*xFF*/,sizeof(uint)*(options->modules+1)); // initialized with UINT_MAX to detect when no Pi/env blocks are within the module
		memset(options->modules_env_index,0/*xFF*/,sizeof(uint)*(options->modules+1));

		for (auto m:module_ids) {
			try {
				uint steps = module_steps.at(m.first);
				uint start = 0;

				auto it = module_precedences.find(m.first);
				while (it!=module_precedences.end()) {
					start += module_steps.at(it->second);
					it = module_precedences.find(it->second);
				}

				options->modules_start[m.second] = start;
				options->modules_end[m.second] = start + steps;
			}
			catch (std::out_of_range & e) {
				cout << "Error parsing modules: modules lengths are defined using only positive integers" << endl;
				return false;
			}
		}
	}

	/// reading skeleton rules
	// Clasify rule blocks by its LHS, the rule and vector of probabilities per environment, inside the map rule_blocks.
	for (auto r: ps.rules) {
		if (r.features.count("pattern")==0 || strcmp(r.features.at("pattern").as_string(),"skeleton_rule")!=0) {
			continue;
		}
		Rule sr(r);
		LHR& key = sr.lhr;
		key.membrane.label.erase(key.membrane.label.begin()+1,key.membrane.label.end());
		sr.rhr.data[0].label.erase(sr.rhr.data[0].label.begin()+1,sr.rhr.data[0].label.end());

		uint mod = options->modules; // default module
		if ((options->modular) && (r.features.count("module")>0))
			mod = module_ids.at(r.features.at("module").as_string());

		pair <uint,LHR&> p (mod,sr.lhr);

		if (rule_blocks[p].count(sr)==0) {
			rule_blocks[p][sr].resize(number_environments);
		}
		if (r.lhr.membrane.label.size()==1) {
			for (int e=0;e<number_environments;e++) {
				rule_blocks[p][sr][e] = r.features.at("probability").as_double();
			}

		} else {
			rule_blocks[p][sr][environment_labels_id.at(r.lhr.membrane.label[1])] = r.features.at("probability").as_double();
		}
	}
	///

	/// reading environment rules
	// Clasify rule blocks by its LHS pair (object,env) insinde the map environ_blocks.
	for (auto r: ps.rules) {
		if (r.features.count("pattern")==0 || strcmp(r.features.at("pattern").as_string(),"communication_rule")!=0) {
			continue;
		}

		ObjectString os;
		LabelString ls;
		bool error = true;
		for (auto m: r.lhr.membrane.data) {
			if (m.multiset.size()==1) { // this is the environment having an object
				ls = m.label[0]; // first label corresponding to env
				os = m.multiset.begin()->first; // first object (must be only one, so must exist)
				if (m.multiset.begin()->second.raw() != 1) {
					cout << "Error: environment rules cannot have more than one object in the LHS: " << m <<  endl;
					return false;
				}
				error = false;
				break;
			}
			else if (m.multiset.size()>0) {
				cout << "Error: environment rules cannot have more than one object in the LHS: " << m <<  endl;
				return false;
			}
		}

		uint mod = options->modules; // default module
		if ((options->modular) && (r.features.count("module")>0))
			mod = module_ids.at(r.features.at("module").as_string());

		std::tuple<uint,LabelString,ObjectString> key(mod,ls,os);
		environ_blocks[key].push_back(r);
	}
	///

	// debugging code to print the order of the rules
	/*uint rbid = 0;
	uint rid = 0;
	for (auto b:rule_blocks) {
		cout << "Module " << b.first.first << "| Rule block " << rbid++ << "(" << b.first.second << "):" << endl;
		for (auto r:b.second) {
			cout << "\t Rule " << rid++ << "("<<  r.first << "):\t THE MODULE IS " << r.first.features.at("module") << endl;
		}
	}

	uint erbid = 0;
	for (auto b:environ_blocks) {
		cout << "Module " << get<0>(b.first) << "| Env rule block " << rbid++ << ", env " << erbid++ << " (e=" << get<1>(b.first) << ",o=" << get<2>(b.first) << "):" << endl;
		for (auto r:b.second) {
			cout << "\t Rule " << rid++ << "("<<  r << "):\t THE MODULE IS " << r.features.at("module") << endl;
		}
	}*/

	// Read strings for objects
	id_objects=new char*[number_objects];
	for (int i=0;i<number_objects;i++){
		id_objects[i] = new char[ALPHABET.getObject(i).length()+1];
		strcpy(id_objects[i],ALPHABET.getObject(i).c_str());
	}
	// Read strings for environments
	id_environments=new char*[number_environments];
	for (auto e:environment_labels_id){
		id_environments[e.second] = new char[e.first.str().length()+1];
		strcpy(id_environments[e.second],e.first.str().c_str());
	}
	// Read strings for membranes
	id_membranes=new char*[number_membranes];
	for (auto m:membrane_labels_id){
		id_membranes[m.second] = new char[m.first.str().length()+1];
		strcpy(id_membranes[m.second],m.first.str().c_str());
	}

	number_rule_blocks_pi=rule_blocks.size();

	number_rule_blocks_env=environ_blocks.size();

#ifdef BIN_DEBUG
	cout << endl << "number objects=" << number_objects << endl;
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
	cout << "number pi rule blocks=" << number_rule_blocks_pi << endl;
	cout << "number env rule blocks=" << number_rule_blocks_env << endl;
#endif
	/* End reading the P system from P-Lingua */

	/* Start creating structures */
	options->num_blocks_env=number_rule_blocks_env;
	options->num_rule_blocks=number_rule_blocks_pi;
	options->num_objects=number_objects;
	options->num_membranes=number_membranes;
	options->num_environments=number_environments;	
	
	// Initialize aux data structures  
	lengthU= new short int[options->num_rule_blocks];

	lengthV= new short int[options->num_rule_blocks];

	active_membrane= new unsigned int[options->num_rule_blocks+options->num_blocks_env];	
	
	rules = new unsigned int[options->num_rule_blocks+options->num_blocks_env+1];
	
	charge = new char[options->num_rule_blocks];
	
	/************************************************/


	/************************************************/
	/* II. Entering reading information 1           */
#ifdef BIN_DEBUG
    cout  << "COLLECTING INFORMATION 1 (blocks): " << endl;
#endif
    
    unsigned int objects_in_lhs=0;
    number_rules_pi=number_rules_env=number_rules=0;
    prob_length=0;
    
    rules[0]=0;
    options->max_lhs=0;
    options->max_num_rules=0;
    
    uint rbi = 0;
    // Iterate rule blocks
    for (auto rb:rule_blocks) {
		int nr=rb.second.size();		
		rules[rbi+1]=rules[rbi]+nr;
		lengthU[rbi]=rb.first.second.multiset.size()+rb.first.second.membrane.multiset.size(); // u+v

		//active_membrane[rbi]=membrane_labels_id.at(rb.first.second.membrane.label[0]);
		charge[rbi]=load_charge_alpha(transform_charge((int) rb.first.second.membrane.charge)); // alpha charge
		charge[rbi]|=load_charge_beta(transform_charge((int) rb.second.begin()->first.rhr.data[0].charge)); // beta charge (just the one from the first rule)
		
		objects_in_lhs+=lengthU[rbi];
		number_rules_pi+=nr;
		//prob_length+=(DIF_PROB_ENV(block_precision[rbi]))?nr:nr*options->num_environments;
		prob_length+=nr*options->num_environments;
		// Update max lhs
		if (options->max_lhs<lengthU[rbi]) options->max_lhs=lengthU[rbi];
		if (options->max_num_rules<nr) options->max_num_rules=nr;

	#ifdef BIN_DEBUG
		cout << "PI Block " << rbi << ":"<< endl;
		cout << "\tNo Rules="<<rules[rbi+1]-rules[rbi]<<endl;
		cout << "\tLength LHS="<<lengthU[rbi]<<endl;
		cout << "\tAMembrane="<<active_membrane[rbi]<<endl;
		cout << "\tCharge Alfa="<<(int)charge_alpha(charge[rbi])<<endl;
		cout << "\tCharge Beta="<<(int)charge_beta(charge[rbi])<<endl;
	#endif

		rbi++;

		if (options->modular)
			options->modules_pi_index[rb.first.first]++; // = rbi; // Note we are storing the following block id
    }  
    
    rbi=0;
    for (auto rb:environ_blocks) {    	
		int nr=0;
		uint bid=rbi+options->num_rule_blocks;		
		nr=rb.second.size();
		rules[bid+1]=nr+rules[bid];
		active_membrane[bid]=environment_labels_id.at(get<1>(rb.first));
		number_rules_env+=nr;
		//prob_length+=(DIF_PROB_ENV(block_precision[rbi]))?nr:nr*options->num_environments;
		prob_length+=nr;
		if (options->max_num_rules<nr) options->max_num_rules=nr;

#ifdef BIN_DEBUG	
		cout << "ENV Block " << rbi << " (" << bid <<"):"<< endl;
		cout << "\tNo Rules="<<nr<<endl;
		cout << "\tEnv="<<active_membrane[bid]<<endl;
#endif

		rbi++;

		if (options->modular)
			options->modules_env_index[get<0>(rb.first)]++; //= rbi; // Note we are storing the following block id
    }

    // calculate indexes in modules
    if (options->modular) {
    	for (int m=options->modules-1; m>0; m--) {
    		options->modules_pi_index[m]=options->modules_pi_index[m-1];
    		options->modules_env_index[m]=options->modules_env_index[m-1];
    	}
    	options->modules_pi_index[0]=0;
    	options->modules_env_index[0]=0;
    	// after this, indexes vectors say the starting index of the module. Calculate the size with following module
    	for (int m=1;m<options->modules;m++) {
    		options->modules_pi_index[m]+=options->modules_pi_index[m-1];
    		options->modules_env_index[m]+=options->modules_env_index[m-1];
    	}
    }

    number_rules=number_rules_pi+number_rules_env;

#ifdef BIN_DEBUG    
    cout << "Total rules Pi="<<number_rules_pi<<endl;
    cout << "Total rules Env="<<number_rules_env<<endl;
    cout << "Total rules="<<number_rules<<endl;
    cout << "Total length probs="<<prob_length<<endl;
    if (options->modular) {
    	cout << "Using modules: " << endl;
    	cout << "modules_start: "; for (int m=0; m<options->modules; m++) cout << m << "=" << options->modules_start[m] << ", ";
    	cout << endl <<  "modules_end: "; for (int m=0; m<options->modules; m++) cout << m << "=" << options->modules_end[m] << ", ";
    	cout << endl <<  "modules_pi_index: "; for (int m=0; m<options->modules; m++) cout << m << "=" << options->modules_pi_index[m] << ", ";
    	cout << endl <<  "modules_env_index: "; for (int m=0; m<options->modules; m++) cout << m << "=" << options->modules_env_index[m] << ", ";
    	cout << endl;
    }
#endif
    
    obj_lhs=new unsigned int[objects_in_lhs+options->num_blocks_env];
    obj_lhs_m=new unsigned int[objects_in_lhs];
    lengthUp=new short int[number_rules];
    lengthVp=new short int[number_rules];
    prob=new float[prob_length];
    
    num_obj_lhs_blocks_pi=objects_in_lhs;
    /************************************************/


    /************************************************/
    /* III. Entering reading information 2          */
#ifdef BIN_DEBUG
    cout << "COLLECTING INFORMATION 2 (rules): " << endl;
#endif
    uint objects_in_rhs=0;
    uint object_pointer=0; // global pointer for objects
    options->max_rhs=0;
    
    rbi = 0;

    // Reading skeleton rules RHR
    for (auto rb:rule_blocks)
	{
		uint ri = rules[rbi];
		// collecting RHR total length and probabilities per environment per rule in the block
		for (auto r: rb.second) {
			auto& t = r.first;
			lengthUp[ri]=t.rhr.multiset.size()+t.rhr.data[0].multiset.size(); // U'+V'
			objects_in_rhs+=lengthUp[ri];

			for (int e=0;e<options->num_environments;e++) {				
				prob[ri*options->num_environments+e]=r.second[e];
			}

			if (options->max_rhs<lengthUp[ri]) options->max_rhs=lengthUp[ri];
			ri++;
		}

		// this is the total size of the LHS
		short int lhsl=lengthU[rbi];

		// get U size and the corresponding objects and their multiplicities (only those appearing)
		lengthU[rbi]=rb.first.second.multiset.size();
		for (auto it = rb.first.second.multiset.begin(); it!=rb.first.second.multiset.end(); it++) {
			obj_lhs[object_pointer]=ALPHABET.getObjectId(it->first.str()).getId();
			obj_lhs_m[object_pointer++]=it->second.raw();
		}

		// get V size and the corresponding objects and their multiplicities
		lengthV[rbi]=rb.first.second.membrane.multiset.size();
		if ((lengthU[rbi]+lengthV[rbi])!=lhsl) {
			cout << "Error: Different lengths of LHS in info 1 vs 2 " << endl;
			cout << "\t Ruleblock of Pi " << rbi << ", |U|=" << lengthU[rbi] << ",|V|=" << lengthV[rbi] << ",|LHS|=" << lhsl << endl;
			return false;
		}

		for (auto it = rb.first.second.membrane.multiset.begin(); it!=rb.first.second.membrane.multiset.end(); it++) {
			obj_lhs[object_pointer]=ALPHABET.getObjectId(it->first.str()).getId();
			obj_lhs_m[object_pointer++]=it->second.raw();
		}
	
#ifdef BIN_DEBUG
		cout << "PI Block " << rbi << ":"<< endl;
		for (int r=rules[rbi]; r<rules[rbi+1];r++)
			cout << "\tLength RHS=" << lengthUp[r]<< endl << "\tProb=" << prob[r*options->num_environments]<<endl;
		cout << "\tLHS=";
		int o_it=object_pointer-lhsl;
		for (int o=0; o<lengthU[rbi]; o++) {
			cout << id_objects[obj_lhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_lhs_m[o_it] << " ";o_it++;
		}
		cout <<"[ ";
		for (int o=0; o<lengthV[rbi]; o++) {
			cout << id_objects[obj_lhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_lhs_m[o_it] << " ";o_it++;
		}
		cout <<"]^" << (int)charge_alpha(charge[rbi]) << "_" << (int)active_membrane[rbi] << endl;
#endif

		rbi++;
    }
    
    num_obj_rhs_rules_pi=objects_in_rhs;
    
    // communication rules
    rbi=0;
    for (auto rb:environ_blocks) {
		obj_lhs[object_pointer++]=ALPHABET.getObjectId(get<2>(rb.first).str()).getId();
		
		uint bid=rbi+options->num_rule_blocks;	
		uint ri=rules[bid];
		for (auto r: rb.second) {
			uint rhsl = 0;
			for (auto m: r.rhr.data[0].data) {	
				if (m.multiset.size()==0)
					continue;
				else if (m.multiset.size()==1) 
					rhsl++;					
				else {
					cout << "Error: environment rules cannot have more than one object in each environment at the RHS: " << m <<  endl;
					return false;
				}
			}

		    lengthUp[ri]=rhsl;
		    objects_in_rhs+=lengthUp[ri];
		    int prob_offset=rules[options->num_rule_blocks]*options->num_environments+ri-rules[options->num_rule_blocks];
		    prob[prob_offset]=r.features.at("probability").as_double();
		    if (options->max_rhs<lengthUp[ri]) options->max_rhs=lengthUp[ri];
		    ri++;
		}

#ifdef BIN_DEBUG	
		cout << "ENV Block " << rbi << ":"<< endl;	
		for (int r=rules[bid]; r<rules[bid+1];r++)
			cout << "\tLength RHS="<<lengthUp[r]<<endl << "\tProb=" << prob[rules[options->num_rule_blocks]*options->num_environments+r-rules[options->num_rule_blocks]]<<endl;
		
		cout << "\tLHS=(";		
		cout << id_objects[obj_lhs[object_pointer-1]] << ")_" << active_membrane[bid] << endl;
#endif
		rbi++;
    }
    
    obj_rhs=new unsigned int[objects_in_rhs];
    obj_rhs_m=new unsigned int[objects_in_rhs];
    /************************************************/


    /************************************************/
    /* IV. Entering reading information 3           */
#ifdef BIN_DEBUG
    cout << "COLLECTING INFORMATION 3 (RHS of rules): " << endl;
#endif
    object_pointer=0; // global pointer for objects
    
    // finally collect the RHS of rules
    rbi = 0;
    for (auto rb:rule_blocks) {

#ifdef BIN_DEBUG
		cout << "PI Block " << rbi << ":"<< endl;	
#endif	
		uint ri=rules[rbi];
		for (auto r: rb.second) {
			auto& t = r.first;			
		    int rhsl=lengthUp[ri];
		    lengthUp[ri]=t.rhr.multiset.size();
		
			for (auto it = t.rhr.multiset.begin(); it!=t.rhr.multiset.end(); it++) {
				obj_rhs[object_pointer]=ALPHABET.getObjectId(it->first.str()).getId();
				obj_rhs_m[object_pointer++]=it->second.raw();
			}		    

		    lengthVp[ri]=t.rhr.data[0].multiset.size();

		    if ((lengthUp[ri]+lengthVp[ri])!=rhsl) {
		    	cout << "Error: Different lengths of RHS in rule" << (ri-rules[rbi]) << endl;
		    	return false;
		    }

			for (auto it = t.rhr.data[0].multiset.begin(); it!=t.rhr.data[0].multiset.end(); it++) {
				obj_rhs[object_pointer]=ALPHABET.getObjectId(it->first.str()).getId();
				obj_rhs_m[object_pointer++]=it->second.raw();
			}		    

	#ifdef BIN_DEBUG	    
		    cout << "\tRule " << ri-rules[rbi] << ", RHS=";
		    int o_it=object_pointer-rhsl;
		    for (int o=0; o<lengthUp[ri]; o++) {
				cout << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_rhs_m[o_it] << " ";o_it++;
		    }
		    cout <<"[ ";
		    for (int o=0; o<lengthVp[ri]; o++) {
				cout << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << "*" << obj_rhs_m[o_it] << " ";o_it++;
		    }
		    cout <<"]^" << (int)charge_beta(charge[rbi]) << "_" << active_membrane[rbi] << endl;
	#endif
		    ri++;
		}
		rbi++;
    }
    
    rbi = 0;
    for (auto rb:environ_blocks) {
		unsigned int bid=rbi+options->num_rule_blocks;
#ifdef BIN_DEBUG	
		cout << "Env Block " << bid << ":"<< endl;
#endif	
		uint ri=rules[bid];
		for (auto r: rb.second) {
			for (auto m: r.rhr.data[0].data) {	
				if (m.multiset.size()==0)
					continue;
				else if (m.multiset.size()==1) {
					obj_rhs[object_pointer]=ALPHABET.getObjectId(m.multiset.begin()->first.str()).getId();
					obj_rhs_m[object_pointer++]=environment_labels_id.at(m.label[0]);//m.multiset.begin()->second.raw();
				}
				else {
					cout << "Error: environment rules cannot have more than one object in each environment at the RHS: " << m <<  endl;
					return false;
				}
			}
#ifdef BIN_DEBUG
		    cout << "\tRule " << ri-rules[bid] << ", RHS=";
		    int o_it=object_pointer-lengthUp[ri];
		    for (int o=0; o<lengthUp[ri]; o++) {
				cout << "(" << id_objects[obj_rhs[o_it]]/*obj_lhs[o_it]*/ << ")_" << obj_rhs_m[o_it] << " ";o_it++;
		    }
		    cout << endl;
#endif
	    	ri++;
		}
		rbi ++;
    }    
	/************************************************/


    /************************************************/
    /* V. Entering reading initial multiset         */
#ifdef BIN_DEBUG
    cout << "COLLECTING INITIAL CONFIGURATION: " << endl;
#endif
    ini_multiset = new unsigned int* [options->num_environments*options->num_membranes];    
    ini_charge = new char [options->num_environments*options->num_membranes];
    ini_info = new unsigned int [options->num_environments*options->num_membranes];
    
    // Initialize everything to 0
    memset(ini_multiset,0,options->num_environments*options->num_membranes*sizeof(uint));
    memset(ini_charge,0,options->num_environments*options->num_membranes*sizeof(char));
    memset(ini_info,0,options->num_environments*options->num_membranes*sizeof(uint));

	for (auto me:ps.multisets) { // for every defined initial multiset
		uint q;
		if (me.first[0] == me.first[1])  // special case of initialization of environments
			q = 0;
		else
			q = membrane_labels_id.at(me.first[0]); // membrane label
		uint e = environment_labels_id.at(me.first[1]); // environment label		

		// number of objects defined in the initial multiset
		unsigned int num_objs = ini_info[INI_OFFSET_PL5(e,q)] = me.second.size();	    
	    
	    if (num_objs>0) {	    
	    	ini_multiset[INI_OFFSET_PL5(e,q)]=new unsigned int [num_objs*2];

	    	int oit = 0;
			for (auto ompair:me.second) { // each object,multiplicity pair
				uint obj = ALPHABET.getObjectId(ompair.first.str()).getId(); // get object ID
				size_t mult = ompair.second.raw(); // get multiplicity

				ini_multiset[INI_OFFSET_PL5(e,q)][oit*2]=obj; // write it in this special array
				ini_multiset[INI_OFFSET_PL5(e,q)][oit*2+1]=mult;
				oit++;
			}
		}
	}	
	
	for (auto e:ps.structure.data) {
		int eid = environment_labels_id.at(e.label[0]);
		traverse_charges_membranes(e.data[0],membrane_labels_id,ini_charge+(eid*options->num_membranes));		
	}
    
#ifdef BIN_DEBUG
    for (int e=0; e<options->num_environments; e++) {
		cout << "Env "<<e<<endl;
		for (int q=0; q<options->num_membranes; q++) {
		    //cout << "\tMembr "<<q<<" ("<<INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET_PL5(e,q)])<<" #objects):";
			cout << "\tMembr "<<q<<" (charge: " << (int)ini_charge[INI_OFFSET_PL5(e,q)] << ", "<<ini_info[INI_OFFSET_PL5(e,q)]<<" #objects):";
		    //for (int o=0;o<(INITIAL_NUMBER_OBJECTS(ini_info[INI_OFFSET_PL5(e,q)]))*2;o+=2) {
			for (int o=0;o<(ini_info[INI_OFFSET_PL5(e,q)])*2;o+=2) 
				cout << id_objects[ini_multiset[INI_OFFSET_PL5(e,q)][o]] <<"*"<< ini_multiset[INI_OFFSET_PL5(e,q)][o+1]<< " ";
		    
		    cout << endl;
		}	
    }
#endif
    /************************************************/
    
    //options->modular = false;
    /*cout << "Pi index: ";
    for (int m=0;m<options->modules; m++) cout << ", PI[" << m << "]=" << options->modules_pi_index[m];
    cout << endl <<"Env index: ";
    for (int m=0;m<options->modules; m++) cout << ", ENV[" << m << "]=" << options->modules_env_index[m];
    cout << endl;*/

    return true;
}



