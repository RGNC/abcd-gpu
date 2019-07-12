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
 * PDP_psystem_sab.h
 *
 *  Created on: 21/11/2011
 *      Author: miguel
 */

#ifndef PDP_PSYSTEM_SAB_H_
#define PDP_PSYSTEM_SAB_H_

#include "pdp_psystem.h"


/* Class of P system structures based on struct of arrays */
class PDP_Psystem_SAB: public PDP_Psystem {
public:
	PDP_Psystem_SAB(PDP_Psystem_source* PDPps_source);
	void print();

//private:

	/*
	 * Definitions for data implementation
	 */

	typedef unsigned int LHS_INI;
	typedef unsigned int LHS_LENGTH;
	typedef unsigned int RHS_INI;
	typedef unsigned int RHS_LENGTH;
	typedef unsigned int RULE_INI;
	typedef unsigned int RULE_LENGTH;
	typedef unsigned int MEMBRANE;
	typedef char CHARGE;
	typedef unsigned int OBJECT;
	typedef unsigned int MULTIPLICITY;
	typedef float PROBABILITY;

	/*
	 * Definition file for structures
	 */

	struct _ruleblock {
		LHS_INI *lhs_ini;
		LHS_LENGTH *lhs_length;
		RULE_INI *rule_ini;
		RULE_LENGTH *rule_length;
		MEMBRANE *active_membrane;
		CHARGE *lhs_charge;
		CHARGE *rhs_charge;
	};

	typedef struct _ruleblock Ruleblock;

	struct _rule_object {
		OBJECT *object;
		MULTIPLICITY *multiplicity;
		MEMBRANE *membrane;
	};

	typedef struct _rule_object Lhs;

	struct _rule {
		RHS_INI *rhs_ini;
		RHS_LENGTH *rhs_length;
	};

	typedef struct _rule Rule;

	typedef PROBABILITY *Probability;

	typedef struct _rule_object Rhs;

	typedef MULTIPLICITY *NR;

	typedef MULTIPLICITY *NB;

	struct _configuration {
		MULTIPLICITY * multiset;
		unsigned int multiset_size;
		CHARGE * membrane;
		unsigned int membrane_size;
	};

	typedef struct _configuration Configuration;

	/*
	 * General structure
	 */

	struct _structures {
		Ruleblock ruleblock;
		unsigned int ruleblock_size;
		Lhs lhs;
		unsigned int lhs_size;
		Rule rule;
		unsigned int rule_size,env_rule_size;
		Probability probability;
		unsigned int probability_size;
		Rhs rhs;
		unsigned int rhs_size;
		NR nr;
		unsigned int nr_size;
		NB nb;
		unsigned int nb_size;
		Configuration configuration;
	};

	typedef struct _structures* Structures;

	Structures structures;
};

#endif /* PDP_PSYSTEM_SAB_H_ */
