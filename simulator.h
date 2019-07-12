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
 * simulator.h
 *
 *  Created on: 31/07/2011
 *      Author: miguel
 */

#ifndef __SIMULATOR__
#define __SIMULATOR__

class Simulator {
public:

	/* Methods for running */
	virtual bool run()=0;
	virtual bool step(int k)=0;
	
	/* Methods for querying data */
	virtual unsigned int get_object_value(  int obj, 
						int membr, 
						int env,
						int sim)=0;
	
	virtual char get_membrane_charge(  int membr,
					   int env,
					   int sim)=0;
};

#endif /* SIMULATOR_H_ */
