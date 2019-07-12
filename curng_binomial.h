/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2015  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Authors: Adolfo Gastalver-Rubio, Miguel Ángel Martínez-del-Amor
    
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

/* Information:
 * 
 * File Name:           curng_binomial.h
 * Dependencies:        time.h, curand_kernel.h (CuRAND library provided by NVIDIA (R))
 * Version:             1.0.2
 * 
 * University:          University Of Seville (US)
 * Department:          Dpt. of Computer Science and Artificial Intelligence (CCIA)
 * Group:               Research Group on Natural Computing (RGNC) (www.gcn.us.es)
 * Member:              A. Gastalver, M. Martínez-del-Amor
 * Date:                03/22/2012
 * Last Modified:	    06/20/2015
 */

/* How to use (follow the steps in order):
 * 1. On host:
 *      Call the method "binomial_init" (parameters grid and block dimensions)
 *              -> actually fixed for dcba/mecohpc algorithm: (to generalize: just uncomment line 35 and comment line 33)
 *                      blockIdx.y = Simulation
 *                      blockIdx.x = Environment
 *                      threadIdx.x = Block
 *              -> it initializes the binomial generators (reserving memory)
 *              -> if recalled, last memory reserved will be free and new memory will be reserved according to new dimensions
 * 2. On device:
 *      Call the method "binomial_random" (parameters n and p needed).
 *              -> it generates an unsigned int number between 0 and n (both included) by binomial distribution
 * 3. On host: (recommended)
 *      Call the method "binomial_free" (without parameters) to get the last memory reserved free.
 */


#ifndef CURNG_BINOMIAL_H
#define CURNG_BINOMIAL_H

#include <sys/time.h>
#include <curand_kernel.h>
#include <iostream>

using namespace std;

//Comment to generalize:
#define __IDX ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x);
//Uncomment to generalize:
//#define __IDX ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x);

__device__ curandStateXORWOW_t * curng_binomial_states_k;
curandStateXORWOW_t * curng_binomial_states;

// TODO: Use state depending functions (avoiding global memory accesses)
//typedef curandStateXORWOW_t * curng_binomial_state_t;

int curng_binomial_first = 0;

/*Based on the article: 
 * [Binomial random variate generation] - 1988
 * Voratas Kachitvichyanukul	Univ. of Iowa, Iowa City
 * Bruce W. Schmeiser	Purdue Univ., West Lafayette, IN
 * And the modification of Fishman [17]
 */
static inline __device__ unsigned int curng_binomial_binv(unsigned int n, float p){
    unsigned int idx = __IDX;
    curandStateXORWOW_t localState = curng_binomial_states_k[idx];
    unsigned int x = 0;
    
    //float p1 = fmin(p,1.f-p);
    //Algorithm starts here. (attribute 'x' assigned to '0')
    float q = 1.f - p;
    double s = fdividef(p,q);    //Divide floating points
    double a = (n+1)*s;
    double r = powf(q,n);        //Pow floating points
    float u = curand_uniform(&localState);
    while(u > r){
        u = u - r;
        x++;
        r = (fdividef(a,((float)x)) - s) * r;
	if (r<=0.0)
		break;
    }
    //Algorithm finishes here.
    if (x>n) 
	x=n;
    
    curng_binomial_states_k[idx] = localState;
    //x = (p<0.5f)? x : n-x;
    return x;
}

/* Normal approximation of binomial distribution: N(np, np(1-p))
 * Condition: n*min(p,1-p) > 30
 * Standardizing normal random variables 
 * z = (x - np) / sqrt(np(1-p) <=> x = (z * sqrt(np(1-p))) + np
 */
static inline __device__ unsigned int curng_binomial_norm(unsigned int n, float p){
    unsigned int idx = __IDX;
    curandStateXORWOW_t localState = curng_binomial_states_k[idx];
    unsigned int x = 0;
    
    float z = curand_normal(&localState);
    float mean = ((float)n)*p;
    float stddev = sqrt(mean*(1.-p));
    x = (unsigned int)((z*stddev)+mean);
    
    if (x>n) 
	x=n;
    
    curng_binomial_states_k[idx] = localState;
    return x;
}

static inline __device__ unsigned int curng_binomial_random(unsigned int n, float p){
    unsigned int k = 0;
    /*if (p==1.0f)
	k=n;
    else if (p==0.0f)
	k=0;
    else */
    if (n*fmin(p,1.0f-p) < 10)
        k = curng_binomial_binv(n,p);
    else{
        k = curng_binomial_norm(n,p);
    }
    return k;
}

__global__ void curng_binomial_init_kernel(unsigned int time){
    unsigned int idx = __IDX;
    curand_init (time , idx , 5000 , &curng_binomial_states_k[idx]) ;
}

void curng_binomial_free(){
    cudaFree(curng_binomial_states);
    curng_binomial_first = 0;
}

void curng_binomial_init(dim3 griddim, dim3 blockdim) {
    const size_t sz = (griddim.x * griddim.y * blockdim.x * blockdim.y * blockdim.z)* sizeof(curandStateXORWOW_t);
   
    if(curng_binomial_first > 0){
        curng_binomial_free();
    }

    curng_binomial_first++;

    cudaMalloc((void **)&curng_binomial_states, sz);
    
    // Old line for CUDA 4
    //cudaMemcpyToSymbol("curng_binomial_states_k", &curng_binomial_states, sizeof(curandState *), size_t(0),cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(curng_binomial_states_k, &curng_binomial_states, sizeof(curandState *), size_t(0),cudaMemcpyHostToDevice);
    
#ifndef DEBUG
    struct timeval tval;
    gettimeofday(&tval,NULL);
    unsigned int timescale = tval.tv_usec;
#else
    unsigned int timescale = 0;
#endif
    curng_binomial_init_kernel<<<griddim,blockdim>>>(timescale);
    getLastCudaError("Kernel initiating curng_binomial launch failure");
    cudaDeviceSynchronize();
}

unsigned int curng_sizeof_state(unsigned int num_threads) {
	return num_threads*sizeof(curandStateXORWOW_t);
}


#endif	
