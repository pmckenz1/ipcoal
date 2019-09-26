#!/usr/bin/env python

"Jitted functions for fast invariants operations"

import numpy as np
from numba import njit


@njit()
def count_matrix_int(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.int64)
    add = np.int64(1) 
    for idx in range(quartsnps.shape[0]):
        i = quartsnps[idx, :]
        arr[(4 * i[0]) + i[1], (4 * i[2]) + i[3]] += add    
    return arr


@njit()
def count_matrix_float(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.float32)
    add = np.float32(1)
    for idx in range(quartsnps.shape[0]):
        i = quartsnps[idx, :]
        arr[(4 * i[0]) + i[1], (4 * i[2]) + i[3]] += add    
    return arr  # / arr.max()    


@njit()
def mutate_jc(geno, ntips):
    """
    mutates sites with 1 into a new base in {0, 1, 2, 3}
    """
    allbases = np.array([0, 1, 2, 3])
    for ridx in np.arange(geno.shape[0]):
        snp = geno[ridx]
        if snp.sum():
            init = np.empty(ntips, dtype=np.int64)
            init.fill(np.random.choice(allbases))
            notinit = np.random.choice(allbases[allbases != init[0]])
            init[snp.astype(np.bool_)] = notinit
            return init
    # return dtypes must match
    return np.zeros(0, dtype=np.int64)  

@njit
def base_to_int(geno_arr):
    basetrans = np.zeros(len(geno_arr),dtype=np.int8)
    for basenum in np.arange(len(geno_arr)):
        geno_arr[basenum]
        if  geno_arr[basenum] == 'A':
            basetrans[basenum] = 0
        if geno_arr[basenum] == 'G':
            basetrans[basenum] = 1
        if geno_arr[basenum] == 'C':
            basetrans[basenum] = 2
        if geno_arr[basenum] == 'T':
            basetrans[basenum] = 3
    return(basetrans)

@njit
def base_to_int_genes(geno_arr):
    basetrans = np.zeros(geno_arr.shape,dtype=np.int8)
    for seqnum in np.arange(geno_arr.shape[0]):
        for basenum in np.arange(geno_arr.shape[1]):
            if  geno_arr[seqnum,basenum] == 'A':
                basetrans[seqnum,basenum] = 0
            if geno_arr[seqnum,basenum] == 'G':
                basetrans[seqnum,basenum] = 1
            if geno_arr[seqnum,basenum] == 'C':
                basetrans[seqnum,basenum] = 2
            if geno_arr[seqnum,basenum] == 'T':
                basetrans[seqnum,basenum] = 3
    return(basetrans)