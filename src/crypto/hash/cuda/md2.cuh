/*
 * md2.cuh CUDA Implementation of MD2 digest       
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 * Based on the public domain Reference Implementation in C, by 
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 */

#pragma once
#include "config.h"
void mcm_cuda_md2_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);
