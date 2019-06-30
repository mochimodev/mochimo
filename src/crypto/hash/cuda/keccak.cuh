/*
 * keccak.cuh CUDA Implementation of BLAKE2B Hashing
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 */

#pragma once
#include "config.h"
void mcm_cuda_keccak_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch);
