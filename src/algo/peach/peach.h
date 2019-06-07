/*
 * peach.h  FPGA-Tough CPU Mining Algo Definitions
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 05 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 */

//#include "../../config.h"

#define HASHLEN 	              32
#define TILE_FACTOR               32
#define TILE (TILE_FACTOR * HASHLEN)
#define TILE_TRANSFORM             8
#define MAP                  1000000
#define MAP_LENGTH      (TILE * MAP)
#define JUMP                       8

