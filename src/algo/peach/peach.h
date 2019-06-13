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

#define HASHLENMID 	                   16
#define HASHLEN                        32
#define TILE_ROWS                      32
#define TILE_LENGTH                  1024 // (TILE_ROWS * HASHLEN)
#define TILE_TRANSFORMS                 8
#define MAP                       1000000
#define MAP_LENGTH    (TILE_LENGTH * MAP)
#define JUMP                            8

#define PEACH_DEBUG                     0
