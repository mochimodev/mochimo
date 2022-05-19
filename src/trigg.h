/**
 * @file trigg.h
 * @brief Trigg's Proof-of-Work algorithm support.
 * @details Trigg's Algorithm uses classic AI techniques to establish proof
 * of work.  By expanding a semantic grammar through heuristic search and
 * combining that with material from the transaction array, we build the
 * TRIGG chain and solve the block as evidenced by the output of haiku
 * with the vibe of Basho...
 * ```
 *    a raindrop
 *    on sunrise air--
 *    drowned
 * ```
 * Emulate a PDP-10 running MACLISP (circa. 1971)...
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TRIGG_H
#define MOCHIMO_TRIGG_H


/* internal support */
#include "types.h"

/* external support */
#include "sha256.h"
#include "extlib.h"
#include "extint.h"

/* The features for the semantic grammar are
 * adapted from systemic grammar (Winograd, 1972). */

#define F_ING         1
#define F_INF         2
#define F_MOTION      4
#define F_VB          ( F_INT | F_INT | F_MOTION )

#define F_NS          8
#define F_NPL         16
#define F_N           ( F_NS | F_NPL )
#define F_MASS        32
#define F_AMB         64
#define F_TIMED       128
#define F_TIMEY       256
#define F_TIME        ( F_TIMED | F_TIMEY )
#define F_AT          512
#define F_ON          1024
#define F_IN          2048
#define F_LOC         ( F_AT | F_ON | F_IN )
#define F_NOUN        ( F_NS | F_NPL | F_MASS | F_TIME | F_LOC )

#define F_PREP        0x1000
#define F_ADJ         0x2000
#define F_OP          0x4000
#define F_DETS        0x8000
#define F_DETPL       0x10000
#define F_XLIT        0x20000

#define S_NL          ( F_XLIT + 1 )
#define S_CO          ( F_XLIT + 2 )
#define S_MD          ( F_XLIT + 3 )
#define S_LIKE        ( F_XLIT + 4 )
#define S_A           ( F_XLIT + 5 )
#define S_THE         ( F_XLIT + 6 )
#define S_OF          ( F_XLIT + 7 )
#define S_NO          ( F_XLIT + 8 )
#define S_S           ( F_XLIT + 9 )
#define S_AFTER       ( F_XLIT + 10 )
#define S_BEFORE      ( F_XLIT + 11 )

#define S_AT          ( F_XLIT + 12 )
#define S_IN          ( F_XLIT + 13 )
#define S_ON          ( F_XLIT + 14 )
#define S_UNDER       ( F_XLIT + 15 )
#define S_ABOVE       ( F_XLIT + 16 )
#define S_BELOW       ( F_XLIT + 17 )

/* Trigg algorithm parameters */

#define TCHAINLEN     312
#define HAIKUCHARLEN  256
#define MAXDICT       256
#define MAXDICT_M1    255
#define MAXH          16
#define NFRAMES       10

typedef struct {
  word8 tok[12];  /**< word token */
  word32 fe;      /**< semantic features */
} DICT;  /**< Dictionary entry with semantic grammar features */

/* Check Trigg's Proof of Work without passing the final hash */
#define trigg_check(btp)  trigg_checkhash(btp, (btp)->difficulty[0], NULL)

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void *trigg_generate(void *out);
void *trigg_generate_fast(void *out);
char *trigg_expand(void *nonce, void *haiku);
int trigg_eval(void *hash, word8 diff);
int trigg_syntax(void *nonce);
int trigg_checkhash(BTRAILER *bt, word8 diff, void *out);
int trigg_solve(BTRAILER *bt, word8 diff, void *out);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
