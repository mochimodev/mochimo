/**
 * @private
 * @headerfile block.h <block.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BLOCK_C
#define MOCHIMO_BLOCK_C


#include "block.h"

/* internal support */
#include "chain.h"
#include "error.h"
#include "ledger.h"
#include "protocol.h"

/* external support */
#include "extlib.h"
#include "extmath.h"
#include "sha256.h"
#include <string.h>

/** Blockchain archive directory (configurable option) */
char *Bcdir_opt = "bc";
/** Mining address filename (configurable option) */
char *Maddr_opt = "maddr.dat";

/**
 * Generate a neo-genesis block. Requires an opened Ledger.
 * Uses the last trailer in the Tfile as state (MUST BE 0x..ff).
 * If the ledger input filename is NULL, the internal ledger tree is used.
 * @param fname Filename of output block (typically "ngblock.dat")
 * @param lfname Filename of the input ledger file to wrap
 * @param tfname Filename of Tfile to use as state
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int neogen(char *fname, char *lfname, char *tfname)
{
   static word8 one[8] = { 1, 0 };

   LENTRY le;              /* ledger entry buffer */
   SHA256_CTX bctx;        /* (entire) block hash */
   BTRAILER bt, prev_bt;   /* neo-genesis and previous block trailers */
   FILE *fp, *lfp;         /* FILE pointers neo-genesis and ledger files */
   size_t total;           /* size counters */
   long llen;              /* ledger length */
   word32 hdrlen;          /* header length for neo block */
   char file[FILENAME_MAX];

   /* read and check trailer is 0x..ff */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;
   if (prev_bt.bnum[0] != 0xff) goto FAIL_BNUM;

   /* check for specified ledger filename */
   if (lfname == NULL) {
      /* try internal ledger */
      if (le_transpose() != VEOK) return VERROR;
      snprintf(file, FILENAME_MAX, "%s.0", Lefname_opt);
      lfname = file;
   }

   /* open ledger read-only and neo-genesis write-only */
   lfp = fopen(lfname, "rb");
   fp = fopen(fname, "wb");
   if (fp == NULL || lfp == NULL) goto FAIL_IO;
   /* fseek() to compute ledger length and check */
   if (fseek(lfp, 0L, SEEK_END) != 0) goto FAIL_IO;
   if ((llen = ftell(lfp)) == EOF) goto FAIL_IO;
   if (llen == 0 || (llen % sizeof(le)) != 0) goto FAIL_IO_FLEN;

   /* add the ledger length to the size of the header length field */
   hdrlen = (word32) llen + 4;
   /* begin the Neo-Genesis block by writing the header length to it */
   if (fwrite(&hdrlen, 4, 1, fp) != 1) goto FAIL_IO;
   /* begin block hash with the header length field */
   sha256_init(&bctx);
   sha256_update(&bctx, &hdrlen, 4);

   /* copy ledger entries it neo-genesis block and hash into bctx */
   for (rewind(lfp), total = 0; ; total += sizeof(le)) {
      if (fread(&le, sizeof(le), 1, lfp) != 1) break;
      /* hash entry and write to neo-genesis */
      sha256_update(&bctx, &le, sizeof(le));
      if (fwrite(&le, sizeof(le), 1, fp) != 1) goto FAIL_IO;
   }
   /* check break condition was not an error */
   if (ferror(lfp)) goto FAIL_IO;
   /* count total copied data matches ledger size */
   if (total != (size_t) llen) goto FAIL_IO_EOF;
   /* ledger no longer required */
   fclose(lfp);
   lfp = NULL;

   /* build block trailer */
   memset(&bt, 0, sizeof(bt));
   memcpy(bt.phash, prev_bt.bhash, HASHLEN);
   add64(prev_bt.bnum, one, bt.bnum);
   put32(bt.stime, get32(prev_bt.stime));
   put32(bt.time0, get32(prev_bt.time0));
   put32(bt.difficulty, get32(prev_bt.difficulty));
   sha256_update(&bctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&bctx, bt.bhash);
   /* write block trailer to neo-genesis block */
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;

   fclose(fp);

   /* success */
   return VEOK;

/* error handling -- remove neo-genesis block */
FAIL_BNUM: set_errno(EMCM_BNUM); return VERROR;
FAIL_IO_FLEN: set_errno(EMCM_FILELEN); goto FAIL_IO;
FAIL_IO_EOF: set_errno(EMCM_EOF);
FAIL_IO:
   if (lfp) fclose(lfp);
   if (fp) fclose(fp);
   remove(fname);
   return VERROR;
}  /* end neogen() */

/**
 * Validate an open pseudo-block.
 * @param fp Open FILE pointer to validate
 * @param prev_btp Pointer to previous block trailer
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int pseudo_val_fp(FILE *fp, BTRAILER *prev_btp)
{
   static word8 one[8] = { 1, 0 };

   BTRAILER bt;
   SHA256_CTX ctx;
   long blocklen;
   word32 hdrlen;
   word8 chash[HASHLEN];
   word8 bnum[8];

   /* read block header, trailer and file length */
   if (fseek(fp, 0L, SEEK_SET)) return VERROR;
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) return VERROR;
   if (fread(&bt, sizeof(bt), 1, fp) != 1) return VERROR;
   if (fseek(fp, 0L, SEEK_END)) return VERROR;
   if ((blocklen = ftell(fp)) == EOF) return VERROR;

   /* compute and check block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, chash);
   if (memcmp(bt.bhash, chash, HASHLEN) != 0) goto BAD_BHASH;

   /* check header/trailer lengths */
   if (hdrlen != sizeof(hdrlen)) goto BAD_HDRLEN;
   if (blocklen != sizeof(hdrlen) + sizeof(bt)) goto BAD_TLRLEN;
   /* check zeros in block trailer*/
   if (get32(bt.tcount) != 0) goto BAD_TCOUNT;
   if (!iszero(bt.mroot, 32)) goto BAD_MROOT;
   if (!iszero(bt.nonce, 32)) goto BAD_NONCE;

   /* check block num, hash, and difficulty */
   add64(prev_btp->bnum, one, bnum);
   if (cmp64(bt.bnum, bnum) != 0) goto BAD_BNUM;
   if (memcmp(bt.phash, prev_btp->bhash, HASHLEN)) goto BAD_PHASH;
   if (get32(bt.difficulty) != next_difficulty(prev_btp)) goto BAD_DIFF;

   /* check block times */
   if (get32(bt.time0) != get32(prev_btp->stime)) goto BAD_TIME0;
   if (get32(bt.stime) != get32(prev_btp->stime) + BRIDGE) goto BAD_STIME;
   if (!iszero(bt.mfee, 8)) goto BAD_MFEE;

   /* pseudo-block is valid */
   return VEOK;

/* block format violation handling */
BAD_BHASH: set_errno(EMCM_BHASH); return VEBAD;
BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TLRLEN: set_errno(EMCM_TLRLEN); return VEBAD;
BAD_TCOUNT: set_errno(EMCM_TCOUNT); return VEBAD;
BAD_MROOT: set_errno(EMCM_MROOT); return VEBAD;
BAD_NONCE: set_errno(EMCM_NONCE); return VEBAD;
BAD_BNUM: set_errno(EMCM_BNUM); return VEBAD;
BAD_PHASH: set_errno(EMCM_PHASH); return VEBAD;
BAD_DIFF: set_errno(EMCM_DIFF); return VEBAD;
BAD_TIME0: set_errno(EMCM_TIME0); return VEBAD;
BAD_STIME: set_errno(EMCM_STIME); return VEBAD;
BAD_MFEE: set_errno(EMCM_MFEE); return VEBAD;
}  /* end pseudo_val_fp() */

/**
 * Validate a pseudo-block.
 * @param pfname Filename of pseudo-block to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int pseudo_val(char *pfname, char *tfname)
{
   BTRAILER prev_bt;
   FILE *fp;
   int ecode;

   /* read last Tfile trailer for validation against */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;

   /* open pseudo-block file for validation */
   fp = fopen(pfname, "rb");
   if (fp == NULL) return VERROR;
   ecode = pseudo_val_fp(fp, &prev_bt);
   fclose(fp);

   return ecode;
}  /* end pseudo_val() */

/**
 * Generate a pseudo-block. Uses the last trailer in the Tfile as state.
 * @param fname Filename of output block (typically "pblock.dat")
 * @param tfname Filename of Tfile to use as state
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int pseudo(char *fname, char *tfname)
{
   static const word32 hdrlen = 4;
   static word8 one[8] = { 1, 0 };

   SHA256_CTX ctx;
   BTRAILER bt, prev_bt;
   FILE *fp;
   word32 time0;

   /* read last trailer in Tfile (for state) */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;

   /* open pseudo-block file and write hdrlen */
   fp = fopen(fname, "wb");
   if (fp == NULL) return VERROR;
   if (fwrite(&hdrlen, sizeof(hdrlen), 1, fp) != 1) goto FAIL_IO;

   /* fill block trailer with appropriate pseudo-data */
   time0 = get32(prev_bt.stime);
   memset(&bt, 0, sizeof(bt));
   memcpy(bt.phash, prev_bt.bhash, HASHLEN);
   add64(prev_bt.bnum, one, bt.bnum);
   put32(bt.time0, time0);
   put32(bt.difficulty, next_difficulty(&prev_bt));
   put32(bt.stime, time0 + BRIDGE);

   /* compute pseudo-block hash directly into block trailer */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, bt.bhash);

   /* write block trailer to pseudo-block file */
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;

   fclose(fp);

   /* success */
   return VEOK;

/* error handling -- remove pblock */
FAIL_IO:
   fclose(fp);
   remove(fname);
   return VERROR;
}  /* end pseudo() */

int update_block(char *fname)
{
   static word8 one[8] = { 1, 0 };

   BTRAILER bt;
   word8 bnum[8] = { 0 };
   int ecode;

   /* read block trailer */
   if (read_trailer(&bt, fname) != VEOK) return VERROR;

   /* perform appropriate update actions, depending on data type */
   if (bt.bnum[0] == 0) {
      /* extract ledger from neo-genesis block */
      ecode = le_extract(fname);
      if (ecode) return ecode;
      /* read block number of Tfile */
      if (read_bnum(bnum, fname) != VEOK) return VERROR;
      /* trim tfile to neo-genesis block */
      if (cmp64(bt.bnum, bnum) <= 0) {
         /* trim tfile for append trailer -- reset weight */
         if (sub64(bt.bnum, one, bnum)) goto FAIL_UNDERFLOW;
         if (trim_tfile("tfile.dat", bnum) != VEOK) return VERROR;
         memset(Weight, 0, sizeof(Weight));
         weigh_tfile("tfile.dat", Weight);
      }
   } else if (get32(bt.tcount) > 0) {
      /* update ledger with transaction data */
      ecode = le_update(fname);
      if (ecode) return ecode;
   }

   /* append trailer to Tfile */
   if (append_tfile(fname, "tfile.dat") != VEOK) return VERROR;

   /* update protocol state */
   put64(Cblocknum, bt.bnum);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   memcpy(Prevhash, bt.phash, HASHLEN);
   add_weight(Weight, bt.difficulty[0], bt.bnum);

   /* perform neogenesis block update -- as necessary */
   if (Cblocknum[0] == 0xff) {
      /* generate neogenesis file */
      if (neogen("neogen.dat", NULL, "tfile.dat") != VEOK) return VERROR;
      /* append trailer to Tfile -- NOT YET */
      /* if (append_tfile(fname, "tfile.dat") != VEOK) return VERROR; */
   }

   return VEOK;

/* error handling */
FAIL_UNDERFLOW: set_errno(EMCM_MATH64_UNDERFLOW); return VERROR;
}

/**
 * Validate any open blockchain file. With the exception of the Genesis
 * block, all other blockchain files (Hashed or WOTS+) can be identified
 * by the value of the 32-bit block header length at the start of the file:
 * - 4, pseudo-block file
 * - 12, Hashed neo-genesis file
 * - 76, Hashed blockchain file
 * - 2220, WOTS+ blockchain file
 * - >2220, WOTS+ neo-genesis file
 * @param fp Open FILE pointer to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int validate_block_fp(FILE *fp, char *tfname)
{
   BTRAILER prev_bt;
   word32 hdrlen;

   /* read previous block trailer from supplied file */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;

   /* seek to beginning and read header length */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) {
      if (feof(fp)) set_errno(EMCM_EOF);
      return VERROR;
   }

   /* determine appropriate validation routine */
   switch (hdrlen) {
      case 4: return pseudo_val_fp(fp, &prev_bt);
   /* case 12: return neogen_val_fp(fp, tfname);
      case 76: return block_val_fp(fp, &prev_bt);  */
      case 2220: return blockw_val_fp(fp, &prev_bt);
      default: return neogenw_val_fp(fp, tfname);
   }  /* end switch (hdrlen) */
}  /* end validate_block_fp() */

/**
 * Validate any blockchain file. With the exception of the Genesis block,
 * all other blockchain files (Hashed or WOTS+) can be identified by the
 * value of the 32-bit block header length at the start of the file:
 * - 4, pseudo-block file
 * - 12, Hashed neo-genesis file
 * - 76, Hashed blockchain file
 * - 2220, WOTS+ blockchain file
 * - >2220, WOTS+ neo-genesis file
 * @param bcfname Filename of blockchain file to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int validate_block(char *bcfname, char *tfname)
{
   word32 hdrlen;

   /* read header length */
   if (read_hdrlen(&hdrlen, bcfname) != VEOK) return VERROR;
   /* determine appropriate validation routine */
   switch (hdrlen) {
      case 4: return pseudo_val(bcfname, tfname);
   /* case 12: return neogen_val(bcfname, tfname);
      case 76: return block_val(bcfname, tfname);  */
      case 2220: return blockw_val(bcfname, tfname);
      default: return neogenw_val(bcfname, tfname);
   }  /* end switch (hdrlen) */
}  /* end validate_block() */

/* end include guard */
#endif
