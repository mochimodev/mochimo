/* init.c  High-level Initialisation functions (included from mochimo.c)
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 11 February 2018
 *
*/

#include "network.h"
#include "util.c"
#include "data.c"
#include "syncup.c"

#ifndef HTTPSTARTPEERS
#define HTTPSTARTPEERS "https://new-api.mochimap.com/network/peers/start"
#endif

int check_directory(char *dirname)
{
   char fname[FILENAME_MAX];

   mkdir_p(dirname);
   snprintf(fname, FILENAME_MAX, "%s/chkfile", dirname);
   if (ftouch(fname) == VEOK) return remove(fname);
   return perrno(errno, "Permission failure, %s", dirname);
}

/**
 * Initialize the server/client from any state
 * after executing the gomochi script. */
int init(void)
{
   /* static word8 FortyEight[8] = { 48, }; */
   char fname[FILENAME_MAX];
   char bnumstr[17], weightstr[65];
   word32 peer, qlen, quorum[MAXQUORUM];
   word8 nethash[HASHLEN], peerhash[HASHLEN];
   word8 netweight[32], netbnum[8]; //, bnum[8];
   /* BTRAILER bt; */
   NODE node;  /* holds peer tx.cblock and tx.cblockhash */
   int result, status /*, count */ ;
   word8 highblock[8];

   show("init");
   plog("Initializing...");
   status = VEOK;
   Running = 1;
   Ininit = 1;

   /* ensure appropriate directories and permissions exist */
   if (check_directory(Bcdir) || check_directory(Spdir)) return VERROR;

   /* update coreip.lst where available */
   if (fcopy("../coreip.lst", "coreip.lst") != VEOK) {
      if (!fexists("coreip.lst")) pwarn("missing coreip.lst");
   }
   /* update maddr.dat - use maddr.MAT as last resort only */
   if (fcopy("../maddr.dat", "maddr.dat") != VEOK) {
      if (!fexists("maddr.dat")) {
         if (fcopy("../maddr.mat", "maddr.dat") != VEOK) {
            return perr("Failed to copy mining address");
         } else pwarn("using maddr.MAT (the founder's mining address)");
      }
   }
   /* restore core chain files if any do not exist */
   snprintf(fname, FILENAME_MAX, "%s/b0000000000000000.bc", Bcdir);
   if (!fexists("tfile.dat") || !fexists(fname)) {
      pdebug("Core chain files compromised, attempting restoration...");
      if (fcopy("../genblock.bc", fname) != VEOK) {
         return perr("Failed to restore %s from ../genblock.bc", fname);
      } else if (fcopy("../tfile.dat", "tfile.dat") != VEOK) {
         return perr("Failed to restore tfile.dat from ../tfile.dat");
      }
   }
   /* open ledger or extract from genesis block */
   if (!fexists("ledger.dat") || le_open("ledger.dat", "rb") != VEOK) {
      pdebug("Extracting ledger from ../genblock.bc ...");
      if (le_extract("../genblock.bc", "ledger.dat") != VEOK) {
         return perr("Failed to extract ledger from ../genblock.bc");
      } else if (le_open("ledger.dat", "rb") != VEOK) { /* try again */
         return perr("Failed to open ledger.dat");
      }
   }

   /* Find the last block in bc/ and reset Time0, and Difficulty */
   if (reset_chain() != VEOK) return perr("reset_chain() failed");
   /* validate our own tfile.dat to compute Weight */
   if (tf_val("tfile.dat", highblock, Weight, 1)) {
      if (cmp64(Cblocknum, highblock) != 0) {
         perr("init(): %d %d", get32(Cblocknum), get32(highblock));
         perr("init(): %s", weight2hex(Weight));
         return perr("init(): bad tfile.dat -- gomochi!");
      }
   }

   /* fresh peer acquisition from mochimap */
   pdebug("downloading fresh peers from mochimap...");
   http_get(HTTPSTARTPEERS, "start.lst", STD_TIMEOUT);

   /* read peer lists */
   readpeers();
   readpink();

   /* shuffle recent peer list */
   shuffle32(Rplist, RPLISTLEN);

   /* scan entire network of peers */
   while(Running) {
      /* scan network for quorum and highest hash/weight/bnum */
      plog("Network scan...");
      qlen = scan_network(quorum, MAXQUORUM, nethash, netweight, netbnum);
      plog("Network scan resulted in %d/%d quorum members", qlen, MAXQUORUM);
      plog("  bnum= 0x%s, weight= 0x%s", val2hex(netbnum, 8, bnumstr, 17),
         val2hex(netweight, 32, weightstr, 65));
      if (qlen == 0 && iszero(Cblocknum, 8)) break; /* all alone... */
      else if (qlen < Quorum) {  /* insufficient quorum */
         plog("Insufficient quorum size, try again...");
         /* without considering the expansion of acceptable
          * quorum size, infinite loop is possible... */
         continue;
      } else shuffle32(quorum, qlen);
      if (!iszero(Cblocknum, 8)) {  /* we've got a chain */
         status = VEOK;  /* don't panic... EVERYTHING IS FINE! */
         result = cmp256(Weight, netweight);  /* compare network weight */
         if (result > 0) {
            pdebug("network weight compares lower");
            plog("\n... an overwhelming sense of power ...\n");
            break;  /* we're heavier, finish */
         } else if (result == 0 && cmp256(Cblockhash, nethash) == 0) {
            pdebug("network weight and hash compares equal");
            plog("\n... an overwhelming sense of belonging ...\n");
            break;  /* we're in sync, finish */
         } else {
            pdebug("network weight and hash compares equal");
            plog("\n... an overwhelming sense of confusion ...\n");
         }
         /* have we fallen behind or split from the chain? */
         while((peer = *quorum)) {  /* use quorum to check... */
            if (status == VEOK) {  /* chain status not yet known... */
               plog("Checking blockchain alignment...");
               if (get_hash(&node, peer, Cblocknum, peerhash) == VEOK) {
                  if (cmp256(Cblockhash, peerhash)) status = VEBAD; /* 2319! */
                  else status = VERROR;  /* we're just behind */
                  continue;  /* restart loop with new status */
               }
            } else if (status == VERROR) {  /* chain is fallen... */
               plog("Blockchain is aligned, perform catchup...");
               catchup(quorum, qlen);  /* try to catchup with blockchain */
               break;
            } else if (status == VEBAD) {  /* chain is split... */
               plog("2319!!! CHAIN FORK DETECTED...");
               /* attempt chain recovery... DISABLED FOR NOW
               put64(bnum, cmp64(Cblocknum, netbnum) > 0 ? netbnum : Cblocknum);
               if (sub64(bnum, FortyEight, bnum)) break;
               count = readtf(&bt, get32(bnum), get32(FortyEight));
               if (count % sizeof(BTRAILER)){
                  perr("init(): error reading tfile, count= %s", count);
                  break;
               } else {
                  // acquire same segment of Tfile as above and compare
                  if (get_hash(&node, peer, bnum, peerhash) == VEOK) {}
                  if (cmp256(Cblockhash, peerhash) == 0) {
                     if (syncup(bnum, netbnum, peer) == VEOK) break;
                  } else continue;
               } */
               break;
            }
            remove32(peer, quorum, MAXQUORUM, &qlen);
         }  /* ... did we catch up? */
         if (cmp256(Weight, netweight) >= 0) {
            if (cmp64(Cblocknum, netbnum) >= 0) break;
         }  /* ... whatever we did, it didn't work... */
      }  /* ... at this point, we might as well resync */
      if (!qlen) plog("Quorum members exhausted...");
      else {
         plog("Blockchain recovery failed: resync...");
         if (resync(quorum, &qlen, netweight, netbnum) == VEOK) break;
      }
      /* we're either out of quorum members, or our resync failed */
      plog("Resync failure: try again...\n\n");
   }

   plog("");
   write_global();
   Ininit = 0;
   return VEOK;
}  /* end init() */
