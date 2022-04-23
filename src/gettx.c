/* gettx.c  Get validated transaction packet (TX) and helpers.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
*/

#include "syncup.c"

/* Look-up and return an address tag to np.
 * Called from gettx() opcode == OP_RESOLVE
 *
 * on entry:
 *     tag string at ADDR_TAG_PTR(np->tx.dst_addr)    tag to query
 * on return:
 *     np->tx.send_total = 1 if found, or 0 if not found.
 *     if found: np->tx.dst_addr has full found address with tag.
 *               np->tx.change_total has balance.
 *
 * Returns VEOK if found, else VERROR.
*/
int tag_resolve(TX *tx)
{
   static word8 one64[8] = { 1, 0, };
   static word8 zero64[8] = { 0, };
   word8 *addr;

   pdebug("tag_resolve() entered...");

   addr = tx->dst_addr;
   put64(tx->send_total, zero64);
   put64(tx->change_total, zero64);
   /* find tag in leger.dat */
   if(tag_find(addr, addr, tx->change_total, get16(tx->len)) == VEOK) {
      put64(tx->send_total, one64);  /* tag found flag */
      pdebug("tag_resolve() success: tag found");
      return VEOK;
   }
   pdebug("tag_resolve() failed: tag not found");
   return VERROR;
}  /* end tag_resolve() */

/* Mark NODE np in Nodes[] empty by setting np->pid to zero.
 * Adjust Nonline and Hi_node.
 * Caller must close np->sd if needed.
 */
int freeslot(NODE *np)
{
   if(np->pid == 0) {
      perr("*** NODE %ld already called freeslot() ***", (long) (np - Nodes));
      return VERROR;
   }
   pdebug("freeslot(): idx=%ld  ip = %-1.20s pid = %d", (long) (np - Nodes),
           ntoa(&np->ip, NULL), np->pid);
   Nonline--;
   np->pid = 0;
   /* Update pointer to just beyond highest used slot in Nodes[] */
   while(Hi_node > Nodes && (Hi_node - 1)->pid == 0) Hi_node--;
   if(Nonline < 0) {
      Nonline = 0;
      perr("Nonline < 0");
      return VERROR;
   }
   return VEOK;
}  /* end freeslot() */


/* A Basic block validator for catchup().
 * Every non-NG block should pass this test.
 * If it does not, the error is intentional (pink-list).
 * Returns: VEOK if valid, VERROR on errors, or VEBAD if bad.
 */
int bval2(char *fname, word8 *bnum, word8 diff)
{
   BTRAILER bt;
   word32 now;
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

   pdebug("bval2() entered");

   if(readtrailer(&bt, fname) != VEOK) {
      pdebug("bval2() readtrailer() failed!");
      return VERROR;
   }
   if(cmp64(bnum, bt.bnum) != 0) {
      pdebug("bval2() bnum != bt.bnum (VEBAD)");
      return VEBAD;
   }
   if(get32(bt.difficulty) != diff) {
      pdebug("bval2() bt.difficulty != diff, likely split chain");
   }
   /* Time Checks */
   if(get32(bt.stime) <= get32(bt.time0)) {
      pdebug("bval2() bt.stime <= bt.time0!");
      return VEBAD; /* bad time sequence */
   }
   now = time(NULL);
   if(get32(bt.stime) > (now + BCONFREQ)) {
      pdebug("bval2() bt.stime in future!");
      return VERROR;  /* future */
   }

   /* Solution Check */
   if(cmp64(bnum, v24trigger) > 0) { /* v2.4 Algo */
      if(peach_check(&bt)) {
         pdebug("bval2() peach() (VEBAD)");
         return VEBAD; /* block didn't validate */
      }
   }
   if(cmp64(bnum, v24trigger) <= 0) { /* v2.3 and prior */
      if(trigg_check(&bt)) {
         pdebug("bval2() trigg_check() (VEBAD)");
         return VEBAD;
      }
   }
   pdebug("bval2() returns VEOK");
   return VEOK;
}  /* end bval2() */


/* Count of trailers that fit in a TX: */
#define NTFTX (TRANLEN / sizeof(BTRAILER))

/* Handle contention
 * Returns:  0 = nothing else to do
 *           1 = do fetch block with child
 */
int contention(NODE *np)
{
   word32 splitblock;
   TX *tx;
   int result, j;
   BTRAILER *bt;

   pdebug("contention(): IP: %s", ntoa(&np->ip, NULL));

   tx = &np->tx;
   /* ignore low weight */
   if(cmp256(tx->weight, Weight) <= 0) {
      pdebug("contention(): Ignoring low weight");
      return 0;
   }
   /* ignore NG blocks */
   if(tx->cblock[0] == 0) {
      epinklist(np->ip);
      return 0;
   }

   if(memcmp(Cblockhash, tx->pblockhash, HASHLEN) == 0) {
      pdebug("contention(): get the expected block");
      return 1;  /* get block */
   }

   /* Try to do a simple catchup() of more than 1 block on our own chain. */
   result = VERROR;
   j = get32(tx->cblock) - get32(Cblocknum);
   if(j > 1 && j <= (int) NTFTX) {
        bt = (BTRAILER *) TRANBUFF(tx);  /* top of tx proof array */
        /* Check for matching previous hash in the array. */
        if(memcmp(Cblockhash, bt[NTFTX - j].phash, HASHLEN) == 0) {
           result = catchup(&(np->ip), 1);  /* try update */
           if(result == VEBAD) return 0;  /* EVIL: ignore bad bval2() */
        }
   }
   if (result != VEOK) {
      /* Catchup failed so check the tx proof and chain weight. */
      if(checkproof(tx, &splitblock) != VEOK) return 0;  /* ignore bad proof */
      /* Proof is good so try to re-sync to peer */
      if(syncup(splitblock, tx->cblock, np->ip) != VEOK) return 0;
   }  /* ... we updated */
   /* send_found on good catchup or syncup */
   send_found();  /* start send_found() child */
   addrecent(np->ip);
   return 0;  /* nothing else to do */
}  /* end contention() */


/* Search txq1.dat and txclean.dat for src_addr.
 * Return VEOK if the src_addr is not found, otherwise VERROR.
 */
int txcheck(word8 *src_addr)
{
   FILE *fp;
   TXQENTRY tx;

   fp = fopen("txq1.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tx.src_addr, src_addr, TXADDRLEN) == 0) {
            fclose(fp);
            return VERROR;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */

   fp = fopen("txclean.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tx.src_addr, src_addr, TXADDRLEN) == 0) {
            fclose(fp);
            return VERROR;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */
   return VEOK;  /* src_addr not found */
}  /* end txcheck() */


/* opcodes in types.h */
#define valid_op(op)  ((op) >= FIRST_OP && (op) <= LAST_OP)
#define crowded(op)   (Nonline > (MAXNODES - 5) && (op) != OP_FOUND)
#define can_fork_tx() (Nonline <= (MAXNODES - 5))

/**
 * Listen gettx()   (still in parent)
 * Reads a TX structure from SOCKET sd.  Handles 3-way and 
 * validates crc and id's.  Also cares for requests that do not need
 * a child process.
 *
 * Returns:
 *          -1 no data yet
 *          0 connection reset
 *          sizeof(TX) to create child NODE to process read np->tx
 *          1 to close connection ("You're done, tx")
 *          2 ip was pinklisted (She was very naughty.)
 *
 * On entry: sd is non-blocking.
 *
 * Op sequence: OP_HELLO,OP_HELLO_ACK,OP_(?x)
 */
int gettx(NODE *np, SOCKET sd)
{
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   int status;
   word16 opcode;
   TX *tx;

   /* init callserver() */
   memset(np, 0, sizeof(NODE));   /* clear structure */
   tx = &np->tx;
   np->sd = sd;
   np->ip = get_sock_ip(sd);  /* uses getpeername() */
   sprintf(np->id, "%.15s", ntoa(&(np->ip), ipaddr));
   pdebug("gettx(%s): connected...", np->id);

   /* There are many ways to be bad...
    * Check pink lists... */
   if(pinklisted(np->ip)) {
      pdebug("gettx(%s): dropped (pink)", np->id);
      Nbadlogs++;
      return VEBAD;
   }

   /* hello? */
   if (recv_tx(np, INIT_TIMEOUT)) return VERROR;
   if (get16(tx->opcode) != OP_HELLO) goto bad1;

   /* hi! */
   np->id2 = rand16();
   np->id1 = get16(tx->id1);
   if(send_op(np, OP_HELLO_ACK) != VEOK) return VERROR;

   /* how can I help you? */
   status = recv_tx(np, INIT_TIMEOUT);
   opcode = get16(tx->opcode);  /* execute() will check opcode */
   sprintf(np->id, "%.15s 0x%08" P32X, ntoa(&(np->ip), ipaddr),
      (word32) (np->id1 | ((word32) np->id2 << 16)));
   pdebug("gettx(%s): got opcode = %d  status = %d", np->id, opcode, status);
   if(status == VEBAD) goto bad2;
   if(status != VEOK) return VERROR;  /* bad packet -- timeout? */
   if(!valid_op(opcode)) goto bad1;  /* she was a bad girl */

   /* check simple responses */
   if(opcode == OP_GET_IPL) {
      send_ipl(np);
      if(get16(np->tx.len) == 0) {  /* do not add wallets */
         addrecent(np->ip);
      }
      return 1;  /* You're done! */
   }
   else if(opcode == OP_TX) {
      if(txcheck(tx->src_addr) != VEOK) {
         pdebug("got dup src_addr");
         Ndups++;
         return 1;  /* suppress child */
      }
      Nlogins++;  /* raw TX in */
      status = process_tx(np);
      if(status > 2) goto bad1;
      if(status > 1) goto bad2;
      if(get16(np->tx.len) == 0) {  /* do not add wallets */
         addrecent(np->ip);
      }
      return 1;  /* no child */
   } else if(opcode == OP_FOUND) {
      /* getblock child, catchup, re-sync, or ignore */
      if(Blockfound) return 1;  /* Already found one so ignore.  */
      status = contention(np);  /* Do we want this block? */
      if(status != 1) return 1; /* nothing to do: contention() fixed it */

      /* Get block */
      /* Check if bcon is running and if so stop her. */
      if(Bcpid) {
         kill(Bcpid, SIGTERM);
         waitpid(Bcpid, NULL, 0);
         Bcpid = 0;
      }
      if(Sendfound_pid) {
         kill(Sendfound_pid, SIGTERM);
         waitpid(Sendfound_pid, NULL, 0);
         Sendfound_pid = 0;
      }
      Peerip = np->ip;     /* get block child will have this ip */
      /* Now we can fetch the found block, validate it, and update. */
      Blockfound = 1;
      /* fork() child in sever() */
      /* end if OP_FOUND */
   } else if(opcode == OP_BALANCE) {
      send_balance(np);
      return 1;  /* no child */
   } else if(opcode == OP_RESOLVE) {
      tag_resolve(&(np->tx));
      send_op(np, OP_RESOLVE);
      return 1;
   } else if(opcode == OP_GET_CBLOCK) {
      if(!Allowpush || !fexists("miner.tmp")) return 1;
   } else if(opcode == OP_MBLOCK) {
      if(!Allowpush || (time(NULL) - Pushtime) < 150) return 1;
      Pushtime = time(NULL);
   } else if(opcode == OP_HASH) {
      send_hash(np);
      return 1;
   } else if(opcode == OP_IDENTIFY) {
      identify(np);
      return 1;
   }

   if(opcode == OP_BUSY || opcode == OP_NACK || opcode == OP_HELLO_ACK)
      return 1;  /* no child needed */
   /* If too many children in too small a space... */
   if(crowded(opcode)) return 1;  /* suppress child unless OP_FOUND */
   return VEOK;  /* success -- fork() child in server() */

bad1: epinklist(np->ip);
bad2: pinklist(np->ip);
      Nbadlogs++;
      pdebug("   gettx(): pinklist(%s) opcode = %d",
              ntoa(&np->ip, NULL), opcode);
   return VEBAD;
}  /* end gettx() */


/**
 * Is called after initial accept() or connect()
 * Adds the connection to Node[] array.
 * and returns a new NODE * with *np's data.
*/
NODE *getslot(NODE *np)
{
   NODE *newnp;

   /*
    * Find empty slot
    */
   for(newnp = Nodes; newnp < &Nodes[MAXNODES]; newnp++)
      if(newnp->pid == 0) break;

   if(newnp >= &Nodes[MAXNODES]) {
      perr("getslot(): Nodes[] full!");
      Nspace++;
      return NULL;
   }

   Nonline++;    /* number of currently connected sockets */
   pdebug("getslot() added NODE %d", (int) (newnp - Nodes));
   if(newnp >= Hi_node)
      Hi_node = newnp + 1;
   memcpy(newnp, np, sizeof(NODE));
   return newnp;
}  /* end getslot() */
