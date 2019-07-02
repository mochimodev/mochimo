/* gettx.c  Get validated transaction packet (TX) and helpers.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
*/

/* forward reference */
int rx2(NODE *np, int checkids, int seconds);

/* Mark NODE np in Nodes[] empty by setting np->pid to zero.
 * Adjust Nonline and Hi_node.
 * Caller must close np->sd if needed.
 */
int freeslot(NODE *np)
{
   if(np->pid == 0)
      return error("*** NODE %ld already called freeslot() ***",
                   (long) (np - Nodes));
   if(Trace)
      plog("freeslot(): idx=%d  ip = %-1.20s pid = %d", (long) (np - Nodes),
           ntoa((byte *) &np->src_ip), np->pid);
   Nonline--;
   np->pid = 0;
   /* Update pointer to just beyond highest used slot in Nodes[] */
   while(Hi_node > Nodes && (Hi_node - 1)->pid == 0)
      Hi_node--;
   if(Nonline < 0) { Nonline = 0; return error("Nonline < 0"); }
   return VEOK;
}  /* end freeslot() */


/* Send packet: set advertised fields and crc16.
 * Returns VEOK on success, else VERROR.
 */
int sendtx(NODE *np)
{
   int count, len;
   time_t timeout;
   byte *buff;

   np->tx.version[0] = PVERSION;
   np->tx.version[1] = Cbits;
   put16(np->tx.network, TXNETWORK);
   put16(np->tx.trailer, TXEOT);

   put16(np->tx.id1, np->id1);
   put16(np->tx.id2, np->id2);
   put64(np->tx.cblock, Cblocknum);  /* 64-bit little-endian */
   memcpy(np->tx.cblockhash, Cblockhash, HASHLEN);
   memcpy(np->tx.pblockhash, Prevhash, HASHLEN);
   if(get16(np->tx.opcode) != OP_TX)  /* do not copy over TX ip map */
      memcpy(np->tx.weight, Weight, HASHLEN);
   crctx(&np->tx);
   count = send(np->sd, TXBUFF(&np->tx), TXBUFFLEN, 0);
   if(count == TXBUFFLEN) return VEOK;
   /* --- v20 retry */
   if(Trace) plog("sendtx(): send() retry...");
   timeout = time(NULL) + 10;
   for(len = TXBUFFLEN, buff = TXBUFF(&np->tx); ; ) {
      if(count == 0) break;
      if(count > 0) { buff += count; len -= count; }
      else {
         if(errno != EWOULDBLOCK || time(NULL) >= timeout) break;
      }
      count = send(np->sd, buff, len, 0);
      if(count == len) return VEOK;
   }
   /* --- v20 end */
   Nsenderr++;
   if(Trace)
      plog("send() error: count = %d  errno = %d", count, errno);
   return VERROR;
}  /* end sendtx() */


int send_op(NODE *np, int opcode)
{
   put16(np->tx.opcode, opcode);
   return sendtx(np);
}


/* A Basic block validator for catchup().
 * Every non-NG block should pass this test.
 * If it does not, the error is intentional (pink-list).
 * Returns: VEOK if valid, VERROR on errors, or VEBAD if bad.
 */
int bval2(char *fname, byte *bnum, byte diff)
{
   BTRAILER bt;
   word32 now;
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

   if(Trace) plog("bval2()");

   if(readtrailer(&bt, fname) != VEOK) return VERROR;
   if(cmp64(bnum, bt.bnum) != 0) return VEBAD;
   if(get32(bt.difficulty) != diff) return VERROR;
   /* Time Checks */
   if(get32(bt.stime) <= Time0) return VERROR; /* bad time sequence */
   now = time(NULL);
   if(get32(bt.stime) > (now + BCONFREQ)) return VERROR;  /* future */

   /* Solution Check */

   if(cmp64(bnum, v24trigger) > 0) { /* v2.4 Algo */
      if(peach(&bt, get32(bt.difficulty), NULL, 1)){
         return VEBAD; /* block didn't validate */
      }
   }
   if(cmp64(bnum, v24trigger) <= 0) { /* v2.3 and prior */
      if(trigg_check(bt.mroot, bt.difficulty[0], bt.bnum) == NULL) {
         return VEBAD;
      }
   }

   return VEOK;
}  /* end bval2() */


/* Catch up by getting blocks: all else waits...
 * Return VERROR if contention, else VEOK.
 */
int catchup(word32 peerip)
{
   byte bnum[8];
   int count, status;

   if(Trace) plog("catchup(%s)", ntoa((byte *) &peerip));

   put64(bnum, Cblocknum);
   for(count = 0; Running; ) {
      add64(bnum, One, bnum);
      if(bnum[0] == 0) continue;  /* do not fetch NG blocks */
      if(get_block2(peerip, bnum, "rblock.dat", OP_GETBLOCK) != VEOK) break;
      status = bval2("rblock.dat", bnum, Difficulty);
      if(status != VEOK) {
         if(status == VEBAD) epinklist(peerip);
         break;
      }
      if(update("rblock.dat", 0) != VEOK) {
         if(count == 0) return VERROR;  /* contention */
         break;  /* ignore */
      }
      count++;
   }  /* end for count */
   return VEOK;
}  /* end catchup() */


/* Handle contention
 * Returns:  0 = ignore
 *           1 = do fetch block with child
 *           On Contention, calls restart().
 */
int contention(NODE *np)
{
   byte diff[8];
   int result;
   TX *tx;

   if(Trace) plog("contention(?)");

   tx = &np->tx;
   /* ignore low block num */
   if(cmp64(tx->cblock, Cblocknum) <= 0) return 0;
   /* ignore low weight */
   if(cmp_weight(tx->weight, Weight) <= 0) return 0;

   if(tx->cblock[0] == 0) {
      epinklist(np->src_ip);  /* protocol error */
      return 0;  /* ignore block */
   }

   sub64(tx->cblock, Cblocknum, diff);
   result = cmp64(diff, One);
   if(result ==  0) {  /* check if one block ahead */
      if(memcmp(Cblockhash, tx->pblockhash, HASHLEN) == 0) {
            if(Trace) plog("   found!");
            return 1;  /* get block */
      }
   }  /* end if result == 0 -- one block ahead */

   /* more than one block ahead or bad hash */
   if(catchup(np->src_ip) != VEOK && checkproof(tx) != VEOK) {
      write_data(&np->src_ip, 4, "rplist.lst");
      restart("contend");  /* Contention - RESTART */
   }
   return 0;
}  /* end contention() */


/* Search txq1.dat and txclean.dat for src_addr.
 * Return VEOK if the src_addr is not found, otherwise VERROR.
 */
int txcheck(byte *src_addr)
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
 *          2 src_ip was pinklisted (She was very naughty.)
 *
 * On entry: sd is non-blocking.
 *
 * Op sequence: OP_HELLO,OP_HELLO_ACK,OP_(?x)
 */
int gettx(NODE *np, SOCKET sd)
{
   int count, status, n;
   word16 opcode;
   TX *tx;
   word32 ip;
   time_t timeout;

   tx = &np->tx;
   memset(np, 0, sizeof(NODE));  /* clear structure */
   np->sd = sd;
   np->src_ip = ip = getsocketip(sd);  /* uses getpeername() */

   /*
    * There are many ways to be bad...
    * Check pink lists...
    */
   if(pinklisted(ip)) {
      Nbadlogs++;
      return 2;
   }

   n = recv(sd, TXBUFF(tx), TXBUFFLEN, 0);
   if(n <= 0) return n;
   timeout = time(NULL) + 3;
   for( ; n != TXBUFFLEN; ) {
      count = recv(sd, TXBUFF(tx) + n, TXBUFFLEN - n, 0);
      if(count == 0) return 1;
      if(count < 0) {
         if(time(NULL) >= timeout) { Ntimeouts++; return 1; }
         continue;
      }
      n += count;  /* collect the full TX */
   }
   count = n;

   /*
    * validate packet and return 1 if bad.
    */
   opcode = get16(tx->opcode);
   if(get16(tx->network) != TXNETWORK
      || get16(tx->trailer) != TXEOT
      || crc16(CRC_BUFF(tx), CRC_COUNT) != get16(tx->crc16) ) {
            if(Trace) plog("gettx(): bad packet");
            return 1;  /* BAD packet */
   }
   if(tx->version[0] != PVERSION) {
      if(Trace) plog("gettx(): bad version");
      return 1;
   }
   
   if(Trace) plog("gettx(): crc16 good");
   if(opcode != OP_HELLO) goto bad1;
   np->id1 = get16(tx->id1);
   np->id2 = rand16();
   if(send_op(np, OP_HELLO_ACK) != VEOK) return VERROR;
   status = rx2(np, 1, 3);
   opcode = get16(tx->opcode);
   if(Trace)
      plog("gettx(): got opcode = %d  status = %d", opcode, status);
   if(status == VEBAD) goto bad2;
   if(status != VEOK) return VERROR;  /* bad packet -- timeout? */
   np->opcode = opcode;  /* execute() will check the opcode */
   if(!valid_op(opcode)) goto bad1;  /* she was a bad girl */

   if(opcode == OP_GETIPL) {
      send_ipl(np);
      if(get16(np->tx.len) == 0) {  /* do not add wallets */
         addcurrent(np->src_ip);  /* v.28 */
         addrecent(np->src_ip);
      }
      return 1;  /* You're done! */
   }
   else if(opcode == OP_TX) {
      if(txcheck(tx->src_addr) != VEOK) {
         if(Trace) plog("got dup src_addr");
         Ndups++;
         return 1;  /* suppress child */
      }
      Nlogins++;  /* raw TX in */
      status = process_tx(np);
      if(status > 2) goto bad1;
      if(status > 1) goto bad2;
      if(get16(np->tx.len) == 0) {  /* do not add wallets */
         addcurrent(np->src_ip);    /* add to peer lists */
         addrecent(np->src_ip);
      }
      return 1;  /* no child */
   } else if(opcode == OP_FOUND) {
      if(Blockfound) return 1;  /* Already found one so ignore.  */
      status = contention(np);  /* Do we want this block? */
      if(status != 1) return 1; /* ignore: low block or weight */

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
      Peerip = np->src_ip;     /* get block child will have this ip */
      /* Now we can fetch the found block, validate it, and update. */
      Blockfound = 1;
      /* fork() child in sever() */
      /* end if OP_FOUND */
   } else if(opcode == OP_BALANCE) {
      send_balance(np);
      Nsent++;
      return 1;  /* no child */
   } else if(opcode == OP_RESOLVE) {
      tag_resolve(np);
      return 1;
   } else if(opcode == OP_GET_CBLOCK) {
      if(!Allowpush || !exists("miner.tmp")) return 1;
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
   return count;  /* success -- fork() child in server() */

bad1: epinklist(np->src_ip);
bad2: pinklist(np->src_ip);
      Nbadlogs++;
      if(Trace)
         plog("   gettx(): pinklist(%s) opcode = %d",
              ntoa((byte *) &np->src_ip), opcode);
   return 2;
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
      error("getslot(): Nodes[] full!");
      Nspace++;
      return NULL;
   }

   Nonline++;    /* number of currently connected sockets */
   if(Trace)
      plog("getslot() added NODE %d", (int) (newnp - Nodes));
   if(newnp >= Hi_node)
      Hi_node = newnp + 1;
   memcpy(newnp, np, sizeof(NODE));
   return newnp;
}  /* end getslot() */
