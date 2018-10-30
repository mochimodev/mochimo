/* mirror.c  Transaction mirroring.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 13 May 2018
*/


/* Add src_ip to tx address map (weight[])
 * Called from process_tx()
 * Returns VERROR if no space in map, else VEOK.
 */
int txmap(TX *tx, word32 src_ip)
{
   int j;
   word32 *ipp;

   /* Apply Matt's Algorithm v1.0 to control mirroring... */
   if(get16(tx->len) != 0) {
      /* from wallet */
      memset(tx->weight, 0, 32);  /* clear address map */
   } else {
      /* try to put src_ip in map */
      for(ipp = (word32 *) tx->weight, j = 0; j < 8; ipp++, j++) {
         if(*ipp == 0) {
            *ipp = src_ip;
            break;
         }
      }
      if(j >= 8) return VERROR;  /* no space in map to mirror() */
   }  /* end if not from wallet */
   return VEOK;
}  /* end txmap() */


/* Create a grandchild to send TX's in mirror.dat to ip... */
pid_t mgc(word32 ip)
{
   pid_t pid;
   FILE *fp;
   long offset;
   int lockfd, count;
   TX mtx;
   NODE node;

   /* create grandchild */
   pid = fork();
   if(pid < 0) {
      error("mgc(): Cannot fork()");
      return 0;  /* to parent */
   }
   if(pid) return pid;  /* to parent */

   /* in (grand) child */
   if(Trace) plog("mgc()...");
   show("mgc()");

   fp = fopen("mirror.dat", "rb");
   if(fp == NULL) {
      error("mgc(): Cannot open mirror.dat");
      exit(1);
   }
   offset = 0;

   while(Running) {
      lockfd = lock("mq.lck", 20);
      if(lockfd == -1) {
         error("mgc(): Cannot lock mq.lck"); fclose(fp); exit(1);
      }
      if(fseek(fp, offset, SEEK_SET)) {
         unlock(lockfd); fclose(fp); exit(1);
      }
      /* read the TX from mirror.dat */
      count = fread(&mtx, 1, sizeof(TX), fp);
      /* preserve seek pos because other mgc()'s may be running */
      offset = ftell(fp);
      unlock(lockfd);
      if(count != sizeof(TX)) break;
      /* if not in -v modes... */
      if(Port == Dstport) {
         /* Skip this TX if ip address is already in map. */
         if(search32(ip, (word32 *) mtx.weight, 8)) continue;
      }
      if(callserver(&node, ip) != VEOK) break;
      put16(node.tx.len, 0);  /* signal not wallet to peer */
      memcpy(TRANBUFF(&node.tx), TRANBUFF(&mtx), TRANLEN);
      /* copy ip address map to outgoing TX */
      memcpy(node.tx.weight, mtx.weight, 32);
      send_op(&node, OP_TX);
      closesocket(node.sd);
   }  /* end while Running */
   fclose(fp);
   exit(0);
}  /* end mgc() */


#if CPLISTLEN > RPLISTLEN
error fix CPLISTLEN: It must be <= RPLISTLEN
#endif

/* Send tx to all current or recent peers on iplist.
 * Called from server()       --  becomes child
 */
pid_t mirror1(word32 *iplist, int len)
{
   pid_t pid, peer[RPLISTLEN];
   int j;
   byte busy;

   /* create child */
   pid = fork();
   if(pid < 0) {
      error("mirror(): Cannot fork()");
      return 0;
   }
   if(pid) return pid;  /* to parent */

   /* in child */
   if(Trace) plog("mirror()...");
   show("mirror");

   shuffle32(iplist, len);  /* NOTE: can create embedded zeros. */
   /* Create up to len mgc() grandchildren */
   for(j = 0; j < len; j++) {
      if(iplist[j] == 0) { peer[j] = 0; continue; }
      peer[j] = mgc(iplist[j]);  /* grandchild */
   }

   /* while Running, wait for grandchildren to finish. */
   while(Running) {
      busy = FALSE;
      for(j = 0; j < len; j++) {
         if(peer[j] == 0) continue;
         pid = waitpid(peer[j], NULL, WNOHANG);
         if(pid <= 0) busy = 1; else peer[j] = 0;
      }
      if(!busy) exit(0);
   }  /* end while Running */
   /* got SIGTERM */
   for(j = 0; j < len; j++) {
      if(peer[j]) {
         kill(peer[j], SIGTERM);     /* Kill grandchild */
         waitpid(peer[j], NULL, 0);  /* and burry her. */
      }
   }
   exit(0);
}  /* end mirror1() */


byte Frisky;  /* command line switch */

/* Send tx to either current or recent peers
 * Called from server()       --  becomes child
 */
pid_t mirror(void)
{
   if(Frisky)
      return mirror1(Rplist, RPLISTLEN);
   else
      return mirror1(Cplist, CPLISTLEN);
}


/* kill mirror() children and grandchildren */
void stop_mirror(void)
{
   if(Mqpid) {
      if(Trace) plog("   Reaping mirror() zombies...");
      kill(Mqpid, SIGTERM);
      waitpid(Mqpid, NULL, 0);
      Mqpid = 0;
   }
}  /* end stop_mirror() */


/* Called by gettx()  -- in parent
 *
 * Validate a TX, write clean TX to txq1.dat, and raw TX to
 * mirror queue, mq.dat.
 * Locks mq.lck while appending mq.dat.
 */
int process_tx(NODE *np)
{
   TX *tx;
   int evilness;
   int count, lockfd;
   int ecode;
   byte tx_id[HASHLEN];
   FILE *fp;

   if(Trace) plog("process_tx()");
   show("tx");

   tx = &np->tx;

   /* Validate addresses, fee, signature, source balance, and total. */
   evilness = tx_val(tx);
   if(evilness) return evilness;

   /* Compute tx_id[] (hash of tx->src_addr) to append to txq1.dat. */
   sha256(tx->src_addr, TXADDRLEN, tx_id);

   fp = fopen("txq1.dat", "ab");
   if(!fp) {
      error("process_tx(): Cannot open txq1.dat");
      return 1;
   }

   /* Now write transaction to txq1.dat followed by tx_id */
   ecode = 0;
   /* 3 addresses (TXADDRLEN*3) + 3 amounts (8*3) + signature (TXSIGLEN) */
   count = fwrite(TRANBUFF(tx), 1, TRANLEN, fp);
   if(count != TRANLEN) ecode = 1;
   /* then append source tx_id */
   count = fwrite(tx_id, 1, HASHLEN, fp);
   if(count != HASHLEN) ecode = 1;
   if(Trace) plog("writing TX to txq1.dat");
   fclose(fp);  /* close txq1.dat */
   if(ecode) {
      error("bad write on txq1.dat");
      return 1;
   }
   else {
      Txcount++;
      if(Trace) plog("incrementing Txcount to %d", Txcount);
   }
   Nrec++;  /* total good TX received */

   /* lock mirror file */
   lockfd = lock("mq.lck", 20);
   if(lockfd == -1) {
      error("process_tx(): Cannot lock mq.lck");  /* should not happen */
      return 1;
   }
   fp = fopen("mq.dat", "ab");
   if(!fp) {
      error("process_tx(): Cannot open mq.dat");
      unlock(lockfd);
      return 1;
   }
   ecode = 0;
   /* If empty slot in mirror address map, fill it
    * in and then write tx to mirror queue, mq.dat.
    */
   if(txmap(tx, np->src_ip) == VEOK) {
      count = fwrite(tx, 1, sizeof(TX), fp);
      if(count != sizeof(TX)) {
         error("bad write on mq.dat");
         ecode = 1;
      } else Mqcount++;
   }
   fclose(fp);      /* close mirror queue, mq.dat */
   unlock(lockfd);  /* unlock mirror queue lock, mq.lck */
   return ecode;
}  /* end process_tx() */
