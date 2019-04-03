/* execute.c  Process NODE transaction, opcodes, and states.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
*/

/* Send a ledger.dat balance query to np.
 * Called from gettx() OP_BALANCE
 * layout:
 * on entry:
 *     np->tx.src_addr    address to query
 * on return:
 *     np->tx.send_total = balance of src_addr (0 if not found)
 *
 * Returns 1 on I/O errors, else 0.
*/
int send_balance(NODE *np)
{
   LENTRY le;
   static byte zeros[8];

   put64(np->tx.send_total, zeros);
   /* look up source address in ledger */
   if(((byte *) (np->tx.src_addr))[2196] == 0x00) {
      /* OP_BALANCE Request Passed ZEROED Tag */
      /* Finding an address in ledger without matching the tag */
      if(le_find(np->tx.src_addr, &le, NULL, 1) == TRUE) {
         put64(np->tx.send_total, le.balance);
         memcpy(np->tx.src_addr, le.addr, TXADDRLEN);
      }
   } else {
      if(le_find(np->tx.src_addr, &le, NULL, 0) == TRUE) {
         put64(np->tx.send_total, le.balance);
     }
   }
   send_op(np, OP_SEND_BAL);
   return 0;  /* success */
} /* end send_balance() */

int sendnack(NODE *np)
{
   put16(np->tx.opcode, OP_NACK);
   return sendtx(np);
}


/*
 * send_file() timeout handler  -- child exits
 */
void sendalrm(int sig)
{
   exit(1);  /* fail */
}

/* Send block to peer  -- called by child
 * Return VERROR on file errors or reset connection, else VEOK.
 */
int send_file(NODE *np, char *fname)
{
   byte *bnum;
   TX *tx;
   int n, status;
   FILE *fp;
   char name[128];

   tx = &np->tx;
   bnum = tx->blocknum;

   show("send");

   if(fname == NULL) {
      sprintf(name, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      fname = name;
   }
   fp = fopen(fname, "rb");
   if(fp == NULL) {
      if(Trace) plog("cannot open %s", fname);
      sendnack(np);
      return VERROR;
   }
   if(Trace) plog("sending %s", fname);
   blocking(np->sd);   /* set blocking I/O for send() */
   signal(SIGALRM, sendalrm);  /* set timeout handler */
   for(; Running; ) {
      n = fread(TRANBUFF(tx), 1, TRANLEN, fp);
      put16(tx->len, n);
      alarm(10);
      status = send_op(np, OP_SEND_BL);
      if(n < TRANLEN) {
         alarm(0);
         fclose(fp);
         return status;  /* VEOK or VERROR -- server does freeslot() */
      }
      if(status != VEOK) break;
      /* Make upload bandwidth dynamic. */
      if(Nonline > 1) usleep((Nonline - 1) * UBANDWIDTH);
   }  /* end for(; Running; ) */
   alarm(0);
   fclose(fp);
   return VERROR;
}  /* end send_file() */


/* Send our recent peer list to NODE np in response to OP_GETIPL.
 * Called from execute().
 */
int send_ipl(NODE *np)
{
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy recent peer list to TX */
   memcpy(TRANBUFF(&np->tx), Rplist, IPCOPYLEN);
   put16(np->tx.len, IPCOPYLEN);
   return send_op(np, OP_SEND_IP);  /* send ip list */
}


int identify(NODE *np)
{
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy recent peer list to TX */
   sprintf((char *) TRANBUFF(&np->tx), "Sanctuary=%u,Lastday=%u,Mfee=%u",
           Sanctuary, Lastday, Myfee[0]);
   return send_op(np, OP_IDENTIFY);
}


/* Called from execute() in execute.c
 * Returns 0.
 */
int send_cblock(NODE *np)
{
   char cmd[100], fname[32];
   pid_t pid;

   show("sendcb");
   if(exists("miner.tmp")) {
      pid = getpid();
      sprintf(fname, "cb%u.tmp", (int) pid);
      sprintf(cmd, "cp miner.tmp %s", fname);
      system(cmd);
      send_file(np, fname);
      unlink(fname);
   }
   return 0;
}  /* end send_cblock() */


/* Get a block named fname from np
 * Returns 0 on good download, else 1
 */
int get_block3(NODE *np, char *fname)
{
   FILE *fp;
   word16 len;
   int n;
   int ecode = 666;

   if(Trace) plog("get_block3() Recfile is '%s'", fname);
   show("getmb");

   fp = fopen(fname, "wb");
   if(fp == NULL) {
      plog("cannot open %s", fname);
      return 1;
   }

   for(;;) {
      if((ecode = rx2(np, 1, 10)) != VEOK) goto bad;
      if(get16(np->tx.opcode) != OP_SEND_BL) goto bad; 
      len = get16(np->tx.len);
      if(len > TRANLEN) goto bad;
      if(len) {
         n = fwrite(TRANBUFF(&np->tx), 1, len, fp);
         if(n != len) {
            error("get_block3() I/O error");
            goto bad;
         }
      }
      /* check EOF */
      if(len < 1 || n < TRANLEN) {
         fclose(fp);
         if(Trace) plog("get_block3(): EOF");
         return 0;
      } /* end if EOF */
   }  /* end for */
bad:
   fclose(fp);
   unlink(fname);  /* delete partial downloads */
   if(Trace)
      plog("get_block3(): fail (%d) len = %d opcode = %d",
           ecode, get16(np->tx.len), get16(np->tx.opcode));
   return 1;
}  /* end get_block3() */


/* Get a mined block from some random node... */
int get_mblock(NODE *np)
{
   int status;

   /* Get a block or file that is pushed on us.  In execute(): */
   if(Blockfound || exists("rblock2.dat") || exists("cblock.lck"))
      return 1;
   status = get_block3(np, "rblock2.dat");
   if(status || exists("mblock.dat")) {
      unlink("rblock2.dat");
      return 1;
   }
   if(rename("rblock2.dat", "mblock.dat")) return 1;
   system("touch cblock.lck");
   return 0;
}  /* end get_mblock() */


/**
 * Called from server()  --  NOTE: we are in child here
 * Returns 0 = Aokay! Veronica says child is done.
 *         1 = there were issues
 *         2 = pinklist
 *         3 = and epinklist too!
 */
int execute(NODE *np)
{
   int status;

   if(Trace)
      plog("execute(): opcode = %d", np->opcode);

   status = 0;  /* for child exit status */
   switch(np->opcode) {
      case OP_FOUND:
         /* get the advertised found block -- synchronous
          * Blockfound was set by gettx()
          */
         closesocket(np->sd);  /* close initial connection */
         if(get_block2(np->src_ip, np->tx.cblock, "rblock.dat",
                       OP_GETBLOCK) != VEOK) return 1;  /* fail */
         return 0;
      case OP_GETBLOCK:
         /* send np->tx.blocknum to peer */
         if(send_file(np, NULL) != VEOK) status = 1;
         closesocket(np->sd);
         return status;
      case OP_GET_TFILE:
         /* send out tfile.dat to peer */
         if(send_file(np, "tfile.dat") != VEOK) status = 1;
         closesocket(np->sd);
         return status;
      case OP_GET_CBLOCK:
         signal(SIGTERM, sendalrm);
         send_cblock(np);
         closesocket(np->sd);
         return 0;
      case OP_MBLOCK:
         signal(SIGTERM, sendalrm);
         get_mblock(np);
         closesocket(np->sd);
         return 0;
      case OP_TF:
         /* send tfile.dat section to peer */
         send_tf(np);
         closesocket(np->sd);
         return 0;

      default:
         Nbadlogs++;  /* bad OP's */
         if(Trace) plog("execute(): bad opcode: %d", np->opcode);
         return 2;
    }  /* end switch op */
}  /* end execute() */
