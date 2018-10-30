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
   if(le_find(np->tx.src_addr, &le, NULL) == TRUE)
      put64(np->tx.send_total, le.balance);
   send_op(np, OP_SEND_BAL);
   return 0;  /* success */
}  /* end send_balance() */


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

      default:
         Nbadlogs++;  /* bad OP's */
         if(Trace) plog("execute(): bad opcode: %d", np->opcode);
         return 2;
    }  /* end switch op */
}  /* end execute() */
