/* execute.c  Process NODE transaction, opcodes, and states.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
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
   word16 len;
   static word8 zeros[8];

   len = get16(np->tx.len);
   put64(np->tx.send_total, zeros);
   put64(np->tx.change_total, zeros);
   /* check for old OP_BALANCE Request with ZEROED Tag */
   if(len == 0 && ((word8 *) (np->tx.src_addr))[2196] == 0x00) {
     len = TXADDRLEN - 12;
   }
   /* look up source address in ledger */
   if(le_find(np->tx.src_addr, &le, NULL, len) == TRUE) {
     put64(np->tx.send_total, le.balance);
     put64(np->tx.change_total, One); /* indicate address was found */
     memcpy(np->tx.src_addr, le.addr, TXADDRLEN); /* return found address */
   }
   send_op(np, OP_SEND_BAL);
   return 0;  /* success */
} /* end send_balance() */


/* Send our recent peer list to NODE np in response to OP_GETIPL.
 * Called from execute().
 */
int send_ipl(NODE *np)
{
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy recent peer list to TX */
   memcpy(TRANBUFF(&np->tx), Rplist, sizeof(word32) * RPLISTLEN);
   put16(np->tx.len, sizeof(word32) * RPLISTLEN);
   return send_op(np, OP_SEND_IPL);  /* send ip list */
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
   if(fexists("miner.tmp")) {
      pid = getpid();
      sprintf(fname, "cb%u.tmp", (int) pid);
      sprintf(cmd, "cp miner.tmp %s", fname);
      system(cmd);
      send_file(np, fname);
      unlink(fname);
   }
   return 0;
}  /* end send_cblock() */


/* Get a mined block from some random node... */
int get_mblock(NODE *np)
{
   int status;

   /* Get a block or file that is pushed on us.  In execute(): */
   if(Blockfound || fexists("rblock2.dat") || fexists("cblock.lck"))
      return 1;

   /* receive file */
   status = recv_file(np, "rblock2.dat");

   if(status || fexists("mblock.dat")) {
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
   word16 opcode;

   opcode = get16(np->tx.opcode);
   pdebug("execute(%s): opcode = %d", np->id, opcode);

   status = 0;  /* for child exit status */
   switch(opcode) {
      case OP_FOUND:
         /* get the advertised found block -- synchronous
          * Blockfound was set by gettx() */
         sock_close(np->sd);  /* close initial connection */
         return get_file(np->ip, np->tx.cblock, "rblock.dat");
      case OP_GET_BLOCK:
         /* send np->tx.blocknum to peer */
         if(send_file(np, NULL) != VEOK) status = 1;
         sock_close(np->sd);
         return status;
      case OP_GET_TFILE:
         /* send out tfile.dat to peer */
         if(send_file(np, "tfile.dat") != VEOK) status = 1;
         sock_close(np->sd);
         return status;
      case OP_GET_CBLOCK:
         send_cblock(np);
         sock_close(np->sd);
         return 0;
      case OP_MBLOCK:
         get_mblock(np);
         sock_close(np->sd);
         return 0;
      case OP_TF:
         /* send tfile.dat section to peer */
         send_tf(np);
         sock_close(np->sd);
         return 0;

      default:
         Nbadlogs++;  /* bad OP's */
         pdebug("execute(): bad opcode: %d", opcode);
         return 2;
    }  /* end switch op */
}  /* end execute() */
