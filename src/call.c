/* call.c  callserver() and friends
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 18 February 2018
*/


/* Receive next packet from NODE *np
 * SOCKET np->sd is already set non-blocking.
 * Returns: VEOK (0) = good, else error code.
 * Check id's if checkids is non-zero.
 * NOTE: Set checkid to zero during handshake.
 */
int rx2(NODE *np, int checkids, int seconds)
{
   int count, n;
   time_t timeout;
   TX *tx;

   tx = &np->tx;
   timeout = time(NULL) + seconds;

   if(Trace)
      plog("Entering rx() sd = %d  id1 = %x  id2 = %x",
           np->sd, np->id1, np->id2); /* debug */

   for(n = 0; ; ) {
      count = recv(np->sd, TXBUFF(tx) + n, TXBUFFLEN - n, 0);
      if(count == 0) return VERROR;
      if(count < 0) {
         if(time(NULL) >= timeout) return VETIMEOUT;
         usleep(10);
         continue;
      }
      n += count;
      if(n == TXBUFFLEN) break;
   }  /* end for */

   /* check tx and return error codes or count */
   if(get16(tx->network) != TXNETWORK)
      return VEBAD;
   if(get16(tx->trailer) != TXEOT)
      return VEBAD;
   if(crc16(CRC_BUFF(tx), CRC_COUNT) != get16(tx->crc16))
      return VEBAD;
   if(checkids && (np->id1 != get16(tx->id1) || np->id2 != get16(tx->id2)))
      return VEBAD;
   return VEOK;  /* 0 success */
}  /* end rx2() */


/* Call peer and complete Three-Way */
int callserver(NODE *np, word32 ip)
{
   int ecode;

   if(Trace) plog("callserver(): Trying %s...", ntoa((byte *) &ip));

   memset(np, 0, sizeof(NODE));   /* clear structure */
   np->sd = connectip(ip);  /* returns non-blocked sd */
   if(np->sd == INVALID_SOCKET) return VERROR;
   np->src_ip = ip;
   np->id1 = rand16();
   if(send_op(np, OP_HELLO) != VEOK) goto bad;

   ecode = rx2(np, 0, ACK_TIMEOUT);
   if(ecode != VEOK) {
      if(Trace) plog("   *** missing HELLO_ACK packet (%d)", ecode);
bad:
      closesocket(np->sd);
      np->sd = INVALID_SOCKET;
      return VERROR;
   }
   np->id2 = get16(np->tx.id2);
   np->opcode = get16(np->tx.opcode);
   if(np->opcode != OP_HELLO_ACK || get16(np->tx.id1) != np->id1) {
      if(Trace) plog("   *** HELLO_ACK is wrong: %d", np->opcode);
      pinklist(ip);   /* protocol violator! */
      epinklist(ip);
      goto bad;
   }
   return VEOK;
}  /* end callserver() */


/* Used for opcode = OP_GETHAL or OP_GETIPL
 * Close socket and sets np->sd to INVALID_SOCKET on return.
 */
int get_tx2(NODE *np, word32 ip, word16 opcode)
{
   if(callserver(np, ip) != VEOK)
      return VERROR;

   send_op(np, opcode);
   if(rx2(np, 1, 10) == VEOK) {
      closesocket(np->sd);
      np->sd = INVALID_SOCKET;
      return VEOK;
   }
   closesocket(np->sd);
   np->sd = INVALID_SOCKET;
   return VERROR;
}  /* end get_tx2() */


/* Get a block or other file from peer, ip.
 * opcode is OP_GETBLOCK or OP_GET_TFILE.
 * bnum can be NULL for OP_GET_FILE.
 * Returns VEOK (0) on good download, else VERROR (1).
 */
int get_block2(word32 ip, byte *bnum, char *fname, word16 opcode)
{
   NODE node;
   FILE *fp;
   word16 len;
   int n;
   int ecode = 666;

   if(Trace) plog("Entering get_block2() Recfile is '%s'", fname);
   show("getblock");

   fp = fopen(fname, "wb");
   if(fp == NULL)
      return error("cannot open %s", fname);

   if(callserver(&node, ip) != VEOK)
      goto bad;
   
   /* set request block number */
   if(bnum) put64(node.tx.blocknum, bnum);
   if(send_op(&node, opcode) != VEOK) goto bad;
   for(;;) {
      if((ecode = rx2(&node, 1, 10)) != VEOK) goto bad;
      if(get16(node.tx.opcode) != OP_SEND_BL) goto bad; 
      len = get16(node.tx.len);
      if(len > TRANLEN) goto bad;
      if(len) {
         n = fwrite(TRANBUFF(&node.tx), 1, len, fp);
         if(n != len) {
            error("get_block2() I/O error");
            goto bad;
         }
      }
      /* check EOF */
      if(len < 1 || n < TRANLEN) {
         fclose(fp);
         closesocket(node.sd);
         node.sd = INVALID_SOCKET;
         if(Trace) plog("get_block2(): EOF");
         return VEOK;
      } /* end if EOF */
   }  /* end for */
bad:
   fclose(fp);
   unlink(fname);  /* delete partial downloads */
   if(node.sd != INVALID_SOCKET)
      closesocket(node.sd);
   node.sd = INVALID_SOCKET;
   if(Trace)
      plog("get_block2(): fail (%d) len = %d opcode = %d",
           ecode, get16(node.tx.len), get16(node.tx.opcode));
   return VERROR;
}  /* end get_block2() */
