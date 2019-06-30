/* optf.c  Handle OP_TF to send section of tfile.dat
 *         and OP_HASH to send a requested block hash.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.TXT   **** NO WARRANTY ****
 *
 * Date: 16 February 2019
*/


/* Process OP_TF.  Return VEOK on success, else VERROR.
 * Called by child -- execute().
 */
int send_tf(NODE *np)
{
   int status;
   word32 first, count;
   char cmd[128], fname[32];

   sprintf(fname, "tf%u.tmp", (int) getpid());

   first = get32(np->tx.blocknum);      /* first trailer to send */
   count = get32(&np->tx.blocknum[4]);  /* count of trailers to send */

   /* limit tfile extract to 1000 trailers */
   if(count > 1000) return VERROR;
   sprintf(cmd, "dd if=tfile.dat of=%s bs=%u skip=%u count=%u 2>/dev/null",
                fname, (int) sizeof(BTRAILER), first, count);
   system(cmd);
   status = send_file(np, fname);  /* returns VEOK or VERROR */
   unlink(fname);
   return status;
}  /* end send_tf() */


/* Process OP_HASH.  Return VEOK on success, else VERROR.
 * Called by gettx().
 */
int send_hash(NODE *np)
{
   BTRAILER bt;
   char fname[128];

   sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(np->tx.blocknum));
   if(readtrailer(&bt, fname) != VEOK) return VERROR;
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy hash of tx.blocknum to TX */
   memcpy(TRANBUFF(&np->tx), bt.bhash, HASHLEN);
   put16(np->tx.len, HASHLEN);
   return send_op(np, OP_HASH);  /* send back to peer */
}  /* end send_hash() */
