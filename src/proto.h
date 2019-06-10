/* proto.h  Mochimo function prototypes
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 28 February 2018
 *
*/

void close_extra(void);
int write_data(void *buff, int len, char *fname);

/* Source file: update.c */
int send_found(void);
int update(char *fname, int mode);

/* Source file: gettx.c */
int freeslot(NODE *np);
int sendtx(NODE *np);
int send_op(NODE *np, int opcode);
int gettx(NODE *np, SOCKET sd);
NODE *getslot(NODE *np);

/* Source file: execute.c */
int process_tx(NODE *np);
int sendnack(NODE *np);
int send_file(NODE *np, char *fname);
int send_ipl(NODE *np);
int execute(NODE *np);
int identify(NODE *np);

int rx2(NODE *np, int checkids, int seconds);
int callserver(NODE *np, word32 ip);
int get_tx2(NODE *np, word32 ip, word16 opcode);
int get_block2(word32 ip, byte *bnum, char *fname, word16 opcode);

/* Source file: init.c */
int get_ipl(NODE *np, word32 ip);
int read_coreipl(char *fname);
word32 init_coreipl(NODE *np, char *fname);
void add_weight(byte *weight, int difficulty, byte *bnum);
int cmp_weight(byte *w1, byte *w2);
int append_tfile(char *fname, char *tfile);
byte *tfval(char *fname, byte *highblock, int weight_only, int *result);
int get_eon(NODE *np, word32 peerip);
int init(void);
void trigg_solve(byte *link, int diff, byte *bnum);
char *trigg_generate(byte *in, int diff);
char *trigg_check(byte *in, byte d, byte *bnum);
int get_ipl(NODE *np, word32 ip);

void stop_mirror(void);
int send_balance(NODE *np);

/* Source file: optf.c */
int send_tf(NODE *np);
int send_hash(NODE *np);

/* Source file: proof.c */
int readtf(void *buff, word32 bnum, word32 count);
int loadproof(TX *tx);
int checkproof(TX *tx);

/* Source file: renew.c */
int renew(void);
int refresh_ipl(void);

/* Source file: algo/peach/peach.c */
int peach(BTRAILER *bt, word32 difficulty, char *haiku, word32 *hps, int mode);

