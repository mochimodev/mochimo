/* types.h   Structure definitions: NODE, block, ledger, transactions, etc.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 1 January 2018
 *
*/


#define OP_NULL           0
#define OP_HELLO          1
#define OP_HELLO_ACK      2
#define FIRST_OP          3  /* first OP after ack's */
#define OP_TX             3
#define OP_FOUND          4
#define OP_GETBLOCK       5
#define OP_GETIPL         6
#define OP_SEND_BL        7
#define OP_SEND_IP        8
#define OP_BUSY           9
#define OP_NACK           10
#define OP_GET_TFILE      11
#define OP_BALANCE        12
#define OP_SEND_BAL       13
#define OP_RESOLVE        14
#define OP_GET_CBLOCK     15
#define OP_MBLOCK         16
#define OP_HASH           17
#define OP_TF             18
#define OP_IDENTIFY       19
#define LAST_OP           19  /* edit when adding  OP's */

#define TXNETWORK 0x0539
#define TXEOT     0xabcd

#define TXADDRLEN 2208
#define TXAMOUNT  8
#define TXSIGLEN  2144  /* WOTS */

/* TX buff offset: */
#define TRANBUFF(tx) ((tx)->src_addr)
/*                      addresses        amounts    signature  crc + trailer */
#define TRANLEN      ( (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN )
#define SIG_HASH_COUNT (TRANLEN - TXSIGLEN)
#define TXBUFF(tx)   ((byte *) tx)
/* for struct size checking: */
#define TXBUFFLEN  ((2*5) + (8*2) + 32 + 32 + 32 + 2 \
                      + (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN + (2+2) )

#define CRC_BUFF(tx) TXBUFF(tx)
#define CRC_COUNT   (TXBUFFLEN - (2+2))  /* tx buff less crc and trailer */
#define CRC_VAL_PTR(tx)  ((tx)->crc16)

#if (RPLISTLEN*4) <= TRANLEN
#define IPCOPYLEN (RPLISTLEN*4)
#else
  Change RPLISTLEN value to make the above #if true
#endif

/* Capability bits */
#define C_PUSH      1
#define C_WALLET    2
#define C_SANCTUARY 4
#define C_MFEE      8
#define C_LOGGING   16

/* Multi-byte numbers are little-endian.
 * Structure is checked on start-up for byte-alignment.
 * HASHLEN is checked to be 32.
 */
typedef struct {
   byte version[2];  /* { PVERSION, Cbits }  */
   byte network[2];  /* 0x39, 0x05 TXNETWORK */
   byte id1[2];
   byte id2[2];
   byte opcode[2];
   byte cblock[8];        /* current block num  64-bit */
   byte blocknum[8];      /* block num for I/O in progress */
   byte cblockhash[32];   /* sha-256 hash of our current block */
   byte pblockhash[32];   /* sha-256 hash of our previous block */
   byte weight[32];       /* sum of block difficulties (or TX ip map) */
   byte len[2];  /* length of data in transaction buffer for I/O op's */
   /* start transaction buffer */
   byte src_addr[TXADDRLEN];
   byte dst_addr[TXADDRLEN];
   byte chg_addr[TXADDRLEN];
   byte send_total[TXAMOUNT];
   byte change_total[TXAMOUNT];
   byte tx_fee[TXAMOUNT];
   byte tx_sig[TXSIGLEN];
   /* end transaction buffer */
   byte crc16[2];
   byte trailer[2];  /* 0xcd, 0xab */
} TX;


typedef struct {
   TX tx;  /* transaction buffer */
   word16 id1;      /* from tx */
   word16 id2;      /* from tx */
   int opcode;      /* from tx */
   word32 src_ip;
   SOCKET sd;
   pid_t pid;     /* process id of child -- zero if empty slot */
} NODE;


/* Structure for clean TX que */
typedef struct {
   /* start transaction buffer (These fields are order dependent) */
   byte src_addr[TXADDRLEN];     /*  2208 */
   byte dst_addr[TXADDRLEN];
   byte chg_addr[TXADDRLEN];
   byte send_total[TXAMOUNT];    /* 8 */
   byte change_total[TXAMOUNT];
   byte tx_fee[TXAMOUNT];
   byte tx_sig[TXSIGLEN];        /* 2144 */
   byte tx_id[HASHLEN];          /* 32 */
} TXQENTRY;


/* The block header */
typedef struct {
   byte hdrlen[4];         /* header length to tran array */
   byte maddr[TXADDRLEN];  /* mining address */
   byte mreward[8];
   /*
    * variable length data here...
    */

   /* array of tcount TXQENTRY's here... */
} BHEADER;


/* The block trailer at end of block file */
typedef struct {
   byte phash[HASHLEN];    /* previous block hash (32) */
   byte bnum[8];           /* this block number */
   byte mfee[8];           /* minimum transaction fee */
   byte tcount[4];         /* transaction count */
   byte time0[4];          /* to compute next difficulty */
   byte difficulty[4];
   byte mroot[HASHLEN];  /* hash of all TXQENTRY's */
   byte nonce[HASHLEN];
   byte stime[4];        /* unsigned start time GMT seconds */
   byte bhash[HASHLEN];  /* hash of all block less bhash[] */
} BTRAILER;

#define BTSIZE (32+8+8+4+4+4+32+32+4+32)


/* ledger entry in ledger.dat */
typedef struct {
   byte addr[TXADDRLEN];    /* 2208 */
   byte balance[TXAMOUNT];  /* 8 */
} LENTRY;

/* ledger transaction ltran.tmp, el.al. */
typedef struct {
   byte addr[TXADDRLEN];    /* 2208 */
   byte trancode[1];        /* '-' = debit, 'A' = credit (sorts last!) */
   byte amount[TXAMOUNT];   /* 8 */
} LTRAN;


/* for mtx */
/* takes TX * or TXQENTRY pointer */
#define ismtx(tx) ((tx)->dst_addr[2196] == 0x00 \
                    && (tx)->dst_addr[2197] == 0x01)

#define ADDR_TAG_LEN 12
#define NR_DST 100       /* number of tags (MDST) in MTX dst[] */
#define NR_DZEROS 208    /* length of MTX zeros[] */

typedef struct {
   byte tag[ADDR_TAG_LEN];    /* Tag value for MTX multi-destination. */
   byte amount[8];            /* MTX Send Amount, to this tag. */
} MDST;

/* Structure for multi-tx is padded to same size as TXQENTRY. */
typedef struct {
   /* start transaction buffer (These fields are order dependent) */
   byte src_addr[TXADDRLEN];     /*  2208 */

   /* dst[] plus zeros[] is same size as TX dst_addr[]. */
   MDST dst[NR_DST];
   byte zeros[NR_DZEROS];  /* padding - reserved - must follow dst[] */

   byte chg_addr[TXADDRLEN];
   byte send_total[TXAMOUNT];    /* 8 */
   byte change_total[TXAMOUNT];
   byte tx_fee[TXAMOUNT];
   byte tx_sig[TXSIGLEN];        /* 2144 */
   byte tx_id[HASHLEN];          /* 32 */
} MTX;
