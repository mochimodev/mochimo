/**
 * @file types.h
 * @brief Mochimo Definitions and Structures
 * @details Provides basic definitions for integer return values (VEOK,
 * VERROR, etc.), operation codes, network and transaction protocol
 * constants, buffer access definitions, and compatibility bits. Also
 * provides struture definitions for a TX packet, TXQENTRY, BHEADER,
 * BTRAILER, LENTRY, LTRAN, MDST and MTX.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TYPES_H
#define MOCHIMO_TYPES_H


/* external support */
#include <time.h>
#include "extio.h"
#include "extint.h"

/* ------ Begin Dev Section -----*/

/* Version checking */
#define PVERSION      4      /* protocol version number (short) */

/* Adjustable Parameters */
#define MAXNODES     37       /**< maximum number of connected nodes */
#define INIT_TIMEOUT 3        /**< initial timeout after accept() */
#define ACK_TIMEOUT  10       /**< hello_ack timeout in callserver() */
#define STD_TIMEOUT  10       /**< connection timeout in callserver() */
#define LQLEN        100      /**< listen() queue length */
#define TXQUEBIG     32       /**< big enough to run bcon */
#define MAXBLTX      32768    /**< max TX's in a block for bcon (~1M) */
#define STATUSFREQ   10       /**< status display interval sec. */
#define CPINKLEN     100       /* maximum entries in pinklists */
#define LPINKLEN     100
#define EPINKLEN     100
#define EPOCHMASK    15       /**< update pinklist Epoch count - 1 */
#define EPOCHSHIFT   4
#define RPLISTLEN    64       /**< recent peer list v.28 */
#define TPLISTLEN    32       /**< trusted peer list */
#define CRCLISTLEN   1024     /**< recent tx crc's */
#define MAXQUORUM    16       /**< for init */
#define BCONFREQ     10       /**< Run con at least */
#define CBITS        0        /**< 8 capability bits for TX */
#define MFEE         500

#define UBANDWIDTH   14300    /**< Dynamic upload bandwidth -- not zero */

#define BCDIR        "bc"     /**< rename to dir for block storage */
#define SPDIR        "split"  /**< rename to dir for backup storage */
#define NGDIR        "ng"     /**< rename to dir for neogen storage */

#define ERRFNAME     "error.log" /**< default error log filename */
#define LOGFNAME     "mochi.log" /**< default standard log filename */

/* ------ end Dev Section  -----*/

/* ----------------- DO NOT CHANGE BELOW THIS LINE --------------------- */

/* boolean return codes */
#ifndef TRUE
	#define TRUE   1   /**< Boolean value for TRUE (#ifndef) */
#endif
#ifndef FALSE
	#define FALSE  0   /**< Boolean value for FALSE (#ifndef) */
#endif

#define HASHLEN   32  /**< Digest length of core hashes - SHA256LEN */

/* status return codes */
#define VEOK      0        /**< Status return code - No error */
#define VERROR    1        /**< Status return code - General error */
#define VEBAD     2        /**< Status return code - client was bad */
#define VEBAD2    3        /**< Status return code - client was naughty */
#define VETIMEOUT (-1)     /**< Status return code - socket timeout */

/* network/transmission definitions */
#define PORT1     2095     /**< Default TCP listening port for network */
#define PORT2     2096     /**< Secondary port, primarily for testnet */
#define TXEOT     0xabcd   /**< End-of-transmission id for packets */
#define TXNETWORK 1337     /**< Network TX protocol version */
#define TXSIGLEN  2144     /**< Standard transaction signature length */
#define TXADDRLEN 96       /**< Hashed transaction address length */
#define TXWOTSLEN 2208     /**< WOTS+ transaction address length */
#define TXTAGLEN  12
#define TXAMOUNT  8        /**< Standard transaction amount length */

#define TRANBUFF(tx) ( (tx)->src_addr )  /**< Transaction buffer accessor */
#define TRANLEN      ( (TXWOTSLEN*3) + (TXAMOUNT*3) + TXSIGLEN ) /**< Total Transaction length */
#define TXBUFF(tx)   ( (word8 *) tx )    /**< Transaction packet accessor */
#define TXBUFFLEN    ( (2*5) + (8*2) + (32*3) + 2 + TRANLEN + 2 + 2 )
#define TXSIG_INLEN  (TRANLEN - TXSIGLEN)
#define TXCRC_INLEN  ( (2*5) + (8*2) + (32*3) + 2 + TRANLEN )

#define TX_IS_MTX(tx) \
   ((tx)->dst_addr[2196] == 0x00 && (tx)->dst_addr[2197] == 0x01)
#define MDST_NUM_DST 100       /* number of tags (MDST) in MTX dst[] */
#define MDST_NUM_DZEROS 208    /* length of MTX zeros[] */

#define ADDR_TAG_PTR(addr) (((word8 *) (addr)) + 2196)
#define ADDR_HAS_TAG(addr) \
   (((word8 *) (addr))[2196] != 0x42 && ((word8 *) (addr))[2196] != 0x00)

/* Historic Compatibility Break Points */
#define DTRIGGER31 17185   /* for v2.0 new set_difficulty() */
#define WTRIGGER31 17185   /* for v2.0 new add_weight() */
#define RTRIGGER31 17185   /* for v2.0 new get_mreward() */
#define FIXTRIGGER 17697   /* for v2.0 difficulty patch */
#define V23TRIGGER 54321   /* for v2.3 pseudoblocks */
#define V24TRIGGER 0x12851 /* for v2.4 new FPGA Tough algo */
#define MTXTRIGGER 133333  /* MTX flag activation block */
#define BRIDGE     949     /* Trouble time -- Edit for testing */

/* break point trigger detection MACROs */
#define NEWYEAR(bnum) ( get32(bnum) >= V23TRIGGER || get32(bnum+4) != 0 )
#define WATCHTIME  (BRIDGE*4+1)  /* minimum watchdog time */
#define TIMES_OF_TROUBLE(bnum) NEWYEAR(bnum)

/**
 * Capability bit for candidate block pushing nodes. Indicates the
 * capability to push Candidate Blocks, primarily to headless miners.
*/
#define C_PUSH          1

/**
 * Capability bit for wallets. Indicates the operation of a wallet.
*/
#define C_WALLET        2

/**
 * Capability bit for nodes activating the Sanctuary Protocol. Indicates
 * the activation of the Sanctuary Protocol.
*/
#define C_SANCTUARY     4

/**
 * Capability bit for nodes indicating a Miner Fee change. For indicating
 * the desired Miner Fee outcome of the Sanctuary Protocol.
*/
#define C_MFEE          8

/**
 * Capability bit for nodes with Logging. Indicates nodes with Logging.
*/
#define C_LOGGING       16

/**
 * Capability bit for nodes using variable PDU protocol.
 * NOTE: implied for protocol version 5 onwards
*/
#define C_VPDU          32

/**
 * "Null" operation code. Not actively used by the node, but can indicate a
 * lack of socket initialization during packet transmission.
*/
#define OP_NULL         0

/**
 * "Hello" operation code. Used to initiate the first step of Mochimo's
 * 3-Way Handshake Protocol.
*/
#define OP_HELLO        1

/**
 * "Hello acknowledgement" operation code. Used to "acknowledge" a "hello"
 * operation code, as the second step of Mochimo's 3-Way Handshake Protocol.
*/
#define OP_HELLO_ACK    2

/**
 * Operation code boundary. Indicates the first valid operation code
 * that can be used after a successful 3-Way Handshake.
*/
#define FIRST_OP        3

/**
 * Transaction operation code. Indicates the presence of a Transaction
 * within the same TX packet.
*/
#define OP_TX           3

/**
 * Block found operation code. Indicates the sender has found a new block.
 * This can indicate either a fresh solve or network distribution.
*/
#define OP_FOUND        4

/**
 * Get blockchain file operation code. Indicates a request for a blockchain
 * file. The number block number should be indicated in the same TX packet.
*/
#define OP_GET_BLOCK     5

/**
 * Get IP/peer list operation code. Indicates a request for a list of peers
 * to use for various communications with the network.
*/
#define OP_GET_IPL      6

/**
 * Send file operation code. Indicates that a TX packet contains at least
 * part of a file, usually in response to a OP_GET_BLOCK or OP_GET_TFILE.
*/
#define OP_SEND_FILE    7

/**
 * Send IP/peer list operation code. Indicates the TX packet contains a
 * list of network peers, usually in response to a OP_GET_IP request.
*/
#define OP_SEND_IPL     8

/**
 * Busy operation code. Indicates that a Node is too busy to respond.
*/
#define OP_BUSY         9

/**
 * No acknowledged operation code. Indicates that a Node acknowledges the
 * request but is unable to respond with meaningful data.
 * @note It should NOT be assumed that a Node will always respond with
 * OP_NACK when it cannot respond. A node may also just close a connection
 * if it cannot respond. Because reasons...
*/
#define OP_NACK         10

/**
 * Get trailer file operation code. Indicates a request for the entire
 * trailer file.
 * @note For requesting only part of a trailer file, see OP_TF.
*/
#define OP_GET_TFILE    11

/**
 * Send balance operation code. Indicates a request for the current balance of
 * an address, directly from the ledger file.
*/
#define OP_BALANCE      12

/**
 * Balance operation code. Indicates that a TX packet contains an address
 * and balance, usually in response to a OP_BALANCE request.
*/
#define OP_SEND_BAL     13

/**
 * Resolve tagged address operation code. Indicates a request to resolve a
 * tagged address and balance.
*/
#define OP_RESOLVE      14

/**
 * Get candidate block operation code. Indicates a request for the latest
 * candidate blockchain file. Used primarily by headless miners.
*/
#define OP_GET_CBLOCK   15

/**
 * Mined block operation code. Indicates a TX packet contains at least
 * part of a mined blockchain file, usually after a candidate block solve.
 * Used primarily by headless miners.
*/
#define OP_MBLOCK       16

/**
 * Block hash operation code. Indicates either a request for the block hash
 * of a particular block number or that a TX packet contains the block hash
 * of a particular block number, as represented in the trailer file.
*/
#define OP_HASH         17

/**
 * Get partial trailer file operation code. Indicates a request for part
 * of the trailer file, as specified in the trailer file.
 * @note For requesting the entire trailer file, see OP_GET_TFILE.
*/
#define OP_TF           18

/**
 * Identify operation code. Indicates either a request for Sanctuary
 * Protocol specifications or that a TX packet contains requested
 * Sanctuary Protocol specifications.
*/
#define OP_IDENTIFY     19

/**
 * Operation code boundary. Indicates the last valid operation code
 * that can be used after a successful 3-Way Handshake.
 * @note Update value when adding operation codes.
*/
#define LAST_OP         19


/* device types (DEVICE_CTX.type) */

#define NO_DEVICE       0  /**< No device */
#define CUDA_DEVICE     1  /**< CUDA device type */
#define OPENCL_DEVICE   2  /**< OPENCL device type */

/* device status (DEVICE_CTX.status) */

#define DEV_STOP  (-2)  /**< Device disabled status */
#define DEV_FAIL  (-1)  /**< Device failure status */
#define DEV_NULL  (0)   /**< Device no status (uninitialized) */
#define DEV_IDLE  (1)   /**< Device idle status */
#define DEV_INIT  (2)   /**< Device initialization status */
#define DEV_WORK  (3)   /**< Device working status */

/* device structs */

typedef struct {
   int id, type, status;            /**< device identification */
   int grid, block, threads;        /**< device config/status */
   unsigned fan, pow, temp, util;   /**< device monitors */
   time_t last_work, last_monitor;  /**< timestamps */
   word64 work, total_work;         /**< work counters */
   char nameId[256];                /**< device properties */
} DEVICE_CTX;  /**< (GPU) Device context for managing device data. */

/**
 * Network transmission packet Multi-byte numbers are little-endian.
 * Structure is checked on start-up for byte-alignment.
 * HASHLEN is checked to be 32.
 */
typedef struct {
   word8 version[2];  /* { PVERSION, Cbits }  */
   word8 network[2];  /* 0x39, 0x05 TXNETWORK */
   word8 id1[2];
   word8 id2[2];
   word8 opcode[2];
   word8 cblock[8];        /* current block num  64-bit */
   word8 blocknum[8];      /* block num for I/O in progress */
   word8 cblockhash[32];   /* sha-256 hash of current block */
   word8 pblockhash[32];   /* sha-256 hash of previous block */
   word8 weight[32];       /* sum of block difficulties (or TX ip map) */
   word8 len[2];  /* length of data in transaction buffer for I/O op's */
   /* start transaction buffer */
   union {
      word8 buffer[WORD16_MAX];
      struct {
         /* legacy access */
         word8 src_addr[TXWOTSLEN];
         word8 dst_addr[TXWOTSLEN];
         word8 chg_addr[TXWOTSLEN];
         word8 send_total[TXAMOUNT];
         word8 change_total[TXAMOUNT];
         word8 tx_fee[TXAMOUNT];
         word8 tx_sig[TXSIGLEN];
      };
   };
   /* end transaction buffer */
   word8 crc16[2];
   word8 trailer[2];  /* 0xcd, 0xab */
} TX;

/* Structure for clean TX queue */
typedef struct {
   /* start transaction buffer (These fields are order dependent) */
   word8 src_addr[TXWOTSLEN];     /*  2208 */
   word8 dst_addr[TXWOTSLEN];
   word8 chg_addr[TXWOTSLEN];
   word8 send_total[TXAMOUNT];    /* 8 */
   word8 change_total[TXAMOUNT];
   word8 tx_fee[TXAMOUNT];
   word8 tx_sig[TXSIGLEN];        /* 2144 */
   word8 tx_id[HASHLEN];          /* 32 */
} TXQENTRY;


/* The block header */
typedef struct {
   word8 hdrlen[4];         /* header length to tran array */
   word8 maddr[TXWOTSLEN];  /* mining address */
   word8 mreward[8];
   /*
    * variable length data here...
    */

   /* array of tcount TXQENTRY's here... */
} BHEADER;


/* The block trailer at end of block file */
typedef struct {
   word8 phash[HASHLEN];    /* previous block hash (32) */
   word8 bnum[8];           /* this block number */
   word8 mfee[8];           /* minimum transaction fee */
   word8 tcount[4];         /* transaction count */
   word8 time0[4];          /* to compute next difficulty */
   word8 difficulty[4];
   word8 mroot[HASHLEN];  /* hash of all TXQENTRY's */
   word8 nonce[HASHLEN];
   word8 stime[4];        /* unsigned start time GMT seconds */
   word8 bhash[HASHLEN];  /* hash of all block less bhash[] */
} BTRAILER;
#define BTSIZE (32+8+8+4+4+4+32+32+4+32)


/* ledger entry in ledger.dat */
typedef struct {
   word8 addr[TXWOTSLEN];    /* 2208 */
   word8 balance[TXAMOUNT];  /* 8 */
} LENTRY;

/* ledger transaction ltran.tmp, el.al. */
typedef struct {
   word8 addr[TXWOTSLEN];    /* 2208 */
   word8 trancode[1];        /* '-' = debit, 'A' = credit (sorts last!) */
   word8 amount[TXAMOUNT];   /* 8 */
} LTRAN;

typedef struct {
   word8 tag[TXTAGLEN];    /* Tag value for MTX multi-destination. */
   word8 amount[8];            /* MTX Send Amount, to this tag. */
} MDST;

/* Structure for multi-tx is padded to same size as TXQENTRY. */
typedef struct {
   /* start transaction buffer (These fields are order dependent) */
   word8 src_addr[TXWOTSLEN];     /*  2208 */

   /* dst[] plus zeros[] is same size as TX dst_addr[]. */
   MDST dst[MDST_NUM_DST];
   word8 zeros[MDST_NUM_DZEROS];  /* padding - reserved - must follow dst[] */

   word8 chg_addr[TXWOTSLEN];
   word8 send_total[TXAMOUNT];    /* 8 */
   word8 change_total[TXAMOUNT];
   word8 tx_fee[TXAMOUNT];
   word8 tx_sig[TXSIGLEN];        /* 2144 */
   word8 tx_id[HASHLEN];          /* 32 */
} MTX;

/* end include guard */
#endif
