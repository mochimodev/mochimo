/**
 * @file types.h
 * @brief Mochimo Definitions and Structures
 * @details Provides basic definitions for operation codes, network
 * and transaction protocol constants, buffer access definitions,
 * compatibility bits and various structures.
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TYPES_H
#define MOCHIMO_TYPES_H


/* internal support */
#include "error.h"

/* external support */
#include "extint.h"
#include <stdio.h>
#include <time.h>

/* ---------------- BEGIN CONFIGURABLE DEFINITIONS --------------------- */

/* Version checking */
#define PVERSION     5        /* protocol version number (short) */

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

/* ---------------- END CONFIGURABLE DEFINITIONS ----------------------- */

/* ---------------- DO NOT CHANGE BELOW THIS LINE ---------------------- */

/* boolean return codes */
#ifndef TRUE
	#define TRUE   1   /**< Boolean value for TRUE (#ifndef) */
#endif
#ifndef FALSE
	#define FALSE  0   /**< Boolean value for FALSE (#ifndef) */
#endif

/* 64-bit compound literal definitions */

/**
 * Make a 64-bit compound literal from a 32-bit value.
 * @param x 32-bit value to build
 */
#define CL64_32(x) ( (word8 *) ((word32[2]) { x }) )

/* 64-bit compound literal (word8 *) representing the mining fee (MFEE) */
#define MFEE64 ( (word8 *) ((word32[2]) { MFEE }) )
/** 64-bit compound literal (word8 *) representing a value of One (1) */
#define ONE64  ( (word8[8]) { 1 } )
/** 64-bit compound literal (word8 *) representing a value of Zero (0) */
#define ZERO64 ( (word8[8]) { 0 } )

#define HASHLEN   32  /**< Digest length of core hashes - SHA256LEN */

/* status return codes */
#define VEWAITING ( -2 )   /**< Socket waiting status */
#define VETIMEOUT ( -1 )   /**< Socket timeout status */
#define VEOK         0     /**< OK/Success status */
#define VERROR       1     /**< General error status */
#define VEBAD        2     /**< Client was bad status */
#define VEBAD2       3     /**< Client was naughty status */
/* NOTE: when unsure if VEBAD or VEBAD2 is more appropriate, think about
 * chain splits. For example, if block validation failed on ledger balance
 * mismatch, it's possible the transactions between chains differ resulting
 * in different ledgers, so we use VEBAD. In a similar example, if block
 * validation failed on block hash mismatch, differing chains do not affect
 * the process used to hash a block, so we use VEBAD2.
 */

/* network/transmission definitions */
#define PORT1     2095     /**< Default TCP listening port for network */
#define PORT2     2096     /**< Secondary port, primarily for testnet */
#define TXEOT     0xabcd   /**< End-of-transmission id for packets */
#define TXNETWORK 1337     /**< Network TX protocol version */
#define TXSIGLEN  2144     /**< Standard transaction signature length */
#define TXADDRLEN 44       /**< Hashed transaction address length */
#define TXWOTSLEN 2208     /**< WOTS+ transaction address length */
#define TXTAGLEN  12
#define TXAMOUNT  8        /**< Standard transaction amount length */

#define TRANBUFF(tx) ( (tx)->src_addr )  /**< Transaction buffer accessor */
#define TRANLEN      ( (TXWOTSLEN*3) + (TXAMOUNT*3) + TXSIGLEN ) /**< Total Transaction length */
#define TXBUFF(tx)   ( (word8 *) tx )    /**< Transaction packet accessor */
#define TXBUFFLEN    ( (2*5) + (8*2) + (32*3) + 2 + TRANLEN + 2 + 2 )
#define TXSIG_INLEN  (TRANLEN - TXSIGLEN)
#define TXCRC_INLEN  ( (2*5) + (8*2) + (32*3) + 2 + TRANLEN )

/* Digital Signature Algorithm definitions */

/** Obtain DSA type code from Transaction Entry pointer */
#define DSA_TYPE(tx) ( get32(((TXQENTRY *) (tx))->tx_adrs + 28) )

/** Invalid DSA type */
#define DSA_NONE  0x00
/** WOTS+ DSA type */
#define DSA_WOTS  0x01

/* Address and Tag definitions */

/** Offset, in bytes, at which a tag begins within an address */
#define TAGOFFSET          ( TXADDRLEN - TXTAGLEN )
#define ADDR_TAG_PTR(addr) ( ((word8 *) (addr)) + TAGOFFSET )
#define ADDR_HAS_TAG(addr) \
   ( *(ADDR_TAG_PTR(addr)) && *(ADDR_TAG_PTR(addr)) != 0x42 )

/* eXtended Transaction (XTX) definitions */

/** Conditional test for an eXtended Transaction */
#define IS_XTX(tx)      ( ((TXQENTRY *)(tx))->dst_addr[TAGOFFSET] == 0 )
/** Type code (word8) for an eXtended Transaction */
#define XTX_TYPE(tx)    ( ((TXQENTRY *)(tx))->dst_addr[TAGOFFSET + 1] )
/** eXtended Data count (not length) for an eXtended Transaction */
#define XTX_COUNT(tx)   ( ((TXQENTRY *)(tx))->dst_addr[TAGOFFSET + 2] )

/** Invalid Transaction type */
#define XTX_NONE   0x00
/** Multi-destination Transaction type */
#define XTX_MDST   0x01
/** Memorandum Transaction type */
#define XTX_MEMO   0x02

/* Historic Compatibility Break Points */
#define V20TRIGGER 0x4321  /* v2.0 open source (adjust diff, reward) */
#define V2001PATCH 0x4521  /* v2.0.1 difficulty patch */
#define V23TRIGGER 0xd431  /* v2.3 pseudoblocks */
#define V24TRIGGER 0x12851 /* v2.4 FPGA-Tough PoW algo */
#define V30TRIGGER 0x9ffff /* v3.0 blockchain reboot */
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
 * Capability bit for opt-in to network activity. Allows a node to
 * receive network activity and be included in network peerlists.
*/
#define C_OPTIN         2

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
 * NACK, or Negative Acknowledgement, indicates that an operation was
 * received but was not processed successfully or was rejected.
 * The reason for the NACK should be included in the buffer.
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

/** Structure for Multi-destination data. */
typedef struct {
   word8 tag[TXTAGLEN]; /* Tag address for XTX multi-destination. */
   word8 amount[8];     /* Send Amount, to this tag. */
} MDST;

/** Union for eXtended Data.
 * Where an eXtended Transaction (XTX) type is indicated, there MAY
 * also be eXtended Data (XDATA) present within the transaction.
*/
typedef union {
   MDST mdst[256];
   /* ... add eXtended Transaction data types here */
} XDATA;

/**
 * @struct TXQENTRY
 * Structure for a standard or eXtended Transaction Entry.
 * For eXtended Transaction Entry details, see TXQENTRY::dst_addr.
 *
 * @property TXQENTRY::dst_addr[TXADDRLEN]
 * Destination address, or eXtended Transaction metadata.
 * Transactions may indicate an eXtended Transaction by using this field
 * as a "metadata" field. In such a case...
 * - dst_addr[0-31] MAY be used to include a transaction reference
 *   - CONTAINS only uppercase [A-Z], digit [0-9], dash [-], null [\0]
 *   - SHALL be null terminated with remaining unused bytes zeroed
 *     - (e.g. VALID   (char[32]) { 'A','-','1','\0','\0','\0', ... } )
 *     - (e.g. INVALID (char[32]) { 'A','-','1','\0','B','\0', ... } )
 *   - MAY have multiple uppercase OR digits (NOT both) grouped together
 *   - SHALL only contain a dash to separate groups of uppercase or digit
 *   - SHALL NOT contain consecutive groups of the same group type
 *     - (e.g. VALID   "AB-00-EF", "123-CDE-789", "ABC", "123")
 *     - (e.g. INVALID "AB-CD-EF", "123-456-789", "ABC-", "-123")
 * - dst_addr[32] SHALL indicate an XTX type transaction (0x00)
 * - dst_addr[33] SHALL indicate the specific type of XTX (>0x00)
 *   - Where the XTX type indicates XTX_MDST...
 *     - dst_addr[33] SHALL indicate Multi-Destination (XTX_MDST)
 *     - dst_addr[34] SHALL indicate the number of destinations (max 256)
 *     - xdata.mdst[] will contain each destination and associated amount
 *     - tx_fee will be adjusted based on the number of destinations + 1
 *   - Where the XTX type indicates XTX_NONE...
 *     - this is illegal; throw it in the bin
 *
 * @property TXQENTRY::tx_btl[8]
 * Transaction block-to-live, expiration indicator.
 * For a block-to-live value of zero, the transaction can never expire.
 * For non-zero block-to-live values, the transaction expires for all
 * block numbers greater than the block-to-live value. Additionally, a
 * block-to-live value is considered invalid if it exceeds the block
 * number by more than 256 blocks into the future.
 *
 * @property TXQENTRY::tx_adrs[HASHLEN]
 * WOTS+ Hash Function Address Scheme, or other Digital Signature Algorithm.
 * Currently only used to validate WOTS+ transaction signatures, provides
 * useable space for identifying alternate Digital Signature Algorithms, or
 * even alternate validation procedures, such as proposed ZCF features.
 * - tx_adrs[20-31] = 0x420000000e00000001000000 (indicates WOTS+)
 * - tx_adrs[20-31] = 0x420000000e00000002000000 (MAY indicate alt DSA)
 * - tx_adrs[20-31] = 0x012345678901234500000000 (MAY indicate ZCF AUTH)
 * - etc.
 *
 * @property TXQENTRY::tx_nonce[8]
 * Transaction nonce.
 * The transaction nonce ensures the integrity of unique transaction IDs
 * (within reasonable doubt) by always containing the block number of the
 * block it is solved into.
*/
typedef struct {
   /* transaction data (These fields are order dependent) */
   word8 src_addr[TXADDRLEN];     /* 44 */
   word8 dst_addr[TXADDRLEN];
/* union {
      MDST mdst[256];
      ... add eXtended Transaction data types here
   } xdata; */                    /* size varies -- see dst_addr docs */
   word8 chg_addr[TXADDRLEN];
   word8 send_total[TXAMOUNT];    /* 8 */
   word8 change_total[TXAMOUNT];
   word8 tx_fee[TXAMOUNT];
   word8 tx_btl[8];               /* 8 -- block-to-live */
   /* validation data */
   word8 tx_sig[TXSIGLEN];        /* 2144 -- signature */
   word8 tx_seed[HASHLEN];        /* 32 -- (public) seed */
   word8 tx_adrs[HASHLEN];        /* 32 -- address scheme */
   /* final transaction nonce and hash (generated by node) */
   word8 tx_nonce[8];
   word8 tx_id[HASHLEN];          /* 32 */
} TXQENTRY;

/**
 * Network transmission packet Multi-byte numbers are little-endian.
 * @todo reconsider struct name to avoid conflict with Block Transaction
 * data; maybe PDU for Protocol Data Unit...
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
   word8 buffer[WORD16_MAX];  /* packet buffer */
MCM_DECL_ALIGNED(4) /* (re)align crc16 to 4 byte boundary */
   word8 crc16[2];
   word8 trailer[2];  /* 0xcd, 0xab */
} TX;

/**
 * Hashed-based neo-genesis block header struct
*/
typedef struct {
   word8 hdrlen[4];  /**< Header length to ledger entry array */
   word8 lbytes[8];  /**< Number of bytes containing the ledger */
   /*
    * array of LENTRY's representing lbytes number of bytes, here...
    */
} NGHEADER;

/* The block header */
typedef struct {
   word8 hdrlen[4];         /* header length to tran array */
   word8 maddr[TXADDRLEN];  /* mining address */
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
   word8 time0[4];         /* to compute next difficulty */
   word8 difficulty[4];
   word8 mroot[HASHLEN];   /* merkle hash of block contents */
   word8 nonce[HASHLEN];   /* solving nonce of standard blocks */
   word8 stime[4];         /* time of solve in seconds since the Epoch */
   word8 bhash[HASHLEN];   /* hash of block trailer less bhash[] */
} BTRAILER;

/**
 * Hash-based ledger entry struct
*/
typedef struct {
   word8 addr[TXADDRLEN];     /* ledger entry address (incl. tag) */
   word8 balance[TXAMOUNT];   /* ledger entry balance */
} LENTRY;

/* ledger transaction ltran.tmp, el.al. */
typedef struct {
   word8 addr[TXADDRLEN];    /* 44 */
   word8 trancode[1];        /* '-' = debit, 'A' = credit (sorts last!) */
   word8 amount[TXAMOUNT];   /* 8 */
} LTRAN;

/* end include guard */
#endif
