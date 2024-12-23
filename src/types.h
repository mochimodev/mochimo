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

/* v3.0 Hash-based Address definitions
 * [<--------- 40 byte Address --------->]
 * [<- 20 byte Tag ->][<- 20 byte Hash ->]
 * ^ADDR_TAG_OFF/PTR  ^ADDR_HASH_OFF/PTR
 */

/** Full Address length, in bytes */
#define ADDR_LEN           40
/** Address Reference length, in bytes */
#define ADDR_REF_LEN       16
/** Address Tag length, in bytes */
#define ADDR_TAG_LEN       20
/** Address Hash length, in bytes */
#define ADDR_HASH_LEN      20
/** Offset, in bytes, at which a tag begins within an address */
#define ADDR_TAG_OFF       0
/** Offset, in bytes, at which a hash begins within an address */
#define ADDR_HASH_OFF      20
/** Pointer to the tag within an address */
#define ADDR_TAG_PTR(ptr)  ( ((word8 *) (ptr)) + ADDR_TAG_OFF )
/** Pointer to the hash within an address */
#define ADDR_HASH_PTR(ptr) ( ((word8 *) (ptr)) + ADDR_HASH_OFF )

/* LEGACY WOTS+ Address and Tag definitions */

/** Full WOTS+ Address length, in bytes */
#define WOTS_ADDR_LEN      2208
/** WOTS+ Public Key length, in bytes */
#define WOTS_PK_LEN        2144
/** WOTS+ Signature length, in bytes */
#define WOTS_SIG_LEN       2144
/** WOTS+ Address Tag length, in bytes */
#define WOTS_TAG_LEN       12
/** Offset, in bytes, at which a tag begins within an address */
#define WOTS_TAG_OFF       ( WOTS_ADDR_LEN - WOTS_TAG_LEN )
/** Pointer to the tag within a WOTS+ address */
#define WOTS_TAG_PTR(ptr)  ( ((word8 *) (ptr)) + WOTS_TAG_OFF )

/* Digital Signature Algorithm (DSA) and Transaction (TX) definitions */

/* Minimum length of a Transaction (received via network) */
#define TXLEN_MIN ( sizeof(TXHDR) + sizeof(MDST) + sizeof(WOTSVAL) )

/* Minimum length of a Transaction (on disk) */
#define TXLEN_DSK_MIN ( TXLEN_MIN + sizeof(TXTLR) )

/**
 * VANITY ADDRESSES ARE NOT SUPPORTED BY THE DIRECTION OF THE CODEBASE.
 * One day, someone will ask, "Hey, wouldn't it be cool if we could?".
 * Well, yes... but actually NO. Vanity Addresses rely on "direct" account
 * creation. Version 3.0 uses "implicit" account creation. Vanity address
 * creation may be used to hijack transactions to addresses using implicit
 * address creation, by withholding transactions to the implicit address
 * until the vanity address is created. Any protections against this have
 * thus far been deemed insufficient or inappropriate to implement.
 */
#define TXDAT_VANITY "ERROR: Vanity Addresses are not supported."

/** Transaction Data type code from first TX options byte */
#define TXDAT_TYPE(options) ( ((word8 *) (options))[0] )
/** Multi-Destination type transaction data */
#define TXDAT_MDST  0x00

/** Transaction DSA type code from second TX options byte */
#define TXDSA_TYPE(options)  ( ((word8 *) (options))[1] )
/** WOTS+ DSA type transaction validation data */
#define TXDSA_WOTS  0x00

/** Multi-Destination transaction count from third TX options byte */
#define MDST_COUNT(options) ( ((word8 *) (options))[2] + 1 )
/* ... options byte for MDST count is zero-based (++) */

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

/**
 * @struct MDST
 * Multi-Destination structure.
 * ```
 * [<-------------------- 44 byte Destination -------------------->] []
 * [<- 20 byte Tag ->][<- 16 byte Reference ->][<- 8 byte Amount ->]
 * ```
 *
 * @property MDST::tag[ADDR_TAG_LEN]
 * Destination address tag. Contains the destination address tag.
 *
 * @property MDST::ref[ADDR_REF_LEN]
 * (Optional) destination reference. Zero-filled where unused.
 * Destination reference fields must adhere to the following rules:
 * - CONTAINS only uppercase [A-Z], digit [0-9], dash [-], null [\0]
 * - SHALL be null terminated with remaining unused bytes zeroed
 *   - (e.g. VALID   (char[]) { 'A','-','1','\0','\0','\0', ... } )
 *   - (e.g. INVALID (char[]) { 'A','-','1','\0','B','\0', ... } )
 * - MAY have multiple uppercase OR digits (NOT both) grouped together
 * - SHALL only contain a dash to separate groups of uppercase or digit
 * - SHALL NOT contain consecutive groups of the same group type
 *   - (e.g. VALID   "AB-00-EF", "123-CDE-789", "ABC", "123")
 *   - (e.g. INVALID "AB-CD-EF", "123-456-789", "ABC-", "-123")
 *
 * @property MDST::amount[8]
 * Destination send amount. Contains the amount to be sent.
 */
typedef struct {
   word8 tag[ADDR_TAG_LEN];
   char ref[ADDR_REF_LEN];
   word8 amount[8];
} MDST;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(MDST) == ( ADDR_REF_LEN + ADDR_TAG_LEN + 8 ), MDST_size);

/**
 * @struct WOTSVAL
 * WOTS+ validation structure.
 *
 * @property WOTSVAL::signature[WOTS_SIG_LEN]
 * WOTS+ Address signature.
 *
 * @property WOTSVAL::pub_seed[32]
 * WOTS+ Address Public Seed.
 *
 * @property WOTSVAL::adrs[32]
 * WOTS+ Hash Function Address Scheme. The last 12 bytes of the Address
 * Scheme MUST contain the hexadecimal string "420000000e00000001000000".
 */
typedef struct {
   word8 signature[WOTS_SIG_LEN];
   word8 pub_seed[32];
   word8 adrs[32];
} WOTSVAL;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(WOTSVAL) == ( WOTS_SIG_LEN + 32 + 32 ), WOTSVAL_size);

/**
 * @struct TXHDR
 * Transaction Header structure.
 *
 * @property TXHDR::options[4]
 * Transaction options. Specifically, the first byte indicates the
 * Transaction Data type (TXDAT), the second byte indicates the Transaction
 * Digital Signature Algorithm (TXDSA) type used to sign the transaction,
 * and the third and fourth bytes are reserved for additional information
 * regarding the data within the transaction, per it's transaction type.
 * As an example, a standard WOTS+ transaction with 24 destinations is:
 * - options[0] = TXTYPE_MDST; (0x00: standard transaction)
 * - options[1] = TXDSA_WOTS; (0x00: WOTS+ signature)
 * - options[2] = 23; (23 additional, 24 total destinations)
 * - options[3] = 0; (reserved)
 *
 * @property TXHDR::blk_to_live[8]
 * Transaction block-to-live, expiration indicator.
 * For a block-to-live value of zero, the transaction can never expire.
 * For non-zero block-to-live values, the transaction expires for all
 * block numbers greater than the block-to-live value. Additionally, a
 * block-to-live value is considered invalid if it exceeds the block
 * number by more than 256 blocks into the future.
 */
typedef struct {
   word8 options[4];
   word8 src_addr[ADDR_LEN];
   word8 chg_addr[ADDR_LEN];
   word8 send_total[8];
   word8 change_total[8];
   word8 fee_total[8];
   word8 blk_to_live[8];
} TXHDR;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(TXHDR) == ( 4 + (ADDR_LEN * 2) + (8 * 4) ), TXHDR_size);

/**
 * @union TXDAT
 * Transaction Data union/structure. Holds transaction data for various
 * Transaction (TX) types used in transactions.
 *
 * @property TXDAT::mdst[256]
 * Multi-Destination array. Holds up to 256 destinations.
 */
typedef union {
   MDST mdst[/* up to */ 256];
/* ... additional transaction data types here */
} TXDAT;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(TXDAT) == ( sizeof(MDST) * 256 ), TXDAT_size);

/**
 * @union TXDSA
 * Transaction Validation union/structure. Holds validation data for
 * various Digital Signature Algorithms (DSA) used in transactions.
 *
 * @property TXDSA::wots
 * WOTS+ DSA validation data.
 */
typedef union {
   WOTSVAL wots;
/* CRYSTALSVAL crystal; *//* FIPS 204 ML-DSA (Module Lattice) */
/* SPHINCSVAL sphincs; *//* FIPS 205 SLH-DSA (Stateless Hash-Based) */
/* FALCONVAL falcon; *//* FIPS 206 FN-DSA (fast-Fourier Transform over NTRU-Lattice-Based) */
} TXDSA;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(TXDSA) == sizeof(WOTSVAL), TXDSA_size);

/**
 * @struct TXTLR
 * Transaction Trailer structure.
 *
 * @property TXTLR::nonce[8]
 * Transaction nonce. The transaction nonce ensures the integrity of unique
 * transaction IDs (within reasonable doubt) by always containing the block
 * number of the block it is solved into.
 *
 * @property TXTLR::id[HASHLEN]
 * Transaction ID. SHA-256 hash of the transaction contents (incl. nonce).
 */
typedef struct {
   word8 nonce[8];
   word8 id[HASHLEN];
} TXTLR;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(TXTLR) == ( 8 + HASHLEN ), TXTLR_size);

/**
 * @struct TXENTRY
 * Transaction Entry structure. Holds a complete transaction entry.
 *
 * @property TXENTRY::buffer
 * Transaction buffer. Holds the complete transaction entry, in a single
 * contiguous block of memory.
 *
 * @property TXENTRY::tx_sz
 * Transaction size within the buffer.
 *
 * @property TXENTRY::hdr
 * Pointer to the Transaction Header structure within the buffer.
 *
 * @property TXENTRY::dat
 * Pointer to the Transaction Data structure within the buffer.
 *
 * @property TXENTRY::dsa
 * Pointer to the Transaction DSA (validation) structure within the buffer.
 *
 * @property TXENTRY::tlr
 * Pointer to the Transaction Trailer structure within the buffer.
 */
typedef struct {
   /* (AVOID DIRECT USAGE) transaction buffer */
   word8 buffer[
      sizeof(TXHDR) +
      sizeof(TXDAT) +
      sizeof(TXDSA) +
      sizeof(TXTLR)
   ];
   size_t tx_sz;

   TXHDR *hdr;
   TXDAT *dat;
   TXDSA *dsa;
   TXTLR *tlr;

   /** @todo convenience pointers exist for ease of access to specific
    * transaction parameters while maintaining a contiguous transaction
    * within the buffer -- it's also considerably unappealing. Anonymous
    * structures/unions are not supported in C99. Consider definitions or
    * getter functions for dereferencing parameters as local pointers.
    */

   /* convenience pointers to relative locations in buffer */

   word8 *options;
   word8 *src_addr;
   word8 *chg_addr;
   word8 *send_total;
   word8 *change_total;
   word8 *tx_fee;
   word8 *tx_btl;
   MDST *mdst;
   WOTSVAL *wots;
   word8 *tx_nonce;
   word8 *tx_id;
} TXENTRY;
/* assertion NOT required */

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
   word8 __align_crc16[1]; /* (re)align crc16 to 4 byte boundary (UNUSED) */
   word8 crc16[2];
   word8 trailer[2];  /* 0xcd, 0xab */
} TX;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(TX) == ( (2 * 5) + 8 + 8 + (32 * 3) + 2 + (WORD16_MAX + 1) + 2 + 2 ), TX_size);

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
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(NGHEADER) == ( 4 + 8 ), NGHEADER_size);

/* The block header */
typedef struct {
   word8 hdrlen[4];           /* header length to tran array */
   word8 maddr[ADDR_TAG_LEN]; /* mining address */
   word8 mreward[8];
   /*
    * array of tcount TXENTRY's here...
    */

   /* ... appended with BTRAILER */
} BHEADER;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(BHEADER) == ( 4 + ADDR_TAG_LEN + 8 ), BHEADER_size);

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
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(BTRAILER) == ( 32 + 8 + 8 + (4 * 3) + (HASHLEN * 2) + 4 + HASHLEN ), BTRAILER_size);

/**
 * Hash-based ledger entry struct
*/
typedef struct {
   word8 addr[ADDR_LEN];   /* ledger entry address (incl. tag) */
   word8 balance[8];       /* ledger entry balance */
} LENTRY;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(LENTRY) == ( ADDR_LEN + 8 ), LENTRY_size);

/**
 * @struct LTRAN
 * ledger transaction struct for ltran.tmp, el.al.
 * Used to coo the intermediate step between block validation and ledger update.
 *
 * @property LTRAN::addr[ADDR_LEN]
 * Ledger transaction address.
 *
 * @property LTRAN::trancode[1]
 * Ledger transaction code. Indicates the type of transaction:
 * - '-' = debit
 * - 'A' = credit
 * - 'H' = rehash (sorts last!)
 *
 * @property LTRAN::amount[8]
 * Ledger transaction amount.
 */
typedef struct {
   word8 addr[ADDR_LEN];
   char trancode[1];
   word8 amount[8];
} LTRAN;
/* structure packing assertion required ... */
STATIC_ASSERT(sizeof(LTRAN) == ( ADDR_LEN + 1 + 8 ), LTRAN_size);

/* end include guard */
#endif
