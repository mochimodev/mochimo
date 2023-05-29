/**
 * @file types.h
 * @brief Mochimo Definitions and Structures
 * @details Defines everything from basic return types to structures.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TYPES_H
#define MOCHIMO_TYPES_H


/* external support */
#include "exttime.h"
#include "extthrd.h"
#include "extio.h"
#include "extint.h"
#include "extinet.h"

/* ---------------- BEGIN CONFIGURABLE DEFINITIONS --------------------- */

/** socket connection timeout (seconds) */
#define TIMEOUT_CONN 3
/** socket communication timeout (seconds) */
#define TIMEOUT      10
/** listen() queue length */
#define LQLEN        512
/** maximum number of entries in cpinklist */
#define CPINKLEN     1024
/** maximum number of entries in epinklist */
#define EPINKLEN     1024
/** update interval for epinklist */
#define EPINKMASK    15
/** local peer list length */
#define LPLISTLEN    16
/** recent peer list length */
#define RPLISTLEN    32

/** directory for blockchain storage */
#define BCDIR        "bc"
/** directory for ledger transaction storage */
#define LTDIR        "lt"
/** directory for transaction storage */
#define TXDIR        "tx"

/** filename of output log */
#define LOGNAME      "mochimo.log.txt"

/* ---------------- END CONFIGURABLE DEFINITIONS ----------------------- */

/* ---------------- DO NOT CHANGE BELOW THIS LINE ---------------------- */

/* DEPRECATION MACRO, marks functions DECPRECATED for certain compilers */
#if defined(__GNUC__) || defined(__clang__)
   #define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
   #define DEPRECATED __declspec(deprecated)
#else
   #define DEPRECATED
#endif

/* set default "preferred path separator" per OS */
#ifndef PREFERRED_PATH_SEPARATOR
   #ifdef _WIN32
      #define PREFERRED_PATH_SEPARATOR  "\\"
   #else
      #define PREFERRED_PATH_SEPARATOR  "/"
   #endif
#endif

/**
 * Path separator to use when concatenating file paths.
 * Path separator is defined by "PREFERRED_PATH_SEPARATOR", or an
 * OS specific separator ("\\" or "/") resulting in the following:
 * - Windows\\path\\to
 * - UNIX/path/to
*/
#define PATH_SEPARATOR  PREFERRED_PATH_SEPARATOR

/* device type values */

/** No device */
#define NO_DEVICE       0
/** CUDA device type */
#define CUDA_DEVICE     1
/** OPENCL device type */
#define OPENCL_DEVICE   2

/* device status values */

/** Device disabled status */
#define DEV_STOP       -2
/** Device failure status */
#define DEV_FAIL       -1
/** Device no status (uninitialized) */
#define DEV_NULL        0
/** Device idle status */
#define DEV_IDLE        1
/** Device initialization status */
#define DEV_INIT        2
/** Device working status */
#define DEV_WORK        3

/* mochimo return/error types */

/** Indicates the operation is "waiting" (usually on I/O) */
#define VEWAITING      -2
/** Indicates the operation has "timed out" */
#define VETIMEOUT      -1
/** Indicates the operation was "successful" */
#define VEOK            0
/** Indicates the operation encountered an internal error.
 * errno "may" be set for additional details. */
#define VERROR          1
/** Indicates the operation encountered a protocol violation.
 * errno "may" be set for additional details. */
#define VEBAD           2
/** Indicates the operation encountered a protocol violation that was
 * likely malicious. errno "may" be set for additional details. */
#define VEBAD2          3

#ifndef FALSE  /* FALSE may already be defined */
   /** Boolean return value indicating "false" */
	#define FALSE        0

#endif

#ifndef TRUE  /* TRUE may already be defined */
   /** Boolean return value indicating "true" */
	#define TRUE         1

#endif

/* network capability bits */

/** Indicates "push" block capabilities, primarily to headless miners */
#define C_PUSH          1
/** Indicates wallet capabilities */
#define C_WALLET        2
/** Indicates the activation of the Sanctuary Protocol */
#define C_SANCTUARY     4
/** Indicates a change to the standard Mining Fee */
#define C_MFEE          8
/** Indicates a node is recording logs */
#define C_LOGGING       16
/** Indicates a node is capable of a variable PDU
 * NOTE: implied for protocol version 5 onwards */
#define C_VPDU          32

/* network operation codes */

/** Connecting socket operation code */
#define OP_NULL         0
/** Handshake request (hello) code */
#define OP_HELLO        1
/** Handshake response (acnowledgement) code */
#define OP_HELLO_ACK    2
/** First valid (post-handshake) code */
#define FIRST_OP        3
/** Transaction operation code */
#define OP_TX           3
/** Block found operation code */
#define OP_FOUND        4
/** Blockchain file request code */
#define OP_GET_BLOCK    5
/** IP list request code */
#define OP_GET_IPL      6
/** File data response code */
#define OP_SEND_FILE    7
/** IP list response code */
#define OP_SEND_IPL     8
/** server busy response code */
#define OP_BUSY         9
/** Negative acknowledgement response code */
#define OP_NACK         10
/** Trailer File request code */
#define OP_GET_TFILE    11
/** Balance request code */
#define OP_BALANCE      12
/** Balance response code */
#define OP_SEND_BAL     13
/** Resolve a tag to an address request code */
#define OP_RESOLVE      14
/** Candidate block request code */
#define OP_GET_CBLOCK   15
/** Mined (solved) block request code */
#define OP_MBLOCK       16
/** Block hash request/response code */
#define OP_HASH         17
/** Trailer File (partial) request code */
#define OP_TF           18
/** Sanctuary Protocol specifications request code */
#define OP_IDENTIFY     19
/** last valid (post-handshake) code.
 * NOTE: increment when adding more operation codes */
#define LAST_OP         19

/* task wait codes */

/** Waiting for connect (write buff) */
#define IO_CONN         0
/** Waiting for socket (write buff) */
#define IO_SEND         1
/** Waiting for socket (read buff) */
#define IO_RECV         2
/** Finished. Not waiting.
 * NOTE: increment when adding more wait types */
#define IO_DONE         3

/* network data */

/** Default API listening port (HTTP) */
#define PORTA           2086
/** Secondary API port (HTTP), not actively used by main-net */
#define PORTB           2087
/** Default TCP listening port for network communication */
#define PORT1           2095
/** Secondary TCP port, not actively used by main-net */
#define PORT2           2096

/** Network protocol version currently in use */
#define PVERSION        4
/** Network TX protocol version */
#define TXNETWORK       1337
/** End-of-transmission id for packets */
#define TXEOT           0xabcd
/** Packet trailer length. ( 2 + 2 ) = 4 */
#define PKTTLRLEN       4
/** Packet header length. ( (2 * 5) + (8 * 2) + (32 * 3) + 2 ) = 124 */
#define PKTHDRLEN       124
/** Packet (old) buffer length. For v3.0 transition compatibility */
#define PKTBUFFLEN_OLD  8792
/** Packet buffer length */
#define PKTBUFFLEN      0xff00
/** Packet buffer accessor */
#define PKTBUFF(tx)  (  (tx)->buffer  )
/** Packet CRC input length */
#define PKTCRC_INLEN(len)  (  PKTHDRLEN + len  )
/** Identify variable PDU capability bit in node packets */
#define PKT_HAS_C_VPDU(pkt) \
   ( (pkt)->version[0] >= 5 || (pkt)->version[1] & C_VPDU )

/* transaction data */

/** Transaction signature length */
#define TXSIGLEN        2144
/** Transaction tag length */
#define TXTAGLEN        12
/** WOTS+ Transaction address length */
#define TXWOTSLEN       2208
/** Hashed transaction address length */
#define TXADDRLEN       64
/** Transaction amount length */
#define TXAMOUNTLEN     8
/** Offset, in bytes, at which a tag begins in a WOTS+ address */
#define WTAGOFFSET   (  TXWOTSLEN - TXTAGLEN  )
/** Offset, in bytes, at which a tag begins in a Hashed address */
#define TAGOFFSET    (  TXADDRLEN - TXTAGLEN  )
/** eXtented TX MEMO transaction memo len */
#define TXMEMOLEN    (  TXADDRLEN - TXTAGLEN  )
/** Digest length of core hashes (SHA256LEN) */
#define HASHLEN         32
/** Minimum network miner fee */
#define MFEE            500
/** Maximum number of transactions per block */
#define MAXBLTX         32768
/** Number of seconds to bridge pseudo-blocks (in times of trouble) */
#define BRIDGE          949
/** Number of seconds of allowable variance in future solve time checks */
#define STIME_VARIANCE  10

/** Get pointer to tag of WOTS+ address */
#define WOTS_TAGp(addr) (((word8 *) (addr)) + WTAGOFFSET)
/** Identify valid tag in WOTS+ address */
#define WOTS_HAS_TAG(addr) \
   ( *(WOTS_TAGp(addr)) != 0x42 && *(WOTS_TAGp(addr)) != 0x00 )
/** Get pointer to tag of Hashed address */
#define ADDR_TAGp(addr) (((word8 *) (addr)) + TAGOFFSET)
/** Identify valid tag in Hashed address */
#define ADDR_HAS_TAG(addr) \
   ( *(ADDR_TAGp(addr)) != 0x42 && *(ADDR_TAGp(addr)) != 0x00 )
/** Identify eXtended TX type transactions of WOTS+ transactions */
#define TXW_IS_XTX(tx) \
   ( WOTS_TAGp((tx)->dst_addr)[0] == 0x00 \
   && WOTS_TAGp((tx)->dst_addr)[1] > 0x00 )
/* Identify eXtended TX type transactions of Hashed transactions */
#define TX_IS_XTX(tx) \
   ( ADDR_TAGp((tx)->dst_addr)[0] == 0x00 \
   && ADDR_TAGp((tx)->dst_addr)[1] > 0x00 )
/** Identify type of eXtended TX transaction */
#define TXW_XTYPE(tx)   (  WOTS_TAGp((tx)->dst_addr)[WTAGOFFSET + 1]  )
/** Identify type of eXtended TX transaction */
#define TX_XTYPE(tx)    (  ADDR_TAGp((tx)->dst_addr)[TAGOFFSET + 1]  )

/* eXtended TX (transaction) types */

/** Multi-destination TX type */
#define XTYPE_MTX       0x01
/** Simple message TX type (proposed for v3.0?) */
#define XTYPE_MEMO      0x02

/** Number of WOTS+ transaction multi-destination tags */
#define MDST_NUM_DST    100
/** Number of WOTS+ transaction multi-destination zeros[] (padding) */
#define MDST_NUM_DZEROS 208

/* compatibility break points */

/** Block number of v2.0 diff/reward/weight update */
#define V20TRIGGER      0x4321
/** Block number of v2.0 difficulty patch */
#define P20TRIGGER      0x4521
/** Block number of v2.3 pseudoblocks */
#define V23TRIGGER      54321
/** Block number of v2.4 new FPGA Tough algo */
#define V24TRIGGER      75857
/** Block number of multi-destination transaction flag activation */
#define MTXTRIGGER      133333
/** Block number of v3.0 blockchain upgrades */
#define V30TRIGGER      0x80000 /* MUST NOT BE A NEOGENESIS */

/* struct size definitions */

#define BTSIZE 160

/**
 * Hashed neo-genesis block header struct.
*/
typedef struct {
   word8 hdrlen[4];  /**< Header length to ledger entry array */
   word8 lbytes[8];  /**< Number of bytes containing the ledger */
} NGHEADER;

/**
 * Hashed block header struct. excl. Genesis, Neogenesis and Pseudo-blocks.
*/
typedef struct {
   word8 hdrlen[4];           /**< Header length to transaction array */
   word8 maddr[TXADDRLEN];   /**< Public mining address */
   word8 mreward[8];          /**< Block reward */
} BHEADER;

/**
 * WOTS+ block header struct. excl. Genesis, Neogenesis and Pseudo-blocks.
*/
typedef struct {
   word8 hdrlen[4];           /**< Header length to transaction array */
   word8 maddr[TXWOTSLEN];   /**< WOTS+ mining address */
   word8 mreward[8];          /**< Block rewards */
} BHEADER_W;

/**
 * Block trailer struct.
 * Placed at the end of a Blockchain file, or
 * chained together in a Trailer file.
*/
typedef struct {
   word8 phash[HASHLEN];   /* previous block hash (32) */
   word8 bnum[8];          /* this block number */
   word8 mfee[8];          /* minimum transaction fee */
   word8 tcount[4];        /* transaction count */
   word8 time0[4];         /* to compute next difficulty */
   word8 difficulty[4];    /* difficulty of this block */
   word8 mroot[HASHLEN];   /* hash of all transactions */
   word8 nonce[HASHLEN];   /* nonce for proving work */
   word8 stime[4];         /* unsigned solve time GMT seconds */
   word8 bhash[HASHLEN];   /* hash of all block less bhash[] */
} BTRAILER;

/**
 * Device context struct
*/
typedef struct {
   int id, type, status;               /**< device identification */
   int grid, block, threads;           /**< device config/status */
   unsigned fan, pow, temp, util;      /**< device monitors */
   long long last_work, last_monitor;  /**< timestamps */
   long long work, total_work;         /**< work counters */
   char nameId[256];                   /**< device properties */
} DEVICE_CTX;

/**
 * Hashed ledger entry struct.
*/
typedef struct {
   word8 addr[TXADDRLEN];        /**< Hashed address */
   word8 balance[TXAMOUNTLEN];   /**< Balance of address */
} LENTRY;

/**
 * WOTS+ ledger entry struct.
*/
typedef struct {
   word8 addr[TXWOTSLEN];        /**< WOTS+ address */
   word8 balance[TXAMOUNTLEN];   /**< Balance of address */
} LENTRY_W;

/**
 * Hashed ledger transaction struct.
*/
typedef struct {
   word8 addr[TXADDRLEN];     /**< Hashed address */
   /** Transaction type code. In sorted order:
    * - 0 = Not found (balance requests),
    * - 1 = Found (balance requests),
    * - '-' = debit (ledger update),
    * - 'A' = credit (ledger update) */
   word8 trancode[1];
   word8 amount[TXAMOUNTLEN]; /**< Transaction amount */
} LTRAN;

/**
 * Multi-destination transaction destination struct.
 * Holds destination address tag and amount to be distributed to said tag.
*/
typedef struct {
   word8 tag[TXTAGLEN]; /**< Address tag for TXW_MDST multi-destination. */
   word8 amount[8];     /**< TXW_MDST Send Amount, to aforementioned tag. */
} MDST;

/**
 * Network transmission packet struct.
 */
typedef struct {
   word8 version[2];          /**< { PVERSION, Cbits } */
   word8 network[2];          /**< { 0x39, 0x05 } TXNETWORK -- Mochimo */
   word8 id1[2];              /**< handshake identification */
   word8 id2[2];              /**< handshake identification */
   word8 opcode[2];           /**< operation code of this packet */
   word8 cblock[8];           /**< current block num 64-bit */
   word8 blocknum[8];         /**< block num for I/O in progress */
   word8 cblockhash[HASHLEN]; /**< sha-256 hash of current block */
   word8 pblockhash[HASHLEN]; /**< sha-256 hash of previous block */
   word8 weight[32];          /**< sum of block diffs (or TX ip map) */
   word8 len[2];              /**< length of data in buffer for I/O op's */
   word8 buffer[PKTBUFFLEN];  /**< packet buffer (actual data may vary) */
   word8 crc16[2];            /**< CRC16 hash of PKT[124 + PKT.len] */
   word8 trailer[2];          /**< { 0xcd, 0xab } TXEOT -- always */
} PKT;

/**
 * The Node struct. Handles Mochimo network protocol communication.
*/
typedef struct {
   PKT pkt;       /**< data for active socket operations */
   FILE *fp;      /**< FILE pointer data (for recv/send operations) */
   time_t to;     /**< socket inactivity timeout time */
   word32 ip;     /**< connection ip of this task */
   word16 id1;    /**< handshake id#1 */
   word16 id2;    /**< handshake id#2 */
   word8 io[8];   /**< data I/O value (OUTGOING CONNECTIONS ONLY) */
   word16 port;   /**< connection port (OUTGOING CONNECTIONS ONLY) */
   word16 opreq;  /**< request operation (OUTGOING CONNECTIONS ONLY) */
   word16 opcode; /**< the last operation code read or written to pkt */
   word16 iowait; /**< node I/O operation wait type */
   SOCKET sd;     /**< socket descriptor of connection */
   int bytes;     /**< bytes read from, or sent to socket descriptor */
   int status;    /**< status result of latest task or operation */
   char id[16];   /**< stores IPv4 address as string, for id logging */
   /**
    * TEMPORARY: Identifies a VPDU connection.
    * Set by recv_pkt(). Read by send_pkt().
    * @todo adjust after v3.0 */
   int c_vpdu;
} NODE;

/**
 * Tag index struct.
 * Used in the tagged ledger entry index, for O(log n) tag searches.
*/
typedef struct {
   word8 tag[TXTAGLEN]; /**< 12 byte tag address */
   word8 idx[8];        /**< 64-bit index of tagged ledger entry */
} TAGIDX;

/**
 * MEMO type (XTX) public seed transaction structure.
 * Embed a memorandum of up to 52 bytes in a transaction.
 * All bytes succeeding the NULL terminator must be 0, per MEMO spec.
 * NOTE: Addresses used in an XTX MUST be tagged addresses.
*/
typedef struct {
   word8 src_addr[TXADDRLEN];       /**< Transaction source address */
   word8 dst_tag[TXTAGLEN];         /**< Transaction destination tag */
   word8 dst_memo[TXMEMOLEN];       /**< Memo character data */
   word8 chg_addr[TXADDRLEN];       /**< Transaction change address */
   word8 send_total[TXAMOUNTLEN];   /**< Amount to send to destination */
   word8 change_total[TXAMOUNTLEN]; /**< Amount remaining (change) */
   word8 tx_fee[TXAMOUNTLEN];       /**< Transaction fee amount */
   word8 tx_ttl[TXAMOUNTLEN];        /**< Transaction time-to-live */
   word8 tx_spk[TXSIGLEN];          /**< Transaction source public key */
   word8 tx_sig[TXSIGLEN];          /**< Transaction signature */
   word8 tx_id[HASHLEN];            /**< Transaction ID (hash) */
} TX_MEMO;

/**
 * Hashed transaction structure.
 * Featured improvements over the legacy WOTS+ Transaction:
 * - much shorter Hashed addreses (incl. user-configurable data and tag)
 * - a time-to-live parameter specifying the transaction's target block
*/
typedef struct {
   word8 src_addr[TXADDRLEN];       /**< Transaction source address */
   word8 dst_addr[TXADDRLEN];       /**< Transaction destination address */
   word8 chg_addr[TXADDRLEN];       /**< Transaction change address */
   word8 send_total[TXAMOUNTLEN];   /**< Amount to send to destination */
   word8 change_total[TXAMOUNTLEN]; /**< Amount remaining (change) */
   word8 tx_fee[TXAMOUNTLEN];       /**< Transaction fee amount */
   word8 tx_ttl[TXAMOUNTLEN];        /**< Transaction time-to-live */
   word8 tx_spk[HASHLEN];           /**< Transaction source public key */
   word8 tx_sig[TXSIGLEN];          /**< Transaction signature */
   word8 tx_id[HASHLEN];            /**< Transaction ID (hash) */
} TX;

/**
 * Multi-destination type (XTX) WOTS+ transaction structure.
 * dst[] is padded to same size as TXW.
*/
typedef struct {
   /* start transaction buffer (These fields are order dependent) */
   word8 src_addr[TXWOTSLEN];      /**< WOTS+ source address */

   /* dst[] plus zeros[] is same size as WTX dst_addr[]. */
   MDST dst[MDST_NUM_DST];
   word8 zeros[MDST_NUM_DZEROS];    /* padding - must follow dst[] */

   word8 chg_addr[TXWOTSLEN];      /**< WOTS+ change address */
   word8 send_total[TXAMOUNTLEN];   /**< Amount sent to destination/s */
   word8 change_total[TXAMOUNTLEN]; /**< Amount sent to change address */
   word8 tx_fee[TXAMOUNTLEN];       /**< Transaction fee amount */
   word8 tx_sig[TXSIGLEN];          /**< Transaction signature */
   word8 tx_id[HASHLEN];            /**< Transaction ID (hash) */
} TXW_MDST;

/**
 * WOTS+ transaction structure.
 * Uses the WOTS+ public key and tag for addresses.
*/
typedef struct {
   word8 src_addr[TXWOTSLEN];      /**< Transaction source address */
   word8 dst_addr[TXWOTSLEN];      /**< Transaction destination address */
   word8 chg_addr[TXWOTSLEN];      /**< Transaction change address */
   word8 send_total[TXAMOUNTLEN];   /**< Amount to send to destination */
   word8 change_total[TXAMOUNTLEN]; /**< Amount remaining (change) */
   word8 tx_fee[TXAMOUNTLEN];       /**< Transaction fee amount */
   word8 tx_sig[TXSIGLEN];          /**< Transaction signature */
   word8 tx_id[HASHLEN];            /**< Transaction ID (hash) */
} TXW;

/* end include guard */
#endif
