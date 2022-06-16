/* miner.c  The Block Miner  -- Child Process
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 13 January 2018
 *
 * Expect this file to be if-def'd up for various miners.
 *
 */

#ifndef MOCHIMO_MINER_C
#define MOCHIMO_MINER_C


/* internal support */
#include "util.h"
#include "types.h"   /* for standard mochimo datatypes */
#include "network.h" /* for mochimo communication protocols */
#include "peach.h"   /* for peach algorithm */

/* external support */
#include "extint.h"
#include "extio.h"
#include "extmath.h"
#include "extos.h"
#include "extprint.h"
#include "extthread.h"
#include "exttime.h"
#include <signal.h>
#include <stdlib.h>
#include <stdarg.h>

#ifndef GIT_VERSION
   #define GIT_VERSION "<no-version>"

#endif

#ifdef OS_WINDOWS
   #pragma comment(lib, "Comdlg32.lib")
   #include <commdlg.h> /* for GetOpenFileName() */
   #define getpid()  _getpid()
   #define pid_t     int

#endif

#define MEMBLOCKMADDRp(memp)  ( ((BHEADER *) (memp)->data)->maddr )
#define MEMBLOCKBTp(memp)  \
   ( (BTRAILER *) (((word8 *) (memp)->data) + ((memp)->size - BTSIZE)) )
#define MEMBLOCKBNUMp(memp)  ( MEMBLOCKBTp(memp)->bnum )
#define MEMBLOCKMROOTp(memp)  ( MEMBLOCKBTp(memp)->mroot )
#define MEMBLOCKNONCEp(memp)  ( MEMBLOCKBTp(memp)->nonce )
#define MEMBLOCKSTIMEp(memp)  ( MEMBLOCKBTp(memp)->stime )
#define MEMBLOCKBHASHp(memp)  ( MEMBLOCKBTp(memp)->bhash )

#define GPUMAX 64

typedef struct {
   void *data;
   size_t size;
} MALLOC;

typedef struct {
   MALLOC block;
   word32 hostip;
   volatile int tr, ts;
} THREADBLOCK;

typedef struct {
   BTRAILER bt;
   word32 diff;
   word32 rand[3];
   word32 hostip;
   volatile int tr, ts;
} THREADWORK;

word8 Solo;
word8 Getting;
word8 Interval;

void open_dialog(char *filepath, size_t len)
{
#ifdef _WIN32
   OPENFILENAME ofn = { 0 };

   filepath[0] = '\0';
   ofn.lStructSize = sizeof(OPENFILENAME);
   ofn.lpstrFile = filepath;
   ofn.nMaxFile = len;
   ofn.lpstrTitle = "WOTS+ mining address...";
   ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
   GetOpenFileName((LPOPENFILENAME) &ofn);

#else
   static const char *openfilecmd =
      "zenity --file-selection --title=\"WOTS+ mining address...\"";
   FILE *fd;

   fd = popen(openfilecmd, "r");
   if (fd == NULL) {
      perrno(errno, "Failed to open file select dialog");
      return;
   }
   fgets(filepath, (int) len, fd);
   pclose(fd);

#endif

   /* remove trailing newlines */
   char *nl = strchr(filepath, '\n');
   if (nl) *nl = '\0';
}

/**
 * Receive packets from NODE *np, and write/malloc into to memory.
 * Returns: VEOK (0) = good, else error code. */
int recv_malloc(NODE *np, MALLOC *mem)
{
   word16 len;
   word8 *bp;
   TX *tx;

   /* init recv_mem() */
   bp = NULL;
   tx = &(np->tx);
   mem->size = 0;
   mem->data = NULL;

   /* receive packets and write */
   pdebug("recv_malloc(%s): receiving...", np->id);
   while (recv_tx(np, STD_TIMEOUT) == VEOK) {
      /* check recv'd packet */
      if(get16(tx->opcode) != OP_SEND_FILE) {
         pdebug("recv_mem(%s): *** invalid opcode", np->id);
         break;
      }
      /* check len */
      len = get16(tx->len);
      if(len > TRANLEN) {
         pdebug("recv_mem(%s): *** oversized TX length", np->id);
         break;
      }
      /* reallocate and transfer memory */
      if(len > 0) {
         bp = (word8 *) realloc(mem->data, mem->size + len);
         if (bp == NULL) {
            pdebug("recv_mem(%s): *** realloc() error", np->id);
            break;
         }
         memcpy(bp + mem->size, TRANBUFF(tx), len);
         mem->data = bp;
         mem->size += len;
      }
      /* check EOF */
      if(len < TRANLEN) {
         pdebug("recv_mem(%s): EOF", np->id);
         /* success */
         return VEOK;
      }
   }  /* end while */

   /* failure -- cleanup */
   if (mem->data) free(mem->data);

   return VERROR;
}  /* end recv_malloc() */

/**
 * Send packets to NODE *np, from memory allocation.
 * Returns: VEOK (0) = good, else error code. */
int send_malloc(NODE *np, MALLOC *mem)
{
   int result;
   size_t sent;
   word16 len;
   TX *tx;

   /* init */
   result = VEOK;
   sent = len = 0;
   tx = &(np->tx);
   pdebug("send_malloc(%s): sending...", np->id);

   /* read and send packets */
   for (sent = 0; sent < mem->size && Running; sent += len) {
      len = (mem->size - sent) < TRANLEN ? (mem->size - sent) : TRANLEN;
      put16(tx->len, len);
      memcpy(TRANBUFF(tx), ((word8 *) mem->data) + sent, len);
      result = send_tx(np, STD_TIMEOUT);
      if (result != VEOK) break;
   }

   /* check send amount and report */
   if (sent != mem->size) {
      pdebug("send_malloc(%s): invalid send amount", np->id);
      result = VERROR;
   } else pdebug("send_malloc(%s): end of malloc", np->id);

   return result;
}  /* end send_malloc() */

/* get cblock from host */
ThreadProc th_get_cblock(void *args)
{
   static size_t PSEUDOSIZE = sizeof(BHEADER) + sizeof(word32);
   static size_t BHBTSIZE = sizeof(BHEADER) + sizeof(BTRAILER);
   THREADBLOCK *thread = (THREADBLOCK *) args;
   NODE node;

   /* make the call */
   thread->tr = callserver(&node, thread->hostip);
   if (thread->tr == VEOK) {
      /* get solo type work -- cblock */
      thread->tr = send_op(&node, OP_GET_CBLOCK);
      if (thread->tr == VEOK) {
         thread->tr = recv_malloc(&node, &thread->block);
         if (thread->tr == VEOK) {
            if (get16(node.tx.opcode) != OP_SEND_FILE) {
               pdebug("thread_get_cblock(%s): bad opcode", node.id);
               free(thread->block.data);
               thread->block.size = 0;
               thread->tr = VEBAD;
            } else if (thread->block.size != PSEUDOSIZE &&
               (thread->block.size % sizeof(TXQENTRY)) != BHBTSIZE) {
               pdebug("thread_get_cblock(%s): bad cblock size", node.id);
               free(thread->block.data);
               thread->block.size = 0;
               thread->tr = VEBAD;
            }
         }
      }
      /* close and clear socket */
      sock_close(node.sd);
      node.sd = INVALID_SOCKET;
   }

   /* flag thread as finished */
   thread->ts = 1;
   Unthread;
}  /* end th_get_cblock() */

/* send mblock solve to host */
ThreadProc th_give_mblock(void *args)
{
   THREADBLOCK *thread = (THREADBLOCK *) args;
   NODE node;

   /* make the call */
   thread->tr = callserver(&node, thread->hostip);
   if (thread->tr == VEOK) {
      /* indicate call is for mblock */
      thread->tr = send_op(&node, OP_MBLOCK);
      if (thread->tr == VEOK) {
         /* give solo type work -- mblock */
         put16(node.tx.opcode, OP_SEND_FILE);
         thread->tr = send_malloc(&node, &thread->block);
      }
      /* close and clear socket */
      sock_close(node.sd);
      node.sd = INVALID_SOCKET;
   }

   /* flag thread as finished */
   thread->ts = 1;
   Unthread;
}  /* end th_give_mblock() */

/**
 * Get work from a node/pool.
 * Protocol...
 *    Perform Mochimo Network three-way handshake.
 *    Send OP code "Send Block" (OP_SEND_FILE).
 *    Receive data into NODE pointer.
 * Data Received...
 *    tx,              TX struct containing the received data.
 *    tx->len,         16 bit unsigned integer (little endian) containing the
 *                     length (in bytes) of data stored in tx->src_addr.
 *    tx->src_addr,    byte array containing at least 164 bytes of data...
 *       byte 0-159,   160 byte block trailer to be used for mining.
 *       byte 160-163, 32 bit (little endian) host difficulty.
 *       byte 164-178, 16 byte array of random data
 *                      (intended use; avoid duplicate work between workers)
 * Worker Function... */
ThreadProc th_get_work(void *args)
{
   THREADWORK *thread = (THREADWORK *) args;
   word32 *randp;
   NODE node;

   /* make the call */
   thread->tr = callserver(&node, thread->hostip);
   if (thread->tr == VEOK) {
      /* request pool type work */
      thread->tr = send_op(&node, OP_SEND_FILE);
      if (thread->tr == VEOK) {
         thread->tr = recv_tx(&node, STD_TIMEOUT);
         if (get16(node.tx.opcode) != OP_SEND_FILE) thread->tr = VEBAD;
      }
      if (thread->tr == VEOK) {
         /* copy share data to thread argument */
         memcpy(&thread->bt, node.tx.src_addr, sizeof(BTRAILER));
         thread->diff = *((word32 *) (node.tx.src_addr + sizeof(BTRAILER)));
         if (get16(node.tx.len) >= 176) {
            randp = (word32 *) (node.tx.src_addr + 164);
            memcpy(thread->rand, randp, sizeof(thread->rand));
         }
      }
      /* close and clear socket */
      sock_close(node.sd);
      node.sd = INVALID_SOCKET;
   }

   /* flag thread as finished and embed  */
   thread->ts = 1;
   Unthread;
} /* end get_work() */

/**
 * @brief Send work to a node/pool.
 *
 * - Protocol:
 *    -# Perform Mochimo Network three-way handshake
 *    -# Construct solution data in NODE.tx to send
 *    -# Send OP code "Block Found" (OP_FOUND)
 * - Send data:
 *    -# tx; TX struct containing the sent data
 *    -# tx->len; 16-bit word containing the byte length of data
 *       stored in tx->src_addr (little endian)
 *    -# tx->src_addr;
 *       - byte 159:0, block trailer containing valid nonce
 *       - byte 163:160, 32-bit share difficulty (little endian)
 * - Optional data:
 *    -# tx->weight; 24 bytes of space for the workers name,
 *       followed by 8 bytes of space for the workers hashrate.
 *    -# tx->chg_addr; 2208 byte mining address. Note, tx->len
 *       must be increased to 6624 to represent the entire buffer.
 * @returns 0 on success, else 1 on error.
*/
ThreadProc th_give_share(void *args)
{
   THREADWORK *thread = (THREADWORK *) args;
   NODE node;

   /* make the call */
   thread->tr = callserver(&node, thread->hostip);
   if (thread->tr == VEOK) {
      /* give pool type work -- share */
      memcpy(node.tx.src_addr, &thread->bt, sizeof(BTRAILER));
      put32(node.tx.src_addr + sizeof(BTRAILER), thread->diff);
      put16(node.tx.len, sizeof(BTRAILER) + sizeof(word32));
      thread->tr = send_op(&node, OP_FOUND);
      /* close and clear socket */
      sock_close(node.sd);
      node.sd = INVALID_SOCKET;
   }

   /* flag thread as finished and embed  */
   thread->ts = 1;
   Unthread;
}  /* end th_give_share() */

void check_push_peers(void)
{
   /* check minimum threshold */
   if (*Rplist && Rplistidx > 6) return;
   /* (re)download push peers into recent peers list */
   remove("./push.lst");
   http_get("https://mochimo.org/peers/push", "./push.lst", 3);
   read_ipl("./push.lst", Rplist, RPLISTLEN, &Rplistidx);
   if (*Rplist == 0) {
      /* use localhost fallback if none found */
      pwarn("Peers Unavailable! Using localhost...");
      addpeer((word32) aton("127.0.0.1"), Rplist, RPLISTLEN, &Rplistidx);
   }
}

int usage(int ecode)
{
   print(
      "usage: mochiminer [options]\n"
      "   -h,  --host <IPv4>       specify a target ipv4 address, IP\n"
      "   -i,  --interval <num>    time, in seconds, between target calls\n"
      "   -ll, --log-level <num>   level of detail in logging (0-5)\n"
      "   -m,  --maddr <file>      mining address file\n"
      "   -n,  --name <string>     name of miner (POOL MINER ONLY)\n"
      "   -P,  --pool <IPv4>       pool/proxy target ipv4 address\n"
      "   -p,  --port <num>        port number of target\n\n"
   );

   return ecode;
}  /* end usage() */

int main(int argc, char *argv[])
{
#ifdef CUDA
   static DEVICE_CTX GPU[GPUMAX];
   static DEVICE_CTX *CudaGPUs;
   static int num_cuda_gpus;

#endif
   /*
   static DEVICE_CTX *OpenCLGPUs;
   static int num_opencl_gpus; */

   /* thread handling Id's and Arguments */
   static ThreadId tid_get, tid_send[RPLISTLEN];
   static THREADWORK tharg_work, tharg_share[RPLISTLEN];
   static THREADBLOCK tharg_cblock, tharg_mblock[RPLISTLEN];
   /* miner solve handling -- blocks(Solo), btrailers/diffs(pool/proxy) */
   static MALLOC cblock, pblock, *blkp;
   static BTRAILER bt, cbt, *btp;
   static word32 cdiff;

   static double solvework, sharework;
   static double allhps, /* avghps, */ hps;
   static unsigned int p;
   static int num_cpu_threads, terr, count, n, j, ecode, stopped;
   static time_t now, gettime, starttime, statstime, worktime, timeout;
   static word32 stats[3], hostip, shares, bnum;
   static word8 maddr[TXADDRLEN];
   static char mfile[FILENAME_MAX];
   char stickystats[BUFSIZ], ipstr[24];
   char *sp, *vp, *m;

   /* init - defaults */
   sock_startup();
   fix_signals();
   signal(SIGINT, sigterm);
   set_print_level(PLEVEL_LOG);
   strncpy((char *) Weight, "MochiMiner", 23);
   cblock.data = malloc(sizeof(BTRAILER));
   cblock.size = sizeof(BTRAILER);
   sp = stickystats;
   Port = Dstport = PORT1;
   Noprivate = 0;
   Interval = 5;
   Running = 1;
   Solo = 1;

   /* splashscreen */
   print_splash("Mochimo Miner", GIT_VERSION);

   /* accept command line options */
   for (j = 1; Running && j < argc; j++) {
      /* check argument validity */
      if (argv[j][0] != '-') {
USAGE:   return usage(ecode);
      } else if (argument(argv[j], "--", NULL)) {
         pdebug(" -- end of arguments");
         break;
      } else if (argument(argv[j], "-h", "--host")) {
         vp = argvalue(&j, argc, argv);
         if (vp == NULL) mError(USAGE, "invalid host");
         hostip = aton(vp);
         addpeer(hostip, Rplist, RPLISTLEN, &Rplistidx);
         plog("... add host= 0x%" P32x " (%s)", hostip, vp);
      } else if (argument(argv[j], "-i", "--interval")) {
         vp = argvalue(&j, argc, argv);
         if (vp == NULL) mError(USAGE, "invalid interval");
         Interval = aton(vp);
         plog("... set interval= 0x%" P32x " (%s)", Interval, vp);
      } else if (argument(argv[j], "-ll", "--log-level")) {
         vp = argvalue(&j, argc, argv);
         if (vp == NULL) mError(USAGE, "invalid log level");
         set_print_level(atoi(vp));
         plog("... set log level= 0x%x (%s)", atoi(vp), vp);
      } else if (argument(argv[j], "-m", "--maddr")) {
         vp = argvalue(&j, argc, argv);
         if (vp) strncpy(mfile, vp, 23);
         plog("... set maddr file= %s (%s)", mfile, vp);
      } else if (argument(argv[j], "-n", "--name")) {
         vp = argvalue(&j, argc, argv);
         if (vp) strncpy((char *) Weight, vp, 23);
         plog("... set name= %s (%s)", (char *) Weight, vp);
      } else if (argument(argv[j], "-P", "--pool")) {
         Solo = 0;
         plog("... activate pool mining");
         vp = argvalue(&j, argc, argv);
         if (vp) {
            hostip = aton(vp);
            addpeer(hostip, Rplist, RPLISTLEN, &Rplistidx);
            plog("... add host= 0x%" P32x " (%s)", hostip, vp);
         }
      } else if (argument(argv[j], "-p", "--port")) {
         vp = argvalue(&j, argc, argv);
         if (vp == NULL) mError(USAGE, "invalid port");
         Port = Dstport = atoi(vp);
         plog("... set port= %" P16u " (%s)", Dstport, vp);
      }
   }

   /* check solo mining requirements */
   if (Running && Solo) {
#if OS_WINDOWS
      /* on Windows, the open_dialog() function sets the working
       * directory to that of the selected file */
      char dirpath[BUFSIZ] = ".";
      GetCurrentDirectory(BUFSIZ, dirpath);
#endif
      if (*mfile == '\0') {
         open_dialog(mfile, BUFSIZ);
         if (*mfile == '\0') mError(USAGE, "Unspecified Mining Address...");
      } else if (!fexists(mfile)) return perr("%s does not exist!", mfile);
#if OS_WINDOWS
      /* restore working directory */
      SetCurrentDirectory(dirpath);
#endif
      /* ensure sufficient push peers */
      check_push_peers();
   } /* end if (Running && Solo) */

   if (Running) {
      print("\n");
      /* load miner stats */
      if (read_data(stats, sizeof(stats), "miner.stats") > 0) {
         plog("Statistics loaded...");
         Nupdated = stats[2];
         Nsolved = stats[1];
         Hps = stats[0];
      }
      /* identify mining type */
      plog("%s mining enabled...", Solo ? "Solo" : "Pool");
      /* read mining address -- if avaialable */
      if (*mfile) {
         count = read_data(maddr, TXADDRLEN, mfile);
         if (count < 0) return perrno(errno, "I/O failure, %s", mfile);
         if (count != TXADDRLEN) return perr("Invalid size, %s", mfile);
         plog("Mining Address= %s...", addr2str(maddr));
      } else if (!Solo) plog("Mining Address= <unspecified>");
      /* initialize random seed based on multiple entropy */
      srand16(time(&now), now ^ get32(maddr), now ^ (time_t) getpid());
      /* initialize mining devices */
      num_cpu_threads = cpu_cores();
      plog("Logical CPU Cores = %d", num_cpu_threads);

#ifdef CUDA
      CudaGPUs = GPU;
      num_cuda_gpus = peach_init_cuda(CudaGPUs, GPUMAX / 2);
      plog("Cuda Devices = %d", num_cuda_gpus);

#endif
      /*
      OpenCLGPUs = &GPU[num_cuda_gpus];
      num_opencl_gpus = peach_init_opencl(OpenCLGPUs, GPUMAX / 2);
      plog("OpenCL Devices = %d", num_cuda_gpus); */

      /* init main loop */
      count = 0;
      worktime = time(&starttime);
      statstime = starttime - 1;
      gettime = starttime - Interval;
   }  /* end if(Running) */

   /* main loop */
   while (Running) {
      /* chillout -- grab latest time */
      millisleep(Dynasleep);
      time(&now);
      /* check timeout -- if specified */
      if (timeout && difftime(timeout, now) <= 0) {
         /* drop current peer -- reset peers if empty */
         if (hostip == 0) remove32(*Rplist, Rplist, RPLISTLEN, &Rplistidx);
         check_push_peers();
         timeout = 0;
         count++;
      }
      /* check send threads -- cleanup */
      /* NOTE: thread sets tharg_*->ts non-zero when done */
      for (p = 0; p < RPLISTLEN && Running; p++) {
         if (tid_send[p] == 0) continue;
         if ((Solo && tharg_mblock[p].ts) || tharg_share[p].ts) {
            /* cleanup -- join thread, cleanup args and Zero tid */
            pdebug("joining send thread #%u", p);
            terr = thread_join(tid_send[p]);
            if (terr) {
               perrno(terr, "thread_join(send) failed");
               plog("Terminating thread...");
               thread_terminate(tid_send[p]);
            }
            if (Solo) memset(&tharg_mblock[p], 0, sizeof(THREADBLOCK));
            else memset(&tharg_share[p], 0, sizeof(THREADWORK));
            tid_send[p] = 0;
         }
      }
      /* check mining type */
      if (Solo) {
         /* BEGIN SOLO MINING SECTION */
         /* check/start "get" thread -- obtains latest block */
         /* NOTE: thread sets tharg_cblock.ts non-zero when done */
         if (tid_get == 0) {
            /* wait Interval seconds between gets */
            if (difftime(now, gettime) >= Interval) {
               tharg_cblock.hostip = *Rplist;
               pdebug("creating get thread to %s...", ntoa(Rplist, NULL));
               terr = thread_create(&tid_get, &th_get_cblock,
                  &tharg_cblock);
               if (terr) {
                  perrno(terr, "thread_create(cblock) failed");
                  time(&gettime);
                  tid_get = 0;
               }
            }
         } else if (tharg_cblock.ts) {
            /* cleanup -- join thread, reset time and Zero tid */
            pdebug("joining get thread...");
            terr = thread_join(tid_get);
            if (terr) {
               perrno(terr, "thread_join(cblock) failed");
               plog("Terminating thread...");
               thread_terminate(tid_get);
            }
            time(&gettime);
            tid_get = 0;
            /* handle thread results, or trigger timeout event */
            if (tharg_cblock.tr == VEOK) {
               timeout = 0;
               time(&worktime);
               /* shift current block to previous */
               if (pblock.data) free(pblock.data);
               pblock.data = cblock.data;
               pblock.size = cblock.size;
               /* introduce latest cblock */
               cblock.data = tharg_cblock.block.data;
               cblock.size = tharg_cblock.block.size;
               /* rehash mroot with own maddr */
               btp = MEMBLOCKBTp(&cblock);
               memcpy(MEMBLOCKMADDRp(&cblock), maddr, TXADDRLEN);
               sha256(cblock.data, cblock.size - 100, btp->mroot);
               /* check new block and count */
               if (cmp64(btp->bnum, MEMBLOCKBNUMp(&pblock))) {
                  if (!iszero(MEMBLOCKBNUMp(&pblock), 8)) Nupdated++;
               }
            } else if (timeout == 0) timeout = now + 30;
            /* cleanup thread argument */
            memset(&tharg_cblock, 0, sizeof(THREADBLOCK));
         }
#ifdef CUDA
         /* check CUDA miners -- Peach algo solo mining */
         for (stopped = n = 0; n < num_cuda_gpus; n++) {
            btp = MEMBLOCKBTp(&cblock);
            /* update/check cuda devices with current block trailer */
            ecode = peach_solve_cuda(&CudaGPUs[n], btp, 0, &bt);
            /* check for unrecoverable failure */
            if (ecode == VETIMEOUT) stopped++;
            if (ecode == VEOK) {
               /* check block solve */
               if (peach_check(&bt) != VEOK) {
                  perr("peach_check() failed to verify solve!");
               } else {
                  /* find block with matching mroot */
                  btp = MEMBLOCKBTp(&cblock);
                  if (memcmp(btp->mroot, bt.mroot, 32)) {
                     btp = MEMBLOCKBTp(&pblock);
                     if (memcmp(btp->mroot, bt.mroot, 32)) {
                        blkp = NULL;
                     } else blkp = &pblock;
                  } else blkp = &cblock;
                  /* if found, embed solve data and send */
                  if (blkp == NULL) {
                     perr("Failed to find mroot match...");
                  } else {
                     /* embed nonce and stime in bt and hash block */
                     memcpy(btp->nonce, bt.nonce, 32);
                     put32(btp->stime, time(NULL));
                     sha256(blkp->data, blkp->size - 32, btp->bhash);
                     /* send immediately -- spawn senders */
                     for (p = 0; p < Rplistidx; p++) {
                        if (tid_send[p]) continue;
                        /* configure send arguments */
                        tharg_mblock[p].block.data = blkp->data;
                        tharg_mblock[p].block.size = blkp->size;
                        tharg_mblock[p].hostip = Rplist[p];
                        terr = thread_create(&tid_send[p],
                           &th_give_mblock, &tharg_mblock[p]);
                        if (terr) {
                           perrno(terr, "thread_create(mblock) failed");
                           tid_send[p] = 0;
                        }
                     }
                  }
                  /* block solve - print haiku */
                  Nsolved++;
                  solvework += pow(2, btp->difficulty[0]);
                  Hps = solvework / (word32) difftime(now, starttime);
                  print_bup(btp, "Solved");
               }  /* end block solve */
            }  /* end if (ecode == VEOK) */
         }  /* end for (failed = n = 0; n < num_cuda_gpus... */
         /* check for complete stoppage of GPUs */
         if (stopped == num_cuda_gpus) {
            pfatal("All Cuda GPUs exhausted!");
            break;
         }
#endif
         /* END SOLO MINING SECTION */
      } else {
         /* POOL MINING SECTION */
         /* check/start "get" thread -- obtains latest work */
         /* NOTE: thread sets tharg_work.ts non-zero when done */
         if (tid_get == 0) {
            /* wait Interval seconds between gets */
            if (difftime(now, gettime) >= Interval) {
               tharg_work.hostip = *Rplist;
               terr = thread_create(&tid_get, &th_get_work, &tharg_work);
               if (terr) {
                  perrno(terr, "thread_create(work) failed");
                  time(&gettime);
                  tid_get = 0;
               }
            }
         } else if (tharg_work.ts) {
            /* cleanup -- join thread, reset time and Zero tid */
            terr = thread_join(tid_get);
            if (terr) {
               perrno(terr, "thread_join(work) failed");
               plog("Terminating thread...");
               thread_terminate(tid_get);
            }
            time(&gettime);
            tid_get = 0;
            /* handle thread results... */
            if (tharg_work.tr == VEOK) {
               time(&worktime);
               /* introduce latest work */
               memcpy(&cbt, &tharg_work.bt, sizeof(cbt));
               cdiff = tharg_work.diff;
               /* set rand seed from pool */
               if (!iszero(tharg_work.rand, 3 * sizeof(word32))) {
                  srand16(tharg_work.rand[0], tharg_work.rand[1],
                     tharg_work.rand[2]);
               }
            }
            /* cleanup thread argument */
            memset(&tharg_work, 0, sizeof(THREADWORK));
         }
#ifdef CUDA
         /* check CUDA miners -- Peach algo solo mining */
         for (n = 0; n < num_cuda_gpus; n++) {
            /* ensure we're working with the current blocks trailer */
            if (peach_solve_cuda(&CudaGPUs[n], &cbt, cdiff, &bt) == VEOK) {
               /* check pool/proxy requirements */
               if (peach_checkhash(&bt, cdiff, NULL) != VEOK) {
                  perr("peach_checkhash() failed to verify solve!");
               } else {
                  /* add to shares and calc estimated work */
                  shares++;
                  sharework += pow(2, cdiff);
                  /* find available send thread for work */
                  for (p = 0; p < RPLISTLEN && tid_send[p] != 0; p++);
                  if ((p + 1) >= RPLISTLEN) {
                     pwarn("Send work unavailable!");
                  } else {
                     /* copy bt and spawn thread to send work */
                     memcpy(&tharg_share[p].bt, &bt, sizeof(bt));
                     tharg_share[p].diff = cdiff;
                     tharg_share[p].hostip = *Rplist;
                     terr = thread_create(&tid_send[p], &th_give_share,
                        &tharg_share[p]);
                     if (terr) {
                        perrno(terr, "thread_create(mblock) failed");
                        memset(&tharg_share[p], 0, sizeof(THREADWORK));
                        tid_send[p] = 0;
                     }
                  }
                  /* check block solve */
                  if (peach_check(&bt) == VEOK) {
                     /* block solve - print haiku */
                     Nsolved++;
                     solvework += pow(2, bt.difficulty[0]);
                     Hps = solvework / (word32) difftime(now, starttime);
                     print_bup(&bt, "Solved");
                  } else {
                     /* clear "share" for continuous solving */
                     memset(&bt, 0, sizeof(bt));
                  }
               }
            }
         }
#endif
         /* END POOL MINING SECTION */
      }  /* end if (Solo)... else... */
      /* check statstime -- display miner stats */
      if (difftime(now, statstime)) {
         time(&statstime);
         sp[0] = '\0';
         allhps = 0;
#ifdef CUDA
         /* build sticky stats from GPUs */
         for (n = 0; n < num_cuda_gpus; n++) {
            switch (GPU[n].status) {
               case DEV_FAIL:
                  asnprintf(sp, BUFSIZ, "%s Failure...\n", GPU[n].nameId);
                  break;
               case DEV_NULL:
                  asnprintf(sp, BUFSIZ, "GPU#%d; Uninitalized...\n", n);
                  break;
               case DEV_IDLE:
                  asnprintf(sp, BUFSIZ, "%s [%uW:%u°C] No txs...\n",
                     GPU[n].nameId, GPU[n].pow, GPU[n].temp);
                  break;
               case DEV_INIT:
                  asnprintf(sp, BUFSIZ, "%s [%uW:%u°C] Init... (%d%%)\n",
                     GPU[n].nameId, GPU[n].pow, GPU[n].temp,
                     (int) (100 * GPU[n].work) / PEACHCACHELEN);
                  break;
               case DEV_WORK:
                  hps = (double) GPU[n].work;
                  hps /= difftime(time(NULL), GPU[n].last_work);
                  allhps += hps;
                  m = metric_reduce(&hps);
                  asnprintf(sp, BUFSIZ, "%s [%uW:%u°C] %.02lf%sH/s\n",
                     GPU[n].nameId, GPU[n].pow, GPU[n].temp, hps, m);
                  break;
               default: break; /* no info -- likely disabled */
            }  /* end switch (GPU[n].status) */
         }  /* end for (n = 0; n < num_cuda_gpus; n++)... */
#endif
         /* add general stats */
         ntoa(Rplist, ipstr);
         m = metric_reduce(&allhps);
         if (Solo) {
            bnum = (unsigned) get32(MEMBLOCKBNUMp(&cblock));
            asnprintf(stickystats, BUFSIZ,
               "(%d) %s:%"P16u" 0x%x(%u) [%"P32u"/%"P32u" Solved] %.02lf%sH/s\n",
               count, ipstr, Dstport, bnum, bnum, Nsolved, Nupdated, allhps, m);
         } else {
            bnum = (unsigned) get32(cbt.bnum);
            asnprintf(stickystats, BUFSIZ,
               "(%d) %s:%"P16u" 0x%x(%u) [%"P32u" Shares] %.02lf%sH/s\n",
               count, ipstr, Dstport, bnum, bnum, shares, allhps, m);
         }
         /* check timeout indicator */
         if (timeout) {
            asnprintf(sp, BUFSIZ, "-- TIMEOUT: %ds until host rotation --",
               (int) difftime(timeout, time(NULL)));
         }
         /* update stats display */
         psticky("%s", stickystats);
      }  /* end if (difftime(now, statstime)) */
   }  /* end while (Running) */

   /* save miner stats data */
   stats[0] = Hps;
   stats[1] = Nsolved;
   stats[2] = Nupdated;
   if (!iszero(stats, sizeof(stats))) {
      if (write_data(stats, sizeof(stats), "miner.stats") > 0) {
         plog("Statistics saved...");
      }
   }

   /* cleanup */
   plog("Miner exiting...");
   sock_cleanup();
   psticky("");
   plog("");

   return 0;
}  /* end main() */

/* end include guard */
#endif
