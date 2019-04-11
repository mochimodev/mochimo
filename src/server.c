/* server.c   The Mochimo Server  (code name: VERONICA)
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 1 January 2018
 *
 * TCP server code.
*/


/**
 * The Mochimo Server/Client!
 *
 * Uses globals from data.c
 */
int server(void)
{
   static time_t nsd_time;  /* event timers */
   static time_t bctime, mwtime, mqtime;
   static time_t ipltime;
   static SOCKET lsd, nsd;
   static NODE *np, node;
   static struct sockaddr_in addr;
   static int status;   /* child return status */
   static pid_t pid;    /* child pid */
   static int lfd;      /* for lock() */
   static unsigned long hps;  /* same as Hps in monitor.c */
   static word32 bigwait;

   Running = 1;          /* globals are in data.c */

   /* Initialise event timers */
   Ltime = time(NULL);      /* real time GMT in seconds */
   Stime = Ltime + 10;      /* status display time */
   bctime = Ltime + 30;     /* block constructor time */
   mwtime = Ltime + 6;
   mqtime = Ltime + 5;      /* mirror() time */
   Utime = Ltime;           /* for watchdog timer */
   Watchdog = WATCHTIME + (rand2() % 600);
   Bridgetime = Time0 + BRIDGE;  /* pseudo-block timer */
   bigwait = (60*60*24) + (rand2() % 10800);  /* 1 day + ~ 3 hours */
   ipltime = Ltime + (rand2() % 600) + 10;  /* ip list fetch time */

   if((lsd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
      fatal("Cannot open listening socket.");
   memset(&addr, 0, sizeof(addr));    /* clear address structure   */
   addr.sin_port = htons(Port);
   addr.sin_addr.s_addr = INADDR_ANY;
   addr.sin_family = AF_INET;

   show("bind");
   for(;;) {
      if(!Running) { closesocket(lsd); return 0; }
      if(bind(lsd, (struct sockaddr *) &addr, sizeof(addr)) == 0) break;
      plog("Trying to bind port %d...", Port);
      sleep(5);
      if(Monitor && !Bgflag) monitor();
   }

   /* set listening port non-blocking for accept() */
   if(nonblock(lsd) == -1)
      fatal("nonblock() failed on lsd.");
   listen(lsd, LQLEN);  /* LQSIZ */
   nsd = INVALID_SOCKET;

   if(Safemode && !iszero(Cblocknum, 8)) {
      plog("Safemode");
      send_found();
   }
   else plog("Listening...");

   /*
    * Main server loop.
    */

   while(Running) {
      /*
       * Get current time for this generation.
       */
      Ltime = time(NULL);

      show("listen");  /* display status for ps */

      /* Reap zombies and collect status.
       * No child left behind...
       */
      for(np = Nodes; np < Hi_node; np++) {
         if(np->pid == 0) continue;
         pid = waitpid(np->pid, &status, WNOHANG);
         if(pid <= 0) continue;  /* child still running or signal */
         freeslot(np);
         if(Trace) plog("np->pid: %d  pid: %d  status: 0x%x  op: %d  (%d)",
                        np->pid, pid, status, np->opcode, errno);  /* debug */
         /* Adds to lists if needed and returns exit status 0-3 */
         status = child_status(np, pid, status);
         if(np->opcode == OP_FOUND) {
            if(Blockfound == 0) error("server(): line %d", __LINE__);
            else {
               if(update("rblock.dat", 0) == VEOK) {
                  send_found();  /* start send_found() child */
                  addcurrent(np->src_ip);  /* v.28 */
                  addrecent(np->src_ip);   /* v.28 */
#ifdef BX_MYSQL
                  // If we've received a new block from peers update the database
                  if (Exportflag) {
                    printf("Exporting to database.\n");
                    system("../bx -e");
                  }
#endif
               }
               Blockfound = 0;
            }
         }  /* end if OP_FOUND child */
         else if(np->opcode == OP_GETBLOCK || np->opcode == OP_GET_TFILE) {
            if(get16(np->tx.len) == 0 && status == 0) {
               addcurrent(np->src_ip);  /* v.28 */
               addrecent(np->src_ip);
            }
         }
      }  /* end for check Node[] zombies */

      /* Reap a send_found() child.  If she is done, pid != 0. */
      if(Sendfound_pid > 0) {
         pid = waitpid(Sendfound_pid, &status, WNOHANG);
         if(pid > 0) Sendfound_pid = 0;
      }

      /* Check for new connection with accept() and set nsd. */
      if(nsd == INVALID_SOCKET) {
         if((nsd = accept(lsd, NULL, NULL)) != INVALID_SOCKET) {
            nonblock(nsd);
            nsd_time = Ltime;
         }
      }

      /*
       * If accept() good,
       * try to read data from socket, nsd, using gettx().
       */
      if(nsd != INVALID_SOCKET) {
         /* gettx() completes the initial handshake and fills node
          * and some parent tables.  It returns -1 if no data yet.
          * If gettx() completes the transaction, it returns 0, 1, 2, or 3;
          * otherwise it returns sizeof(TX) and needs help from child
          * so getslot() allocates a new np and copies node into it.
          */
         status = gettx(&node, nsd);  /* fills in node */
         if(status != -1) {
            if(status == sizeof(TX) && (np = getslot(&node)) != NULL) {
               pid = fork();  /* create child to handle TX */
               if(pid == 0) {
                  /* in child */
                  exit(execute(np));  /* parent calls waitpid() for status */
               }
               /* parent puts valid child pid in parent table */
               if(pid != -1) np->pid = pid;
               else {
                  /* fork() failed so freeslot() removes child data from 
                   * parent Node[] table.
                   */
                  freeslot(np);
                  error("fork() failed!");
                  restart("cannot fork()");
               }
            }  /* end if need child and slot found */
            /* parent closes its socket if gettx() did not */
            if(node.sd != INVALID_SOCKET)
               closesocket(nsd);
            nsd = INVALID_SOCKET;
         } else {   /* status == -1 no data yet -- so check timeout */
            if(Ltime - nsd_time > INIT_TIMEOUT) {
               Ntimeouts++;  /* log statistics */
               closesocket(nsd);
               nsd = INVALID_SOCKET;
            }
         }  /* end if timeout */
      }  /* endif nsd valid */

      Ngen++;  /* loop counter */

      /*
       * Take care of business...
       */

      /* Check miner */
      if(Blockfound == 0 && exists("mblock.dat")) {
         Blockfound = 1;
         if(cmp64(Cblocknum, Bcbnum) == 0) {
            /* We solved a block! */
            if(update("mblock.dat", 1) == VEOK)
               send_found();  /* start send_found() child */
         }
         unlink("mblock.dat");
         Blockfound = 0;
      }

      /*
       * Time to call Block Constructor?
       */
      if(Txcount >= TXQUEBIG)
         bctime = Ltime;
      if(Bcpid == 0 && Blockfound == 0
         && Ltime >= bctime
         && (Txcount > 0 || (Mpid == 0 && existsnz("txclean.dat")))) {
         /* append txq1.dat to txclean.dat */
         system("cat txq1.dat >>txclean.dat 2>/dev/null");
         unlink("txq1.dat");
         stop_miner();  /* pause miner during block construction */
         if(Trace)
            plog("spawning bcon with %d more transactions", Txcount);
         Txcount = 0;  /* txq1.dat is empty now */
         put64(Bcbnum, Cblocknum);  /* save current block number */
         write_global();
         Bcpid = fork();
         if(Bcpid == 0) {
            /* in child */
            execl("../bcon", "bcon", "txclean.dat", "cblock.dat", NULL);
            error("server(): Cannot execl('bcon',...)");
            exit(1);  /* error but not pink */
         }
         if(Bcpid == -1) { error("Cannot fork() bcon");  Bcpid = 0; }
         bctime = Ltime + BCONFREQ;
      }

      /* Collect bcon status when she is 'done'.  pid == 0 means she
       * is still busy.
       */
      if(Bcpid > 0) {
         pid = waitpid(Bcpid, &status, WNOHANG);
         if(pid > 0) {
            Bcpid = 0;  /* pid not zero means she is done. */
            if(!Nominer) {
               if(!Bgflag) printf("Solving...\n");
               start_miner();  /* start or re-start miner */
            }
         }
      }
      /* bcon sequence will wait on miner if Txcount > 0,
       * else...
       */
      if(Mpid && Ltime >= mwtime) {
         pid = waitpid(Mpid, &status, WNOHANG);
         if(pid > 0) Mpid = 0;  /* Miner exited. */
         mwtime = Ltime + 120;
      }

      /* Start mirror()? */
      if(Ltime >= mqtime && Mqcount > 0 && Mqpid == 0) {
         /* get exclusive access to txq1.dat */
         lfd = lock("mq.lck", 10);
         if(lfd != -1) {
            unlink("mirror.dat");
            rename("mq.dat", "mirror.dat");
            Mqcount = 0;
            unlock(lfd);
            Mqpid = mirror();  /* start child */
         }
      }
      if(Mqpid) {
         pid = waitpid(Mqpid, NULL, WNOHANG);
         if(pid > 0) {
            Mqpid = 0;
            mqtime = Ltime + 2;
         }
      }

      if(TIMES_OF_TROUBLE()) {
         if(bridge() != VEOK || update("pblock.dat", 2) != VEOK) {
            restart("Cannot make pseudo-block");
         }
      }

      /*
       * Display system statistics
       */
      if(Ltime >= Stime) {
         if(read_data(&hps, sizeof(hps), "hps.dat") == sizeof(hps))
            Hps = hps;
         if(Betabait && Bgflag == 0) betabait();
         Stime = Ltime + STATUSFREQ;
      }
      /*
       * Monitor interrupt on Ctrl-C if not in background
       */
      if(Monitor && !Bgflag) monitor();

      if(Watchdog && (Ltime - Utime) >= Watchdog) {
         if(Cblocknum[0] != 0xfe || (Ltime - Utime) >= bigwait) {
            restart("watchdog");
         }
      }

      if(Ltime >= ipltime) {
         refresh_ipl();
         ipltime = Ltime + (rand2() % 600) + 10;
      }

      /* dynamic sleep function */
      if(Dynasleep != 0 && Nonline < 1) usleep(Dynasleep);

   } /* end while(Running) */
   /*
    * Clean up server and exit
    */
   closesocket(lsd);  /* close listening socket */
   return 0;          /* main() will finish cleanup */
} /* end server() */
