
#include "mcmd.h"

#ifndef SERVER_THREADS
   #define SERVER_THREADS  ( cpu_cores() )   /* system dependant */

#endif

int server_task_append(LinkedNode *lnp, LinkedList *llp);
void server_task_cleanup(LinkedNode *lnp);
LinkedNode *server_task_copy(SNODE *snp);
LinkedNode *server_task_receive(SOCKET sd, word32 ip);
LinkedNode *server_task_request(word32 ip, word16 opreq, void *bnum);
void server_tasklist_cleanup(LinkedList *llp);

/* exclusive support */
#include "mcmd_task.c"

/* Server Thread Mutex's, Condition, LinkedList, and Idle counter */
static Mutex ThreadLock = MUTEX_INITIALIZER;
static Mutex ThreadSyncLock = MUTEX_INITIALIZER;
static LinkedList Threads, JoinThreads;
static int IdleThreadCount;

/**
 * Start a server thread, to perform work queued by the server.
 * Allocated resources are free'd in server() before it returns.
 * @param thread_fn Thread function to initiate thread with
 * @returns VEOK on success, else VERROR
*/
static int server_start(ThreadRoutine thread_fn)
{
   LinkedNode *lnp;
   int ecode;

   /* malloc space for LinkedNode */
   lnp = malloc(sizeof(LinkedNode));
   if (lnp == NULL) return VERROR;
   /* malloc space for LinkedNode data (ThreadId) */
   lnp->data = malloc(sizeof(ThreadId));
   if (lnp->data == NULL) goto FAIL_LNP;
   /* create thread in ThreadNode */
   ecode = thread_create(lnp->data, thread_fn, lnp);
   if (ecode) goto FAIL_DATA;

   return VEOK;

FAIL_DATA: free(lnp->data);
FAIL_LNP: free(lnp);
   return VERROR;
}  /* end server_start() */

/**
 * Append a server task LinkedNode to a LinkedList.
 * Uses appropriate locks for "ActiveIO" and "InactiveIO" lists.
 * Signals "ActiveIOAlarm" if appending to "ActiveIO" list.
 * NOTE: sets "ServerOk" global to Zero on error
 * @param lnp Pointer to LinkedNode to append
 * @param llp Pointer to LinkedList to append to
 * @returns VEOK on success, else error number
*/
int server_task_append(LinkedNode *lnp, LinkedList *llp)
{
   int ecode;

#undef FnMSG
#define FnMSG(x)  "server_task_append(): " x

   /* check for special lists */
   if (llp == &ActiveIO) {
      /* ActiveIO list requires ActiveIOLock */
      lock_on_ecode_goto_perrno( ActiveIOLock, FATAL, {
         on_ecode_goto_perrno( link_node_append(lnp, &ActiveIO),
            FATAL, FnMSG("link_node_append(ActiveIO) FAILURE"));
         /* Thread signal required */
         condition_signal(&ActiveIOAlarm);
      });
   } else if (llp == &InactiveIO) {
      /* InactiveIO list requires InactiveIOLock */
      lock_on_ecode_goto_perrno( InactiveIOLock, FATAL, {
         on_ecode_goto_perrno( link_node_append(lnp, &InactiveIO),
            FATAL, FnMSG("link_node_append(InactiveIO) FAILURE"));
      });
   } else {
      /* no identifiable lock required */
      on_ecode_goto_perrno( link_node_append(lnp, llp),
         FATAL, FnMSG("link_node_append(llp) FAILURE"));
   }

   return VEOK;

FATAL:
   ServerOk = 0;
   return VERROR;
}  /* end server_task_append() */

/**
 * Cleanup task releasing held resources back to the system.
 * @param lnp Pointer to LinkedNode containing SNODE
*/
void server_task_cleanup(LinkedNode *lnp)
{
   /* perform cleanup of any internal resources */
   cleanup_node((SNODE *) lnp->data);
   /* deallocate task data and task */
   free(lnp->data);
   free(lnp);
}  /* end server_task_cleanup() */

/**
 * Create and prepare a SNODE task, as a copy of another SNODE.
 * @param snp Pointer to a reference SNODE
 * @return Pointer to LinkedNode on success, or NULL on error.
*/
LinkedNode *server_task_copy(SNODE *snp)
{
   LinkedNode *lnp;

#undef FnMSG
#define FnMSG(x)  "server_task_copy(): " x

   /* create LinkedNode with SNODE data, and copy reference */
   lnp = link_node_create(sizeof(SNODE));
   if (lnp == NULL) perrno(errno, FnMSG("link_node_create() FAILURE"));
   else memcpy(lnp->data, snp, sizeof(SNODE));

   return lnp;
}  /* end server_task_copy() */

/**
 * Create and prepare a SNODE task, to receive connections.
 * @param sd Connection socket of task
 * @param ip Connection ip of task
 * @return Pointer to LinkedNode task on success, or NULL on error.
*/
LinkedNode *server_task_receive(SOCKET sd, word32 ip)
{
   LinkedNode *lnp;

#undef FnMSG
#define FnMSG(x)  "server_task_receive(): " x

   /* create LinkedNode with SNODE data */
   lnp = link_node_create(sizeof(SNODE));
   if (lnp == NULL) perrno(errno, FnMSG("link_node_create() FAILURE"));
   else init_receive((SNODE *) lnp->data, sd, ip);

   return lnp;
}  /* end server_task_receive() */

/**
 * Create and prepare a SNODE task, specifically to serve a request.
 * @param ip Connection ip of task
 * @param opreq Request operation code
 * @param bnum Request IO value (blocknum), or NULL
 * @return Pointer to LinkedNode task on success, or NULL on error.
*/
LinkedNode *server_task_request(word32 ip, word16 opreq, void *bnum)
{
   LinkedNode *lnp;

#undef FnMSG
#define FnMSG(x)  "server_task_request(): " x

   /* create LinkedNode with SNODE data */
   lnp = link_node_create(sizeof(SNODE));
   if (lnp == NULL) perrno(errno, FnMSG("link_request_node() FAILURE"));
   else init_request((SNODE *) lnp->data, ip, Dstport_opt, opreq, bnum);

   return lnp;
}  /* end server_task_request() */

/**
 * Cleanup all tasks of a list releasing held resources back to the system.
 * @param lnp Pointer to LinkedList of SNODE tasks
*/
void server_tasklist_cleanup(LinkedList *llp)
{
   LinkedNode *lnp;
   int ecode;

#undef FnMSG
#define FnMSG(x)  "server_tasklist_cleanup(): " x

   while ((lnp = llp->next)) {
      on_ecode_goto_perrno( link_node_remove(lnp, llp),
         FATAL, FnMSG("link_node_remove() FAILURE"));
      server_task_cleanup(lnp);
   }

FATAL:
   ServerOk = 0;
}  /* end server_tasklist_cleanup() */

/**
 * @private
 * Dynamic handling of work queued by the server.
 * All tasks pulled from ActiveIO undergo asynchronous processing.
 * After asynchronous processing returns:
 * - unfinished tasks are placed in InactiveIO for wait handling
 * - finished "request" tasks are placed in SyncIO for sync handling
 * - finished "receive" tasks are (deallocated and) discarded
 * All tasks pulled from SyncIO undergo synchronous processing.
 * After synchronous processing returns:
 * - unfinished tasks are placed in InactiveIO for wait handling
 * - all other tasks are (deallocated and) discarded
 * All threads start as asynchronous threads. If a task is deemed to
 * require synchronous processing, a thread will promote itself to a
 * synchronous thread to process synchronous tasks. If a synchronous
 * thread already exists, the task will be passed to SyncIO and will
 * eventually be processed by the current synchronous thread.
*/
static ThreadProc server_thread(void *lnp)
{
   LinkedNode *thrd_lnp;   /* pointer to LinkedNode of this thread */
   LinkedNode *task_lnp;   /* pointer to LinkedNode of next task */
   SNODE *snp;        /* pointer to SNODE within task */
   int ecode;

   /* init */
   thrd_lnp = (LinkedNode *) lnp;
   /* set name of thread - visible in htop */
   thread_setname(*((ThreadId *) thrd_lnp->data), "server_thread");

#undef FnMSG
#define FnMSG(x)  "server_thread(%x): " x, *((int *) thrd_lnp->data)
   pdebug(FnMSG("created..."));

   /* add self to Active thread list and acquire ActiveIOLock */
   lock_on_ecode_goto_perrno( ThreadLock, FATAL, {
      on_ecode_goto_perrno( link_node_append(thrd_lnp, &Threads),
         FATAL, FnMSG("Threads LIST FAILURE"));
   });
   on_ecode_goto_perrno( mutex_lock(&ActiveIOLock),
      FATAL, FnMSG("ActiveIO (initial) LOCK FAILURE"));

   /* main thread loop -- prioritize SyncIO tasks */
   while (ServerOk) {
      /* check/pull next ActiveIO task */
      while (ServerOk && (task_lnp = ActiveIO.next)) {
         /* remove task node from list */
         on_ecode_goto_perrno( link_node_remove(task_lnp, &ActiveIO),
            FATAL, FnMSG("ActiveIO LIST FAILURE"));
         /* release lock on ActiveIO */
         on_ecode_goto_perrno( mutex_unlock(&ActiveIOLock),
            FATAL, FnMSG("ActiveIO UNLOCK FAILURE"));
         /* dereference and process SNODE asynchronously */
         snp = (SNODE *) task_lnp->data;
         if (server_async(snp) == VEWAITING) {
            /* return task to InactiveIO for waiting */
            lock_on_ecode_goto_perrno( InactiveIOLock, FATAL, {
               on_ecode_goto_perrno(
                  link_node_append(task_lnp, &InactiveIO),
                  FATAL, FnMSG("InactiveIO LIST FAILURE"));
            });
         } else if (snp->opreq) {
            /* move and execute synchronous tasks, if not already */
            lock_on_ecode_goto_perrno( SyncIOLock, FATAL, {
               /* place task in SyncIO */
               on_ecode_goto_perrno(
                  link_node_append(task_lnp, &SyncIO),
                  FATAL, FnMSG("SyncIO LIST FAILURE"));
               /* try acquire ThreadSyncLock for task synchronization */
               trylock_on_ecode_goto_perrno( ThreadSyncLock, FATAL, {
                  while (ServerOk && (task_lnp = SyncIO.next)) {
                     /* remove task node from list */
                     on_ecode_goto_perrno(
                        link_node_remove(task_lnp, &SyncIO),
                        FATAL, FnMSG("SyncIO LIST FAILURE"));
                     /* release lock on SyncIO */
                     on_ecode_goto_perrno( mutex_unlock(&SyncIOLock),
                        FATAL, FnMSG("SyncIO UNLOCK FAILURE"));
                     /* dereference and process SNODE synchronously */
                     snp = (SNODE *) task_lnp->data;
                     if (server_sync(snp) == VEWAITING) {
                        /* return task to ActiveIO */
                        lock_on_ecode_goto_perrno(
                           ActiveIOLock, FATAL, {
                              on_ecode_goto_perrno(
                                 link_node_append(task_lnp, &ActiveIO),
                                 FATAL, FnMSG("ActiveIO LIST FAILURE"));
                              condition_signal(&ActiveIOAlarm);
                           }
                        );
                     } else server_task_cleanup(task_lnp);
                     /* (re)acquire SyncIO Lock */
                     on_ecode_goto_perrno( mutex_lock(&SyncIOLock),
                        FATAL, FnMSG("SyncIO re-LOCK FAILURE"));
                  }  /* end while ((task_lnp = SyncIO.next)) */
               });  /* end trylock...(ThreadSyncLock... */
            });  /* end lock...(SyncIOLock... */
         } else server_task_cleanup(task_lnp);
         /* (re)acquire ActiveIO Lock */
         on_ecode_goto_perrno( mutex_lock(&ActiveIOLock),
            FATAL, FnMSG("ActiveIO (pre condition) LOCK FAILURE"));
      }  /* end while ((task_lnp = ActiveIO.next)) */
      if (!ServerOk) break;
      /* wait for condition, sleepy time ... */
      IdleThreadCount++;
      ecode = condition_wait(&ActiveIOAlarm, &ActiveIOLock);
      IdleThreadCount--;
      /* ... wakeup (spurious?), check ecode ... */
      if (ecode) {
         perrno(ecode, FnMSG("ActiveIO CONDITION FAILURE"));
         goto FATAL;
      }
   }  /* end while (ServerOk) */
   pdebug(FnMSG("recv'd shutdown signal"));

   /* release held lock on ActiveIO */
   on_ecode_goto_perrno( mutex_unlock(&ActiveIOLock),
      FATAL, FnMSG("ActiveIO (shutdown) UNLOCK FAILURE"));

   /* move thread node to Join list */
   on_ecode_goto_perrno( link_node_remove(lnp, &Threads),
      FATAL, FnMSG("link_node_remove(Threads) FAILURE"));
   on_ecode_goto_perrno( link_node_append(lnp, &JoinThreads),
      FATAL, FnMSG("link_node_append(JoiinThreads) FAILURE"));
   /* send signal to ActiveIOAlarm before exit */
   condition_signal(&ActiveIOAlarm);
   /* exit thread */
   Unthread;

FATAL:
   /* kill server on fatal thread error */
   ServerOk = 0;
   /* exit thread */
   Unthread;
}  /* end server_thread() */

/**
 * The Mochimo Server. Listens for connections on port 2095.
 * Queues/Monitors connections and IO buffers for scheduling work.
*/
int server(word16 server_port, word16 api_port)
{
   static const int enable = 1;

   static LinkedList ReadyIO, WaitIO;
   static fd_set rfds, wfds;        /* descriptor sets for IO checks */
   static struct sockaddr_in addr;  /* internet socket address struct */
   static struct timeval tv;        /* sleep time during IO checks */
   static long dynasleep;           /* dynamic sleep for IO checks */
   static SOCKET lsd, nsd;          /* listen/next sockets */
   static word32 nip;               /* next socket ip address */
   static time_t Ltime;             /* server base time */
   // static time_t Ntime;          /* server network scan time */
   static time_t Utime;             /* server up-time */

   /* additional vars for IO checks */
   LinkedNode *lnp, *lnp_next;
   SNODE *snp;
   SOCKET nfds;
   int ecode, count, i;
   char ipstr[15];

#undef FnMSG
#define FnMSG(x)  "server(%" P16u ", %" P16u "): " x, server_port, api_port

   (void)api_port;

   /* init */
   ServerOk = 1;
   dynasleep = i = 0;
   nsd = INVALID_SOCKET;
   memset(&addr, 0, sizeof(addr));
   /* obtain listening socket */
   lsd = socket(AF_INET, SOCK_STREAM, 0);
   if (lsd == INVALID_SOCKET) {
      perrno(errno, FnMSG("cannot open socket for listening"));
      goto SHUTDOWN;
   }
   /* enable socket's "reuse address" flag */
   setsockopt(lsd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
   /* prepare address structure for binding */
   addr.sin_addr.s_addr = INADDR_ANY;
   addr.sin_family = AF_INET;
   addr.sin_port = htons(server_port);

   /* Running check */
   while (Running) {
      /* bind address with listening socket */
      if (bind(lsd, (struct sockaddr *) &addr, sizeof(addr)) == 0) {
         /* start (at least 1) server threads to handle queued work */
         for (i = 0; i == 0 || i < SERVER_THREADS; i++) {
            on_ecode_goto_perrno( server_start(server_thread),
               SHUTDOWN, FnMSG("server_start() FAILURE"));
         }
         /* set listen() port non-blocking */
         on_ecode_goto_perrno( sock_set_nonblock(lsd) == SOCKET_ERROR,
            SHUTDOWN, FnMSG("sock_set_nonblock(lsd) FAILURE"));
         /* start "listening..." */
         on_ecode_goto_perrno( listen(lsd, LQLEN),
            SHUTDOWN, FnMSG("listen(lsd, " makeSTR(LQLEN) ") FAILURE"));
         plog("Listening...\n");
         /* signal Verisimility that we are up. */
         remove("vstart.lck");
         /* set Uptime */
         time(&Utime);
         /* start server initialization */
         server_init(NULL);
         /* done */
         break;
      }  /* end if (bind ... */
      /* failed to bind(), wait and retry */
      plog(FnMSG("trying to bind port %" P16u "..."), server_port);
      /* use select() as cross-platform capable sleep */
      tv.tv_sec = 0;
      tv.tv_usec = 5000000L;
      select(0, NULL, NULL, NULL, &tv);
   }  /* end while (Running) */

   /* Running check -- main server listening loop */
   while (Running && ServerOk) {

      /* set time for this loop */
      time(&Ltime);

      /* accept() up to LQLEN connections per iteration */
      for (i = 0; i < LQLEN; i++) {
         /* accept() new connection -- set non-blocking */
         nsd = accept(lsd, NULL, NULL);
         if (nsd == INVALID_SOCKET) break;
         /* drop pinklisted ip */
         nip = get_sock_ip(nsd);
         if (pinklisted(nip)) {
            pfine(FnMSG("dropped %s (pink)"), ntoa(&nip, ipstr));
            sock_close(nsd);
            continue;
         }
         /* dynamic sleep reset */
         if (dynasleep) dynasleep = 0;
         /* set socket non-blocking */
         on_ecode_goto_perrno( sock_set_nonblock(nsd),
            SHUTDOWN, FnMSG("sock_set_nonblock(nsd) FAILURE"));
         /* create recv task with nsd/nip data -- append to WaitIO */
         lnp = server_task_receive(nsd, nip);
         if (lnp == NULL || server_task_append(lnp, &WaitIO) != VEOK) {
            pfine(FnMSG("dropped %s (err)"), ntoa(&nip, ipstr));
            sock_close(nsd);
            continue;
         }
      }  /* end for (i = 0; i < SERVER_LISTEN_LIMIT ... */

      /* try (NON-BLOCKING) link InactiveIO to WaitIO */
      trylock_on_ecode_goto_perrno( InactiveIOLock, SHUTDOWN, {
         on_ecode_goto_perrno( link_list_append(&InactiveIO, &WaitIO),
            SHUTDOWN, FnMSG("link_list(InactiveIO, WaitIO) FAILURE"));
      });

      /* zero fd sets and reset highest descriptor */
      FD_ZERO(&rfds);
      FD_ZERO(&wfds);
      nfds = INVALID_SOCKET;
      /* add "waiting" sockets to appropriate fd_set's -- adjust nfds */
      for (lnp = WaitIO.next; lnp; lnp = lnp_next) {
         lnp_next = lnp->next;
         /* dereference the SNODE pointer */
         snp = (SNODE *) lnp->data;
         /* check wait type -- update fd_sets accordingly */
         switch (snp->iowait) {
            case IO_CONN: /* fallthrough -- same as IO_SEND */
            case IO_SEND: FD_SET(snp->sd, &wfds); break;
            case IO_RECV: FD_SET(snp->sd, &rfds); break;
            default: {
               /* remove stray finished tasks from WaitIO */
               on_ecode_goto_perrno( link_node_remove(lnp, &WaitIO),
                  SHUTDOWN, FnMSG("link_node_remove(WaitIO) FAILURE"));
               /* deallocate task resources */
               server_task_cleanup(lnp);
               continue;
            }
         }  /* end switch (snp->iowait) */
         /* update nfds value if necessary */
         if (nfds < snp->sd) nfds = snp->sd;
      }  /* end for (lnp = WaitIO... */

      /* set timeout as preferred sleep duration for this iteration */
      tv.tv_sec = 0;
      tv.tv_usec = 1000L * dynasleep++;
      /* check number of "waiting" sockets ready for send/recv/fail */
      count = select((int) (nfds + 1), &rfds, &wfds, NULL, &tv);
      /* IMPORTANT: tv's value at this point SHOULD NOT be re-used */

      /* while remaining (count), walk WaitIO -- hide 'n' seek */
      for (lnp = WaitIO.next; lnp; lnp = lnp_next) {
         /* store next link, as links may change */
         lnp_next = lnp->next;
         /* dereference the SNODE pointer */
         snp = (SNODE *) lnp->data;
         /* check send/recv fd_set's and timeout */
         /* NOTE: "<= IO_SEND" includes IO_CONN in conditional check */
         if ((snp->iowait <= IO_SEND && FD_ISSET(snp->sd, &wfds)) ||
            (snp->iowait == IO_RECV && FD_ISSET(snp->sd, &rfds)) ||
            (snp->to && difftime(Ltime, snp->to) > 0)) {
            /* move node from "wait" list to "ready" list */
            on_ecode_goto_perrno( link_node_remove(lnp, &WaitIO),
               SHUTDOWN, FnMSG("link_node_remove(WaitIO) FAILURE"));
            on_ecode_goto_perrno( link_node_append(lnp, &ReadyIO),
               SHUTDOWN, FnMSG("link_node_append(ReadyIO) FAILURE"));
         }  /* end (in)activity checks... */
      }  /* end for (taskp = WaitIO... */

      /* trigger OP_FOUND broadcast after ~2min of inactivity */
      if (dynasleep > 500) {
         dynasleep = 0;
         rdlock_on_ecode_goto_perrno( RplistLock, SHUTDOWN, {
            for (i = 0; i < RPLISTLEN && Rplist[i]; i++) {
               lnp = server_task_request(Rplist[i], OP_FOUND, NULL);
               if (lnp) server_task_append(lnp, &ReadyIO);
               else {
                  perr(FnMSG("OP_FOUND broadcast failed, %d"), i);
                  break;
               }
            }  /* end for... */
         });  /* end rdlock_on_ecode_goto_perrno() */
      }  /* end if (dynasleep > 500) */

      /* check ReadyIO list */
      if (ReadyIO.count) {
         /* dynamic sleep reset */
         if (dynasleep) dynasleep = 0;
         /* store ReadyIO list count */
         count = ReadyIO.count;
         /* try (NON-BLOCKING) acquire ActiveIO lock for link list */
         trylock_on_ecode_goto_perrno( ActiveIOLock, SHUTDOWN, {
            /* link ReadyIO to ActiveIO */
            on_ecode_goto_perrno( link_list_append(&ReadyIO, &ActiveIO),
               SHUTDOWN, FnMSG("link_list(ReadyIO, ActiveIO) FAILURE"));
            /* signal idle threads indicating available work */
            if (count < IdleThreadCount) {
               while (count--) condition_signal(&ActiveIOAlarm);
            } else condition_broadcast(&ActiveIOAlarm);
         });
      }  /* end if (ReadyIO.count) */
   }  /* end while (Running) */
   /* server ended gracefully -- jump to SHUTDOWN */
   pdebug(FnMSG("recv'd shutdown signal"));

SHUTDOWN:
   plog("Server exiting, please wait...");
   /* try flagging graceful shutdown */
   ServerOk = 0;
   /* close listening socket */
   if (lsd != INVALID_SOCKET) sock_close(lsd);
   /* wait for lock on Threads */
   on_ecode_goto_perrno( mutex_lock(&ThreadLock),
      FATAL, "mutex_lock(ThreadLock) FAILURE");
   /* alert threads of state change */
   condition_broadcast(&ActiveIOAlarm);
   /* wait for threads to exit */
   while (Threads.count) {
      /* wait for threads to finish, up to 5 seconds */
      if (condition_timedwait(&ActiveIOAlarm, &ThreadLock, 5000)) {
         plog("Taking too long, terminating...");
         break;
      }
      /* join with any threads added to the Join list */
      while ((lnp = JoinThreads.next)) {
         on_ecode_goto_perrno( link_node_remove(lnp, &JoinThreads),
            FATAL, FnMSG("link_node_remove() FAILURE"));
         on_ecode_goto_perrno( thread_join(*((ThreadId *) lnp->data)),
            FATAL, FnMSG("thread_join() FAILURE"));
         /* free LinkedNode and associated data */
         free(lnp->data);
         free(lnp);
      }
   }  /* end while (Threads.count) */
   /* release thread lock when finished */
   on_ecode_goto_perrno( mutex_unlock(&ThreadLock),
      FATAL, "mutex_unlock(ThreadLock) FAILURE");

FATAL:

   /* done */
   return ecode;
} /* end server() */
