
#include "server.h"

/* internal support */
#include "error.h"

/* external support */
#include "extinet.h"
#include "extlib.h"
#include <string.h>

/* standard server connection TIMEOUT, 30 seconds */
#define SERVER_TIMEOUT  30

#define ACCEPT_DROP(sd, x) \
   do { \
      pcustom(errno, PLEVEL_DEBUG, FnMSG("drop sd= %d (" x ")"), sd); \
      sock_close(sd); continue; \
   } while(0)

/* MACRO for handling excessive connections on various systems
 * NOTE: unnecessary when the server handler moves away from select() */
#ifdef _WIN32
   #define SELECT_CANNOT_HANDLE(sd) ( 0 )

#else
   #define SELECT_CANNOT_HANDLE(sd) ( sd >= FD_SETSIZE )

#endif

/**
 * @private for internal use only
*/
static LinkedNode *server__accept_work(LinkedNode *lnp, SOCKET sd, int sio)
{
   AsyncWork *wp;

   /* create (or reuse supplied) LinkedNode with socket */
   if (lnp == NULL) lnp = link_node_create(sizeof(AsyncWork));
   if (lnp == NULL) return NULL;
   wp = (AsyncWork *) lnp->data;
   wp->to = SERVER_TIMEOUT;
   wp->sio = sio;
   wp->sd = sd;

   return lnp;
}  /* end server__accept_work() */

/**
 * Cleanup server work releasing held resources back to the system.
 * @param sp Pointer to Server
 * @param wp Pointer to AsyncWork
 * @private for internal use only
*/
static void server__cleanup_work(Server *sp, AsyncWork *wp)
{
   if (wp == NULL) return;
   /* check for and execute additional cleanup process */
   if (sp && sp->donefn) sp->donefn(wp);
   /* cleanup and deallocate work resources */
   if (wp->sd != INVALID_SOCKET) sock_close(wp->sd);
   if (wp->data != NULL) free(wp->data);
   free(wp);
}  /* end server__cleanup_work() */

/**
 * Cleanup LinkedNode releasing held resources back to the system.
 * @param sp Pointer to Server
 * @param lnp Pointer to LinkedNode
 * @private for internal use only
*/
static void server__cleanup_node(Server *sp, LinkedNode *lnp)
{
   if (lnp == NULL) return;
   /* cleanup and deallocate node resources */
   server__cleanup_work(sp, lnp->data);
   free(lnp);
}  /* end server__cleanup_node() */

/**
 * Cleanup Server releasing held resources back to the system.
 * @param sp Pointer to Server context
 * @private for internal use only
*/
static void server__cleanup(Server *sp)
{
   LinkedNode *lnp;

   /* ensure socket is closed */
   if (sp->lsd != INVALID_SOCKET) {
      sock_close(sp->lsd);
      sp->lsd = INVALID_SOCKET;
   }
   while ((lnp = sp->inIO.next)) {
      link_node_remove(lnp, &(sp->inIO));
      server__cleanup_node(sp, lnp);
   }
   while ((lnp = sp->outIO.next)) {
      link_node_remove(lnp, &(sp->outIO));
      server__cleanup_node(sp, lnp);
   }
}  /* end server__cleanup() */

/**
 * @private for internal use only
*/
static int server__create_work(Server *sp, void *data)
{
   LinkedNode *lnp;
   AsyncWork *wp;

   /* create LinkedNode with empty AsyncWork */
   lnp = server__accept_work(NULL, INVALID_SOCKET, IO_CONN);
   wp = lnp->data;
   /* pass data pointer if supplied */
   if (data) wp->data = data;
   /* place in "ready" list for server worker processing */
   if (mutex_lock(&(sp->lock)) != 0) goto FAIL;
   if (link_node_append(lnp, &(sp->inIO)) != 0) goto FAIL;
   condition_signal(&(sp->alarm));
   if (mutex_unlock(&(sp->lock)) != 0) goto FATAL;

   return VEOK;

FAIL:
   /* release lock and return error */
   if (mutex_unlock(&(sp->lock)) == 0) return VERROR;

FATAL:
   /* kill server on fatal error */
   sp->running = 0;
   return VERROR;
}  /* end server__accept_work() */

/**
 * Obtain the deferred work threshold. Allows a server with multiple
 * workers to avoid being bound by a flood of operations requiring
 * a certain type of processing (as determined by the developer).
 * @param sp Pointer to server context
 * @return Allowable concurrent threads processing deferred work
 * @private for internal use only
*/
static int server__defer_threshold(Server *sp)
{
   return (sp->numthreads + 2) / 3;
}

/**
 * @private for internal use only
*/
static ThreadProc server__exit(Server *sp)
{
   LinkedNode *lnp;
   Thread self;

   /* get current thread */
   self = thread_self();

   /* acquire server lock */
   if (mutex_lock(&(sp->lock)) != 0) goto FATAL;
   /* search for LinkedNode holding current thread */
   for (lnp = sp->active.next; ; lnp = lnp->next) {
      if (thread_equal(self, *((Thread *) lnp->data))) break;
      if (lnp->next == NULL) goto FAIL;
   }
   /* move thread node to exited list -- signal for join */
   if (link_node_remove(lnp, &(sp->active))) goto FAIL;
   if (link_node_append(lnp, &(sp->exited))) goto FAIL;
   condition_signal(&(sp->alarm));

FAIL:
   /* ensure acquired lock is released -- exit */
   if (mutex_unlock(&(sp->lock)) != 0) goto FATAL;
   Unthread;

FATAL:
   /* ensure server running is (un)flagged on fatal error */
   sp->running = 0;
   Unthread;
}  /* end server__exit() */

static ThreadProc server__main(void *arg)
{
   Server *sp = arg;

   fd_set rfds, wfds;            /* descriptor sets for IO checks */
   LinkedList readyIO = { 0 };   /* lock-free work list ready for IO */
   LinkedList waitIO = { 0 };    /* lock-free work list waiting for IO */
   LinkedNode *lnp, *next_lnp;
   LinkedNode *alnp;
   AsyncWork *wp;
   struct sockaddr *baddrp;
   struct timeval tv;            /* sleep time during IO checks */
   long dynasleep;               /* dynamic sleep for IO checks */
   time_t Ltime;
   SOCKET asd;                   /* next socket descriptor */
   SOCKET nfds;                  /* for select(nfds, ...) */
   int ecode, count, i;
   char ipstr[16];
   char name[16];

#undef FnMSG
#define FnMSG(x)  "server__main(%s:%u): " x, \
   ntoa(&(sp->addr.sin_addr.s_addr), ipstr), sp->addr.sin_port

   /* init */
   baddrp = (struct sockaddr *) &(sp->addr);
   dynasleep = 0;
   alnp = NULL;
   i = 0;

   /*********************/
   /* SERVER BIND PHASE */
   snprintf(name, sizeof(name), "%s-binding", sp->name);
   thread_setname(thread_self(), name);

   /* (try) bind address with listening socket */
   while (bind(sp->lsd, baddrp, sizeof(sp->addr))) {
      /* check shutdown signal -- wait a sec before re-attempt... */
      if (sp->running == 0) goto SHUTDOWN;
      millisleep(1000);
   }  /* end while (bind(sp->lsd... */
   /* set listen() port non-blocking and start listening */
   if (sock_set_nonblock(sp->lsd) != 0) goto SHUTDOWN;
   if (listen(sp->lsd, sp->backlog) != 0) goto SHUTDOWN;

   /*******************/
   /* SERVER IO PHASE */
   snprintf(name, sizeof(name), "%s-listening", sp->name);
   thread_setname(thread_self(), name);

   /* running check */
   while (sp->running) {

      /* iterate backlog'd connections */
      for (i = 0; i < sp->backlog; i++) {
         /* accept() new connections -- get socket ip */
         asd = accept(sp->lsd, NULL, NULL);
         if (asd == INVALID_SOCKET) break;
         /* check socket can be handled */
         if (SELECT_CANNOT_HANDLE(asd)) ACCEPT_DROP(asd, "sd");
         /* accept work with asd and aip data */
         alnp = server__accept_work(alnp, asd, IO_RECV);
         if (alnp == NULL) ACCEPT_DROP(asd, "err");
         /* call custom init function on accept()'d work */
         if (sp->initfn && sp->initfn(alnp->data)) ACCEPT_DROP(asd, "init");
         /* add work to wait list */
         on_ecode_goto_perrno( link_node_append(alnp, &waitIO),
            SHUTDOWN, FnMSG("link_node_append(alnp, waitIO) FAILURE"));
         /* dynamic sleep reset -- reset alnp */
         if (dynasleep) dynasleep = 0;
         alnp = NULL;
      }  /* end for (i = 0; i < SERVER_LISTEN_LIMIT ... */

      /* update Ltime for timeout check */
      time(&Ltime);

      /* zero fd sets and reset highest descriptor */
      FD_ZERO(&rfds);
      FD_ZERO(&wfds);
      nfds = INVALID_SOCKET;
      /* prepare fd_set's only if waiting for IO */
      if (waitIO.count) {
         count = FD_SETSIZE;
         /* add "waiting" sockets to appropriate fd_set's -- adjust nsd */
         for (lnp = waitIO.next; count > 0 && lnp; lnp = next_lnp) {
            next_lnp = lnp->next;
            /* dereference the server work */
            wp = (AsyncWork *) lnp->data;
            /* cleanup any stray work with an invalid socket descriptor */
            if (wp->sd == INVALID_SOCKET) {
               on_ecode_goto_perrno( link_node_remove(lnp, &waitIO),
                  SHUTDOWN, FnMSG("link_node_remove(waitIO) FAILURE"));
               server__cleanup_node(sp, lnp);
               continue;
            } else if (wp->to && difftime(Ltime, wp->to) > 0) {
               /* move node from "wait" list to "ready" list */
               on_ecode_goto_perrno( link_node_remove(lnp, &waitIO),
                  SHUTDOWN, FnMSG("link_node_remove(waitIO) FAILURE"));
               /* append work to the appropriate list */
               on_ecode_goto_perrno( link_node_append(lnp, &readyIO),
                  SHUTDOWN, FnMSG("link_node_append(readyIO) FAILURE"));
               continue;
            }
            /* check wait type -- update fd_sets accordingly */
            switch (wp->sio) {
               case IO_CONN: /* fallthrough -- same as IO_SEND */
               case IO_SEND: FD_SET(wp->sd, &wfds); break;
               case IO_RECV: FD_SET(wp->sd, &rfds); break;
               default: continue;
            }  /* end switch (np->sio) */
            /* update nfds value if necessary */
            if (nfds < wp->sd) nfds = wp->sd;
            /* decrement count */
            count--;
         }  /* end for (lnp = waitIO... */
      }  /* end for if (waitIO.count) */
      /* set timeout as preferred sleep duration for this iteration */
      tv.tv_sec = 0;
      tv.tv_usec = 1000L * dynasleep;
      if (dynasleep < 500) dynasleep++;
      /* check number of "waiting" sockets ready for send/recv/fail */
      count = select((int) (nfds + 1), &rfds, &wfds, NULL, &tv);
      /* IMPORTANT: tv's value at this point SHOULD NOT be re-used */

      /* walk waitIO with respect to count -- hide 'n' seek */
      for (lnp = waitIO.next; count > 0 && lnp; lnp = next_lnp) {
         /* store next link, as links may change */
         next_lnp = lnp->next;
         /* dereference the work */
         wp = (AsyncWork *) lnp->data;
         /* check send/recv fd_set's and timeout */
         /* NOTE: "<= IO_SEND" includes IO_CONN in conditional check */
         if ((wp->sio <= IO_SEND && FD_ISSET(wp->sd, &wfds)) ||
            (wp->sio == IO_RECV && FD_ISSET(wp->sd, &rfds)) ) {
            /* move node from "wait" list to "ready" list */
            on_ecode_goto_perrno( link_node_remove(lnp, &waitIO),
               SHUTDOWN, FnMSG("link_node_remove(waitIO) FAILURE"));
            /* append work to the appropriate list */
            on_ecode_goto_perrno( link_node_append(lnp, &readyIO),
               SHUTDOWN, FnMSG("link_node_append(readyIO) FAILURE"));
            count--;
         }  /* end (in)activity checks... */
      }  /* end for (lnp = waitIO... */

      /* check "ready" lists */
      if (readyIO.count) {
         /* dynamic sleep reset */
         if (dynasleep) dynasleep = 0;
         /* store list counts */
         count = readyIO.count;
         /* try (NON-BLOCKING) acquire work lock for link list */
         trylock_on_ecode_goto_perrno( sp->lock, SHUTDOWN, {
            /* link ReadyIO to ActiveIO */
            on_ecode_goto_perrno( link_list_append(&readyIO, &(sp->inIO)),
               SHUTDOWN, FnMSG("link_list(readyIO, sp->inIO) FAILURE"));
            /* signal idle threads indicating available work */
            if (count <= sp->idlethreads) {
               while (count--) condition_signal(&(sp->alarm));
            } else condition_broadcast(&(sp->alarm));
         });
      }  /* end if (readyIO.count) */

      /* try (NON-BLOCKING) link InactiveIO to WaitIO */
      trylock_on_ecode_goto_perrno( sp->lock2, SHUTDOWN, {
         on_ecode_goto_perrno( link_list_append(&(sp->outIO), &waitIO),
            SHUTDOWN, FnMSG("link_list(outIO, waitIO) FAILURE"));
      });
   }  /* end while (sp->running) */
   pdebug(FnMSG("recv'd shutdown signal"));

SHUTDOWN:
   /* close listening socket */
   if (sp->lsd != INVALID_SOCKET) {
      sock_close(sp->lsd);
      sp->lsd = INVALID_SOCKET;
   }
   /* free internally allocated or held resources */
   server__cleanup_node(sp, alnp);
   while ((lnp = readyIO.next)) {
      on_ecode_goto_perrno( link_node_remove(lnp, &readyIO),
         FATAL, FnMSG("link_node_remove(readyIO) SHUTDOWN FAILURE"));
      server__cleanup_node(sp, lnp);
   }
   while ((lnp = waitIO.next)) {
      on_ecode_goto_perrno( link_node_remove(lnp, &waitIO),
         FATAL, FnMSG("link_node_remove(readyIO) SHUTDOWN FAILURE"));
      server__cleanup_node(sp, lnp);
   }
   /* local shutdown triggers global shutdown */
   sp->running = 0;
   /* return result of server__exit() */
   return server__exit(sp);

FATAL:
   pdebug(FnMSG("FATAL ERROR, TERMINATING..."));
   /* kill server on fatal error */
   sp->running = 0;
   Unthread;
}  /* end server__main() */

/**
 * @private for internal use only
 */
static ThreadProc server__worker(void *arg)
{
   Server *sp = arg;

   LinkedNode *lnp;
   AsyncWork *wp;
   int deferproc;    /* indicates the processing of deferred work */
   int ecode;
   char name[16];

#undef FnMSG
#define FnMSG(x)  "server__worker(%x): " x, thread_selfid()
   pdebug(FnMSG("created..."));

   /* init */
   deferproc = 0;

   /* acquire "in" work Lock */
   on_ecode_goto_perrno( mutex_lock(&(sp->lock)),
      FATAL, FnMSG("server lock (init) LOCK FAILURE"));

   /* main thread loop -- prioritize SyncIO tasks */
   while (sp->running) {
      snprintf(name, sizeof(name), "%s-async", sp->name);
      thread_setname(thread_self(), name);
      /* check/pull next work */
      while ((lnp = sp->inIO.next)) {
         /* determine if work needs to be deferred */
         do {
            /* dereference server work */
            wp = (AsyncWork *) lnp->data;
            /* check work deference */
            if (wp->defer == 0) break;
            if (sp->deferthreads < server__defer_threshold(sp)) {
               sp->deferthreads++;
               deferproc = 1;
               break;
            }
         } while ((lnp = lnp->next));
         /* check work was obtained */
         if (lnp == NULL) break;
         /* remove task node from work list */
         on_ecode_goto_perrno( link_node_remove(lnp, &(sp->inIO)),
            FATAL, FnMSG("server inIO LIST FAILURE"));
         /* release work lock */
         on_ecode_goto_perrno( mutex_unlock(&(sp->lock)),
            FATAL, FnMSG("server lock UNLOCK FAILURE"));
         /* dereference and process thread work io */
         if ((sp->procfn && sp->procfn(wp) == VEWAITING)) {
            /* return task to wait list for IO waiting */
            lock_on_ecode_goto_perrno( sp->lock2, FATAL, {
               on_ecode_goto_perrno( link_node_append(lnp, &(sp->outIO)),
                  FATAL, FnMSG("server outIO LIST FAILURE"));
            });
         } else server__cleanup_node(sp, lnp);
         /* (re)acquire "in" work Lock */
         on_ecode_goto_perrno( mutex_lock(&(sp->lock)),
            FATAL, FnMSG("server lock LOCK FAILURE"));
         /* check SHUTDOWN for jump */
         if (sp->running == 0) goto SHUTDOWN;
         /* restore deference parameters */
         if (deferproc) {
            sp->deferthreads--;
            deferproc = 0;
         }
      }  /* end while ((lnp = sp->worklst.next)) */
      snprintf(name, sizeof(name), "%s-idle", sp->name);
      thread_setname(thread_self(), name);
      /* wait for condition, sleepy time ... */
      sp->idlethreads++;
      ecode = condition_wait(&(sp->alarm), &(sp->lock));
      sp->idlethreads--;
      /* ... wakeup (spurious?), check ecode ... */
      if (ecode) {
         perrno(FnMSG("CONDITION FAILURE"));
         goto FATAL;
      }
   }  /* end while (sp->running) */

SHUTDOWN:
   pdebug(FnMSG("recv'd shutdown signal"));

   /* release work list lock */
   on_ecode_goto_perrno( mutex_unlock(&(sp->lock)),
      FATAL, FnMSG("server lock (end) UNLOCK FAILURE"));

   return server__exit(sp);

FATAL:
   pdebug(FnMSG("FATAL ERROR, TERMINATING..."));
   /* kill server on fatal error */
   sp->running = 0;
   Unthread;
}  /* end server__worker() */

/**
 * Destroy a server context. Deallocates server resources as necessary.
 * @param sp Pointer to Server context
 * @return (int) value indicating operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 * @exception errno=EACCES Permission denied. Server is running.
*/
int server_destroy(Server *sp)
{
   /* check for running server */
   if (sp->running) goto FAIL_ACCES;

   /* deallocate server resources */
   server__cleanup(sp);
   /* destroy condition variables */
   condition_destroy(&(sp->alarm2));
   condition_destroy(&(sp->alarm));
   /* destroy mutually exclusive access locks */
   mutex_destroy(&(sp->lock2));
   mutex_destroy(&(sp->lock));

   /* server resources destroyed */
   return VEOK;

/* error handling / de-initialization */
FAIL_ACCES: set_errno(EACCES); return VERROR;
}  /* end server_destroy() */

/**
 * Initialize a server context.
 * @param sp Pointer to Server context
 * @return (int) value indicating operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int server_init(Server *sp, const char *name, int af, int type, int proto)
{
   memset(sp, 0, sizeof(*sp));
   /* initialize mutually exclusive access locks */
   if (mutex_init(&(sp->lock)) != 0) goto FAIL_INIT0;
   if (mutex_init(&(sp->lock2)) != 0) goto FAIL_INIT1;
   /* initialize condition variables */
   if (condition_init(&(sp->alarm)) != 0) goto FAIL_INIT3;
   if (condition_init(&(sp->alarm2)) != 0) goto FAIL_INIT4;

   /* obtain listening socket descriptor */
   sp->lsd = socket(af, type, proto);
   if (sp->lsd == INVALID_SOCKET) goto FAIL_INIT5;

   /* prepare family of address structure for binding */
   sp->addr.sin_family = af;

   /* set default backlog size */
   sp->backlog = 1024;

   /* set server name */
   sp->name = name;

   /* server created */
   return VEOK;

/* error handling / de-initialization */
FAIL_INIT5: condition_destroy(&(sp->alarm2));
FAIL_INIT4: condition_destroy(&(sp->alarm));
FAIL_INIT3: mutex_destroy(&(sp->lock2));
FAIL_INIT1: mutex_destroy(&(sp->lock));
FAIL_INIT0: return VERROR;
}  /* end server_init() */

/**
 * Set the io process functions for a Server context.
 * @param sp Pointer to Server context
 * @return (int) value indicating operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int server_setioprocess
   (Server *sp, AsyncProc donefn, AsyncProc initfn, AsyncProc procfn)
{
   /* check server context */
   if (sp == NULL) goto FAIL_INVAL;

   sp->donefn = donefn;
   sp->initfn = initfn;
   sp->procfn = procfn;

   return VEOK;

/* error handling */
FAIL_INVAL: set_errno(EINVAL); return VERROR;
}  /* end server_setioprocess() */

/**
 * Set a socket option for the Server socket.
 * @param sp Pointer to Server context
 * @return (int) value indicating operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int server_setsockopt(Server *sp, int level, int optname,
   const void *optval, socklen_t optlen)
{
   /* check server data can accept a socket option */
   if (sp == NULL || sp->lsd == INVALID_SOCKET) goto FAIL_INVAL;

   /* return the result of set socket option */
   return setsockopt(sp->lsd, level, optname, optval, optlen);

/* error handling */
FAIL_INVAL: set_errno(EINVAL); return VERROR;
}  /* end server_setsockopt() */

/**
 * Shutdown a server context. Closes server socket
 * @param data Pointer to data for AsyncWork
 * @returns (int) value representing the operation result
 * @retval VERROR on error; check errono for details
 * @retval VEOK on success
*/
int server_shutdown(Server *sp)
{
   Condition *alarmp;
   LinkedList *llp;
   LinkedNode *lnp;
   Mutex *lockp;
   int ecode;

#undef FnMSG
#define FnMSG(x) "server_shutdown(): " x

   /* init */
   alarmp = &(sp->alarm);
   lockp = &(sp->lock);

   /* acquire server lock */
   on_ecode_goto_perrno( mutex_lock(lockp), FATAL, "LOCK FAILURE");

   /* flag for graceful shutdown */
   sp->running = 0;
   /* alert threads of state change */
   condition_broadcast(alarmp);
   /* wait for active threads to end */
   for (llp = &(sp->active); llp->count; llp = &(sp->active)) {
      /* wait for threads to finish, up to 5 seconds... */
      if (condition_timedwait(alarmp, lockp, 5000)) {
         if (errno == CONDITION_TIMEOUT) {
            plog("Taking too long, terminating...");
         } else perrno(FnMSG("condition_timedwait() FAILURE"));
         break;
      }
      /* join with any exited threads */
      llp = &(sp->exited);
      while ((lnp = llp->next)) {
         on_ecode_goto_perrno( link_node_remove(lnp, llp),
            FATAL, FnMSG("link_node_remove() FAILURE"));
         on_ecode_goto_perrno( thread_join(*((Thread *) lnp->data)),
            FATAL, FnMSG("thread_join() FAILURE"));
         /* free LinkedNode and associated data */
         free(lnp->data);
         free(lnp);
      }  /* end while (sp->exited.next) */
   }  /* end while (sp->active.count) */

   /* release thread lock when finished */
   on_ecode_goto_perrno( mutex_unlock(lockp), FATAL, "UNLOCK FAILURE");

   return VEOK;

FATAL:
   /* done */
   return ecode;
}  /* ens server_shutdown() */

/**
 * Start a server on a specified port with number of worker threads.
 * @param sp Pointer to Server context
 * @param addr 32-bit IPv4 address value
 * @param port 16-bit port number of server
 * @param workers Number of worker threads
 * @return (int) value indicating operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int server_start(Server *sp, word32 addr, word16 port, int workers)
{
   LinkedNode *lnp;

   /* prepare remaining address structure for binding */
   sp->addr.sin_addr.s_addr = addr;
   sp->addr.sin_port = htons(port);

   sp->numthreads = workers;

   /* start the main handler thread... */
   sp->running = 1;

   /* create LinkedNode for Thread */
   lnp = link_node_create(sizeof(Thread));
   if (lnp == NULL) goto FAIL_INIT1;
   /* add LinkedNode to thread list */
   if (link_node_append(lnp, &(sp->active))) goto FAIL_INIT2;
   /* create (start) thread in LinkedNode data */
   if (thread_create(lnp->data, server__main, sp)) goto FAIL_INIT3;

   /* start (at least 1) worker threads to handle work... */

   do {
      /* create LinkedNode for Thread */
      lnp = link_node_create(sizeof(Thread));
      if (lnp == NULL) goto FAIL_INIT1;
      /* add LinkedNode to thread list */
      if (link_node_append(lnp, &(sp->active))) goto FAIL_INIT2;
      /* create (start) thread in LinkedNode data */
      if (thread_create(lnp->data, server__worker, sp)) goto FAIL_INIT3;
   } while (--workers > 0);

   /* server started */
   return VEOK;

FAIL_INIT3: link_node_remove(lnp, &(sp->active));
FAIL_INIT2: free(lnp->data); free(lnp);
FAIL_INIT1:
   /* shutdown and cleanup threads */
   server_shutdown(sp);
   return VERROR;
}  /* end server_start() */

/**
 * Create and enqueue server work with specified data.
 * @param sp Pointer to Server
 * @param data Pointer to data for AsyncWork
*/
int server_work_create(Server *sp, void *data)
{
   return server__create_work(sp, data);
}  /* end server_work_create() */

/**
 * Cleanup AsyncWork releasing held resources back to the system.
 * @param sp Pointer to Server
 * @param wp Pointer to AsyncWork
*/
void server_work_cleanup(Server *sp, AsyncWork *wp)
{
   server__cleanup_work(sp, wp);
}  /* end server_work_cleanup() */
