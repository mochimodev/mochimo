
#include "server.h"

/* internal support */
#include "error.h"

/* external support */
#include "extinet.h"
#include "extlib.h"
#include <string.h>

/* standard server connection TIMEOUT, 30 seconds */
#define SERVER_TIMEOUT  30

/* MACRO for handling excessive connections on various systems */
/** @todo migrate to cross platform polling solution */
#ifdef _WIN32
   #define SELECT_CANNOT_HANDLE(sd) ( 0 )

#else
   #define SELECT_CANNOT_HANDLE(sd) ( sd >= FD_SETSIZE )

#endif

#define server__fatal(name, ...) { \
   ptrace_functions(1); \
   perrno(__VA_ARGS__); \
   palert("CRITICAL SERVER(%s) RUNTIME ERROR", name); \
}

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
   if (sp->on_finish) sp->on_finish(wp);
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
   LinkedList ready_io = { 0 };  /* lock-free work list ready for IO */
   LinkedList wait_io = { 0 };   /* lock-free work list waiting for IO */
   LinkedNode *lnp, *next_lnp;
   LinkedNode *alnp;
   AsyncWork *wp;
   struct sockaddr *bindaddrp;
   struct timeval tv;            /* sleep time during IO checks */
   long dynasleep;               /* dynamic sleep for IO checks */
   time_t currtime;
   SOCKET asd;                   /* next socket descriptor */
   SOCKET nfds;                  /* for select(nfds, ...) */
   int ecode, count, i;
   char error[64];
   char ipstr[16];
   char name[16];

   /* init */
   bindaddrp = (struct sockaddr *) &(sp->addr);
   dynasleep = 0;
   alnp = NULL;
   i = 0;

   /*********************/
   /* SERVER BIND PHASE */
   snprintf(name, sizeof(name), "%s-binding", sp->name);
   thread_setname(thread_self(), name);

   /* (try) bind address with listening socket */
   while (bind(sp->lsd, bindaddrp, sizeof(sp->addr))) {
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
         if (SELECT_CANNOT_HANDLE(asd)) {
            pdebug("drop sd= %d: select() OVERLOAD", asd);
            sock_close(asd);
            continue;
         }
         /* accept work with asd and aip data */
         alnp = server__accept_work(alnp, asd, IO_RECV);
         if (alnp == NULL) {
            strerror_mcm((ecode = errno), error, sizeof(error));
            pdebug("drop sd= %d: (%d) %s", asd, ecode, error);
            sock_close(asd);
            continue;
         }
         /* call custom event function on accept()'d work... */
         if (sp->on_accept && sp->on_accept(alnp->data) != 0) {
            /* ... if function returns non-zero, drop connection */
            sock_close(asd);
            continue;
         }
         /* add work to wait list */
         if (link_node_append(alnp, &wait_io)) {
            server__fatal(sp->name, "link_node_append() FAILURE");
            goto SHUTDOWN;
         };
         /* dynamic sleep reset -- reset alnp */
         if (dynasleep) dynasleep = 0;
         alnp = NULL;
      }  /* end for (i = 0; i < SERVER_LISTEN_LIMIT ... */

      /* update currtime for timeout check */
      time(&currtime);

      /* zero fd sets and reset highest descriptor */
      FD_ZERO(&rfds);
      FD_ZERO(&wfds);
      nfds = INVALID_SOCKET;
      /* prepare fd_set's only if waiting for IO */
      if (wait_io.count) {
         count = FD_SETSIZE;
         /* add "waiting" sockets to appropriate fd_set's -- adjust nsd */
         for (lnp = wait_io.next; count > 0 && lnp; lnp = next_lnp) {
            next_lnp = lnp->next;
            /* dereference the server work */
            wp = (AsyncWork *) lnp->data;
            /* cleanup any stray work with an invalid socket descriptor */
            if (wp->sd == INVALID_SOCKET) {
               if (link_node_remove(lnp, &wait_io)) {
                  server__fatal(sp->name, "link_node_remove() FAILURE");
                  goto SHUTDOWN;
               };
               server__cleanup_node(sp, lnp);
               continue;
            } else if (wp->to && difftime(currtime, wp->to) > 0) {
               /* move node from "wait" list to "ready" list */
               if (link_node_remove(lnp, &wait_io)) {
                  server__fatal(sp->name, "link_node_remove() FAILURE");
                  goto SHUTDOWN;
               }
               if (link_node_append(lnp, &ready_io)) {
                  server__fatal(sp->name, "link_node_append() FAILURE");
                  goto SHUTDOWN;
               }
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
         }  /* end for (lnp = wait_io... */
      }  /* end for if (wait_io.count) */
      /* set timeout as preferred sleep duration for this iteration */
      tv.tv_sec = 0;
      tv.tv_usec = 1000L * dynasleep;
      if (dynasleep < 500) dynasleep++;
      /* check number of "waiting" sockets ready for send/recv/fail */
      count = select((int) (nfds + 1), &rfds, &wfds, NULL, &tv);
      /* IMPORTANT: tv's value at this point SHOULD NOT be re-used */

      /* walk wait_io with respect to count -- hide 'n' seek */
      for (lnp = wait_io.next; count > 0 && lnp; lnp = next_lnp) {
         /* store next link, as links may change */
         next_lnp = lnp->next;
         /* dereference the work */
         wp = (AsyncWork *) lnp->data;
         /* check send/recv fd_set's and timeout */
         /* NOTE: "<= IO_SEND" includes IO_CONN in conditional check */
         if ((wp->sio <= IO_SEND && FD_ISSET(wp->sd, &wfds)) ||
            (wp->sio == IO_RECV && FD_ISSET(wp->sd, &rfds)) ) {
            /* move node from "wait" list to "ready" list */
            if (link_node_remove(lnp, &wait_io)) {
               server__fatal(sp->name, "link_node_remove() FAILURE");
               goto SHUTDOWN;
            } else if (link_node_append(lnp, &ready_io)) {
               server__fatal(sp->name, "link_node_append() FAILURE");
               goto SHUTDOWN;
            }
            count--;
         }  /* end (in)activity checks... */
      }  /* end for (lnp = wait_io... */

      /* check "ready" lists */
      if (ready_io.count) {
         /* dynamic sleep reset */
         if (dynasleep) dynasleep = 0;
         /* store list counts */
         count = ready_io.count;
         /* try (NON-BLOCKING) acquire work lock for link list */
         if (mutex_trylock(&(sp->lock)) == 0) {
            /* link ready_io to ActiveIO */
            if (link_list_append(&ready_io, &(sp->inIO))) {
               server__fatal(sp->name, "link_list_append() FAILURE");
               goto SHUTDOWN;
            }
            /* signal idle threads indicating available work */
            if (count <= sp->idlethreads) {
               while (count--) condition_signal(&(sp->alarm));
            } else condition_broadcast(&(sp->alarm));
            /* release lock if acquired */
            if (mutex_unlock(&(sp->lock))) {
               server__fatal(sp->name, "UNLOCK FAILURE");
               goto SHUTDOWN;
            }
         }  /* end if (mutex_trylock... */
      }  /* end if (ready_io.count) */

      /* try (NON-BLOCKING) link InactiveIO to wait_io */
      if (mutex_trylock(&(sp->lock2)) == 0) {
         if (link_list_append(&(sp->outIO), &wait_io)) {
            server__fatal(sp->name, "link_list_append() FAILURE");
            goto SHUTDOWN;
         }
         /* release lock if acquired */
         if (mutex_unlock(&(sp->lock2))) {
            server__fatal(sp->name, "UNLOCK FAILURE");
            goto SHUTDOWN;
         }
      }  /* end if (mutex_trylock... */
   }  /* end while (sp->running) */
   pdebug("server recv'd shutdown signal");

SHUTDOWN:
   /* close listening socket */
   if (sp->lsd != INVALID_SOCKET) {
      sock_close(sp->lsd);
      sp->lsd = INVALID_SOCKET;
   }
   /* free internally allocated or held resources */
   server__cleanup_node(sp, alnp);
   while ((lnp = ready_io.next)) {
      if (link_node_remove(lnp, &ready_io)) {
         server__fatal(sp->name, "ready_io SHUTDOWN FAILURE");
         goto FATAL;
      }
      server__cleanup_node(sp, lnp);
   }
   while ((lnp = wait_io.next)) {
      if (link_node_remove(lnp, &wait_io)) {
         server__fatal(sp->name, "wait_io SHUTDOWN FAILURE");
         goto FATAL;
      }
      server__cleanup_node(sp, lnp);
   }
   /* trigger worker thread shutdown */
   sp->running = 0;
   /* return result of server__exit() */
   return server__exit(sp);

FATAL:
   pdebug("FATAL ERROR, TERMINATING...");
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

   /* init */
   pdebug("server worker(%x) starting...", thread_selfid());
   deferproc = 0;

   /* acquire "in" work Lock */
   if (mutex_lock(&(sp->lock))) {
      server__fatal(sp->name, "server inIO (init) LOCK FAILURE");
      goto FATAL;
   }

   /* main thread loop -- prioritize SyncIO tasks */
   while (sp->running) {
      /* indicate thread is currently in asynchronous mode */
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
         if (link_node_remove(lnp, &(sp->inIO))) {
            server__fatal(sp->name, "server inIO LIST FAILURE");
            goto FATAL;
         }
         /* release work lock */
         if (mutex_unlock(&(sp->lock))) {
            server__fatal(sp->name, "server inIO UNLOCK FAILURE");
            goto FATAL;
         }
         /* dereference and process thread work io */
         if ((sp->on_io && sp->on_io(wp) == VEWAITING)) {
            /* return task to wait list for IO waiting */
            if (mutex_lock(&(sp->lock2))) {
               server__fatal(sp->name, "server outIO LOCK FAILURE");
               goto FATAL;
            }
            if (link_node_append(lnp, &(sp->outIO))) {
               server__fatal(sp->name, "server outIO LIST FAILURE");
               /* release lock on failure */
               mutex_unlock(&(sp->lock2));
               goto FATAL;
            }
            if (mutex_unlock(&(sp->lock2))) {
               server__fatal(sp->name, "server outIO UNLOCK FAILURE");
               goto FATAL;
            }
         } else server__cleanup_node(sp, lnp);
         /* (re)acquire "in" work Lock */
         if (mutex_lock(&(sp->lock))) {
            server__fatal(sp->name, "server inIO LOCK FAILURE");
            goto FATAL;
         }
         /* check SHUTDOWN flag for jump */
         if (sp->running == 0) goto SHUTDOWN;
         /* restore deference parameters */
         if (deferproc) {
            sp->deferthreads--;
            deferproc = 0;
         }
      }  /* end while ((lnp = sp->worklst.next)) */
      /* indicate thread is currently in idle mode */
      snprintf(name, sizeof(name), "%s-idle", sp->name);
      thread_setname(thread_self(), name);
      /* wait for condition, sleepy time ... */
      sp->idlethreads++;
      ecode = condition_wait(&(sp->alarm), &(sp->lock));
      sp->idlethreads--;
      /* ... wakeup (spurious?), check ecode ... */
      if (ecode) {
         perrno("CONDITION FAILURE");
         goto FATAL;
      }
   }  /* end while (sp->running) */

SHUTDOWN:
   pdebug("server worker recv'd shutdown signal");

   /* release work list lock */
   if (mutex_unlock(&(sp->lock))) {
      server__fatal(sp->name, "server inIO (end) UNLOCK FAILURE");
      goto FATAL;
   }

   return server__exit(sp);

FATAL:
   palert("FATAL SERVER WORKER ERROR, TERMINATING...");
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

   /* init */
   alarmp = &(sp->alarm);
   lockp = &(sp->lock);

   /* acquire server lock */
   if (mutex_lock(lockp)) {
      server__fatal(sp->name, "LOCK FAILURE");
      goto FATAL;
   }

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
         } else perrno("condition_timedwait() FAILURE");
         break;
      }
      /* join with any exited threads */
      llp = &(sp->exited);
      while ((lnp = llp->next)) {
         if (link_node_remove(lnp, llp)) {
            server__fatal(sp->name, "link_node_remove() FAILURE");
            goto FATAL;
         }
         if (thread_join(*((Thread *) lnp->data))) {
            server__fatal(sp->name, "thread_join() FAILURE");
            goto FATAL;
         }
         /* free LinkedNode and associated data */
         free(lnp->data);
         free(lnp);
      }  /* end while (sp->exited.next) */
   }  /* end while (sp->active.count) */

   /* release thread lock when finished */
   if (mutex_unlock(lockp)) {
      server__fatal(sp->name, "UNLOCK FAILURE");
      goto FATAL;
   }

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
