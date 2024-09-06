/**
 * @private
 * @headerfile server.h <server.h>
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_SERVER_C
#define MOCHIMO_SERVER_C


#include "server.h"

/* external support */
#include <stdio.h>

/* cross-platform compatible definitions and/or includes */
#ifdef _WIN32
   #define poll(fds, nfds, to)   WSAPoll(fds, nfds, to)

   typedef ULONG nfds_t;

#endif

/* define MACRO server logger */
#define SERVER__LOG(SP, E, MSG) { if ((SP)->logger) (SP)->logger(E, MSG); }

/* define MACRO abort() procedure on mutex_unlock() failure */
#define SERVER__MUTEX_UNLOCK_OR_ABORT(SP) \
   do { \
      if (mutex_unlock(&((SP)->mutex)) != 0) { \
         SERVER__LOG(sp, errno, "FATAL SERVER MUTEX UNLOCK ERROR"); \
         SERVER__LOG(sp, 0, "Unrecoverable // Aborting..."); \
         abort(); \
      } \
   } while (0)

/* default timeout for incoming connections */
#define DEFAULT_TIMEOUT 1

static inline void server__cleanup_connection(SERVER *sp, CONNECTION *cp)
{
   if (cp == NULL) return;
   /* execute additional cleanup routines */
   if (sp->on_cleanup && sp->on_cleanup(cp) == 0) return;
   /* cleanup and deallocate connection resources */
   if (cp->pollfd.fd != INVALID_SOCKET) closesocket(cp->pollfd.fd);
   if (cp->data) free(cp->data);
   free(cp);
}

static inline int server__cleanup(SERVER *sp, DLNODE *np, DLLIST *lp)
{
   if (np == NULL) return 0;

   /* cleanup and deallocate connection resources */
   server__cleanup_connection(sp, (CONNECTION *) np->data);

   /* cleanup and deallocate node resources */
   if (lp) {
      /* remove node from list */
      if (dlnode_remove(np, lp) != 0) {
         return SOCKET_ERROR;
      }
   }
   free(np);

   return 0;
}  /* end server__cleanup() */

static inline void server__flag_shutdown(SERVER *sp)
{
   /* flag server for shutdown */
   sp->shutdown = 1;
   /* alert server of state change */
   condition_broadcast(&(sp->cnd));
}  /* end server__flag_shutdown */

/*
static void server__name(SERVER *sp, struct sockaddr *addrp, socklen_t len)
{
   char name[NI_MAXHOST + NI_MAXSERV + 18];
   char host[NI_MAXHOST];
   char serv[NI_MAXSERV];
*/
   /* verify bound sockaddr *//*
   if (getsockname(sp->sd, addrp, &len) == 0) {
      if (getnameinfo(addrp, len, host, NI_MAXHOST, serv, NI_MAXSERV, \
            NI_NUMERICHOST | NI_NUMERICSERV) == 0) {*/
         /* build host name and service port *//*
         snprintf(name, sizeof(name), "%s:%s", host, serv);
      }
   }
}*/  /* end server__name */

/**
 * Queue a connection for the server to process.
 * @param sp Server structure pointer
 * @param data Auxiliary data to associate with the connection
 * @param addrp Socket address struct for connection
 * @param len Length of the socket address struct
 * @return 0 on successful queue, or SOCKET_ERROR on error.
 * Check errno for details.
 */
int server_queue(SERVER *sp, void *data, struct sockaddr *addrp)
{
   CONNECTION *cp;
   DLNODE *np;
   SOCKET sd;
   int ecode;

   /* check server and shutdown flag */
   if (sp == NULL) {
      set_errno(EINVAL);
      return SOCKET_ERROR;
   } else if (sp->shutdown) {
#ifdef _WIN32
      set_alterrno(WSAESHUTDOWN);
#else
      set_errno(ESHUTDOWN);
#endif
      return SOCKET_ERROR;
   }

   /* create connection for server work */
   np = dlnode_create(sizeof(CONNECTION));
   if (np == NULL) return SOCKET_ERROR;
   memset(np->data, 0, sizeof(CONNECTION));
   cp = (CONNECTION *) np->data;

   /* create socket and set non-blocking */
   sd = socket(addrp->sa_family, SOCK_STREAM, 0);
   if (sd == INVALID_SOCKET || set_nonblocking(sd) != 0) {
      goto ERROR_CLEANUP;
   }

   /* initiate connection to sockaddr */
   if (connect_auto(sd, addrp) != 0) {
      ecode = socket_errno;
      /* check for immediate failures -- not connected nor connecting */
      if (!socket_is_connected(ecode) && !socket_is_connecting(ecode)) {
         goto ERROR_CLEANUP;
      }
   }
   /* set default CONNECTION data */
   cp->pollfd.fd = sd;
   cp->pollfd.events = POLLOUT;
   cp->to = time(NULL) + DEFAULT_TIMEOUT;
   cp->data = NULL;

   /* (BLOCKING) lock and add node to server queue */
   if (mutex_lock(&(sp->mutex)) != 0) goto ERROR_CLEANUP;
   /* add node to server queue */
   if (dlnode_append(np, &(sp->queue)) != 0) {
      SERVER__MUTEX_UNLOCK_OR_ABORT(sp);
      goto ERROR_CLEANUP;
   }
   /* pass data on successful queue */
   cp->data = data;
   /* signal server of new work */
   condition_broadcast(&(sp->cnd));
   /* unlock exclusive hold on queue */
   SERVER__MUTEX_UNLOCK_OR_ABORT(sp);

   return 0;

   /* cleanup / error handling */
ERROR_CLEANUP:
   server__cleanup(sp, np, NULL);

   return SOCKET_ERROR;
}  /* end server_queue() */

/**
 * Requeue an existing connection for the server to process.
 * @param sp Server structure pointer
 * @param cp Connection structure pointer
 * @return 0 on successful queue, or SOCKET_ERROR on error.
 * Check errno for details.
 */
int server_requeue(SERVER *sp, CONNECTION *cp)
{
   DLNODE *np;
   int ecode;

   /* check server and shutdown flag */
   if (sp == NULL) {
      set_errno(EINVAL);
      return SOCKET_ERROR;
   } else if (sp->shutdown) {
#ifdef _WIN32
      set_alterrno(WSAESHUTDOWN);
#else
      set_errno(ESHUTDOWN);
#endif
      return SOCKET_ERROR;
   }

   /* create connection for server work */
   np = dlnode_create(0);
   if (np == NULL) return SOCKET_ERROR;

   /* (BLOCKING) lock and add node to server queue */
   if (mutex_lock(&(sp->mutex)) != 0) goto ERROR_CLEANUP;
   /* add node to server queue */
   if (dlnode_append(np, &(sp->queue)) != 0) {
      SERVER__MUTEX_UNLOCK_OR_ABORT(sp);
      goto ERROR_CLEANUP;
   }
   /* pass CONNECTION on successful queue */
   np->data = cp;
   /* signal server of new work */
   condition_broadcast(&(sp->cnd));
   /* unlock exclusive hold on queue */
   SERVER__MUTEX_UNLOCK_OR_ABORT(sp);

   return 0;

   /* cleanup / error handling */
ERROR_CLEANUP:
   server__cleanup(sp, np, NULL);

   return SOCKET_ERROR;
}  /* end server_queue() */

/**
 * Trigger (and optionally wait for) server to shutdown and release
 * all server resources.
 * @param sp Server structure pointer
 * @param seconds Timeout in seconds to wait for server to shutdown
 * @return 0 on successful shutdown, or SOCKET_ERROR on error.
 * Check errno for details.
 */
int server_shutdown(SERVER *sp, int seconds)
{
   int waited = 0;

   /* check function parameter */
   if (sp == NULL) {
      set_errno(EINVAL);
      return SOCKET_ERROR;
   }

   /* (wait for) acquire exclusive lock */
   if (mutex_lock(&(sp->mutex)) != 0) {
      return SOCKET_ERROR;
   }

   /* flag shutdown if not already */
   if (sp->shutdown == 0) {
      /* execute shutdown signal */
      server__flag_shutdown(sp);
   }

   /* wait for server to shutdown */
   for (waited = 0; waited < seconds && sp->shutdown != 2; waited++) {
      /* wait for a signal (or timeout) to continue processing */
      if (condition_timedwait(&(sp->cnd), &(sp->mutex), 1000) != 0) {
         if (errno != CONDITION_TIMEOUT) break;
      }  /* ... spurious wakeup? */
   }  /* ... check shutdown value */

   /* release exclusive lock */
   SERVER__MUTEX_UNLOCK_OR_ABORT(sp);

   /* check time waited */
   if (waited > 0 && waited >= seconds) {
      if (sp->shutdown != 2) {
         set_errno(ETIMEDOUT);
         return SOCKET_ERROR;
      }
   }

   return 0;
}  /* end server_shutdown() */

/**
 * Start a server on a provided with provided parameters. Runs until
 * shutdown signal, either by internal failure or via server_shutdown().
 * Use in alternate thread or process to avoid blocking.
 * @param sp Server structure pointer
 * @param sd Socket descriptor to use for listening
 * @param addr Socket address struct for binding
 * @return 0 on successful server completion, or non-zero on error.
 * Check errno for details.
 * @example
 * @code
 * SERVER node_server = SERVER_INITIALIZER;
 * struct sockaddr_in addr;
 * socklen_t len;
 *
 * node_server.sd = socket(AF_INET, SOCK_STREAM, 0);
 * if (node_server.sd == INVALID_SOCKET) return EXIT_FAILURE;
 * node_server.backlog = 1024;
 * node_server.logger = log_handler;
 * node_server.on_accept = assign_data;
 * node_server.on_cleanup = cleanup_data;
 * node_server.on_io = io_processor;
 *
 * addr.sin_addr.s_addr = htonl(INADDR_ANY);
 * addr.sin_family = AF_INET;
 * addr.sin_port = htons(2095);
 *
 * len = (socklen_t) sizeof(addr);
 *
 * server_start(&node_server, &addr, len);
 *
 * @endcode
 */
int server_start(SERVER *sp, struct sockaddr *addrp, socklen_t len)
{
   struct sockaddr_storage addr; /* socket address store for accept() */
   void *ptr = NULL;             /* generic pointer */
   CONNECTION *cp = NULL;        /* connection pointer */
   DLLIST queue = { 0 };         /* local connection queue list */
   DLNODE *anp = NULL;           /* (accept) list node pointer */
   DLNODE *hnp = NULL;           /* (held) list node pointer */
   DLNODE *nnp;                  /* (next) list node pointer */
   DLNODE *np;                   /* list node pointer */
   struct pollfd *fds = NULL;    /* pollfd array (copy) */
   nfds_t fdlen = 0;             /* pollfd array length */
   nfds_t fdi;                   /* pollfd array index */
   nfds_t nfds;                  /* number of active pollfd's */
   int dynasleep = 0;            /* dynamic sleep for IO checks */
   int ready;                    /* poll() return value */
   int ecode;                    /* error code */

   /* (try) bind address with listening socket */
   while (bind(sp->sd, addrp, len) != 0) {
      ecode = socket_errno;
      /* check shutdown flag and errors */
      if (sp->shutdown || !socket_is_inuse(ecode)) {
         SERVER__LOG(sp, ecode, "bind()");
         goto SHUTDOWN;
      }
      /* re-attempt after sleep (no spam) */
      millisleep(5000);
   }  /* end while (bind(sp->sd... */
   /* reassign sockaddr pointer for internal operations */
   addrp = (struct sockaddr *) &addr;
   len = sizeof(struct sockaddr_storage);
   /* set socket non-blocking and start listening */
   if (set_nonblocking(sp->sd) != 0) goto SHUTDOWN;
   if (listen(sp->sd, sp->backlog) != 0) goto SHUTDOWN;

   /* loop while no shutdown trigger -- update currtime time */
   while (!sp->shutdown) {
      /* accept incoming connections -- balance backlog */
      while (queue.count < sp->backlog) {
         /* prepare connection and containing list node */
         if (anp == NULL) {
            anp = dlnode_create(sizeof(CONNECTION));
            if (anp == NULL) {
               SERVER__LOG(sp, errno, "dlnode_create()");
               break;
            }
            /* update connection pointer */
            cp = (CONNECTION *) anp->data;
            cp->pollfd.fd = INVALID_SOCKET;
         }

         /* socket is held for recoverable errors */
         if (cp->pollfd.fd == INVALID_SOCKET) {
            /* update sockaddr len for accept() */
            len = sizeof(struct sockaddr_storage);
            /* accept() new connection from listening socket */
            cp->pollfd.fd = accept(sp->sd, addrp, &len);
            if (cp->pollfd.fd == INVALID_SOCKET) {
               ecode = socket_errno;
               /* break on empty queue and continue on connection reset */
               if (socket_is_waiting(ecode)) break;
               if (socket_is_reset(ecode)) continue;
               /* report unexpected errors */
               SERVER__LOG(sp, ecode, "accept()");
               break;
            }
         }

         /* set default CONNECTION data */
         cp->pollfd.events = POLLIN;
         cp->to = time(NULL) + DEFAULT_TIMEOUT;
         cp->data = NULL;

         /* trigger accept event function */
         if (sp->on_accept && sp->on_accept(cp, addrp, len) != 0) {
            /* break on OOM error, see docs for SERVER::on_accept() */
            if (errno == ENOMEM) break;
            continue;
         }
         /* add node to queue list for polling */
         if (dlnode_append(anp, &queue) != 0) goto SHUTDOWN;
         /* reset dynamic sleep on accept */
         dynasleep = 0;
         anp = NULL;
      }  /* end while (queue.count < sp->backlog... */

      /* ensure space in pollfd array */
      for (nfds = (nfds_t) queue.count; nfds > fdlen; nfds /= 3) {
         /* since pollfd[] array is already being copied every iteration
          * and we don't downsize pollfd[] at this stage, using realloc()
          * here will likely just add an erroneous memcpy()
          */
         ptr = malloc(sizeof(struct pollfd) * nfds);
         if (ptr == NULL && errno != ENOMEM) break;
         if (ptr) {
            if (fds) free(fds);
            fds = (struct pollfd *) ptr;
            fdlen = (nfds_t) nfds;
            break;
         }
      }  /* end for (nfds = ... */
      /* report errors where resulting list is less than requested */
      if (nfds < (nfds_t) queue.count) SERVER__LOG(sp, errno, "malloc()");
      /* update "held" node pointer for reduced loop iterations */
      if (hnp == NULL) hnp = queue.next;

      /* iterate through queue list preparing available poll space */
      for (nfds = 0, np = hnp; nfds < fdlen && np; np = nnp) {
         /* dereference connection and store "next" node pointer */
         cp = (CONNECTION *) np->data;
         nnp = np->next;
         /* trigger cleanup on invalid communications */
         if (cp->pollfd.fd == INVALID_SOCKET || cp->pollfd.events == 0) {
            if (server__cleanup(sp, np, &queue) != 0) goto FATAL;
            continue;
         }
         /* add pollfd data to fds array */
         fds[nfds++] = cp->pollfd;
      }  /* end for (nfds = 0, np = hnp ... */

      /* poll() (w/ dynamic timeout) */
      if (nfds > 0) {
         /* poll fds for ready sockets -- limit dynamic sleep */
         ready = poll(fds, nfds, dynasleep);
         if (dynasleep < 250) dynasleep++;
         if (ready == (-1)) SERVER__LOG(sp, errno, "poll()");
         if (ready > 0) {
            /* reset dynamic sleep on activity */
            dynasleep = 0;
            /* iterate through list of pollfd's -- check node sync */
            for (fdi = 0, np = hnp; fdi < nfds; fdi++, np = nnp) {
               /* dereference connection and store "next" node pointer */
               cp = (CONNECTION *) np->data;
               nnp = np->next;
               /* copy returned event to origin */
               cp->pollfd.revents = fds[fdi].revents;
               /* trigger io event function on io event */
               if (fds[fdi].revents & (POLLIN | POLLOUT)) {
                  if (sp->on_io && sp->on_io(cp) != IOWAIT) {
                     if (server__cleanup(sp, np, &queue) != 0) goto FATAL;
                  }
                  continue;
               }
               /* trigger cleanup on unexpected event or timeout */
               if (fds[fdi].revents || time(NULL) > cp->to) {
                  if (server__cleanup(sp, np, &queue) != 0) goto FATAL;
                  continue;
               }
            }  /* end for (fdi = 0... */
         }  /* end if (ready > 0) */
      }  /* end if (nfds > 0) */

      /* try (NON-BLOCKING) exclusive link server queue to local queue */
      if (mutex_trylock(&(sp->mutex)) == 0) {
         /* if there was no work, schleepy time... */
         if (nfds == 0 && dynasleep > 0) {
            /* wait for a signal (or timeout) to continue processing */
            if (condition_timedwait(&(sp->cnd), &(sp->mutex), dynasleep)) {
               if (errno != CONDITION_TIMEOUT) goto FATAL;
            }  /* ... spurious wakeups irrelevent */
         }
         /* check incoming queue count... */
         if (sp->queue.count > 0) {
            /* ... and transfer contents to local processing queue */
            if (dllist_append(&(sp->queue), &queue) != 0) goto FATAL;
         }
         /* release exclusive server lock */
         SERVER__MUTEX_UNLOCK_OR_ABORT(sp);
      }  /* end if (mutex_trylock... */
   }  /* end while (!sp->shutdown) */
   /* check shutdown value for "fatal" trigger */
   if (sp->shutdown == 2) goto FATAL;

SHUTDOWN:
   SERVER__LOG(sp, errno, "recv'd shutdown signal");
   /* close listening socket */
   if (sp->sd != INVALID_SOCKET) closesocket(sp->sd);
   /* free internally allocated or held resources */
   if (fds) free(fds);
   if (server__cleanup(sp, anp, NULL) != 0) goto FATAL;
   while (( np = queue.next )) {
      if (server__cleanup(sp, np, &queue) != 0) goto FATAL;
   }
   /* acquire exclusive lock for incoming resources */
   if (mutex_lock(&(sp->mutex)) != 0) goto FATAL;
   /* free incoming queued resources */
   while (( np = sp->queue.next )) {
      if (server__cleanup(sp, np, &(sp->queue)) != 0) goto FATAL;
   }
   /* release exclusive lock */
   SERVER__MUTEX_UNLOCK_OR_ABORT(sp);

   return 0;

FATAL:
   /* rebort unrecoverable and abort() to avoid (potential) deadlock */
   SERVER__LOG(sp, errno, "FATAL SERVER ERROR");
   SERVER__LOG(sp, 0, "Unrecoverable // Aborting...");
   abort();
}  /* end server_start() */

/* cleanup internal MACROs */
#undef SERVER__MUTEX_UNLOCK_OR_ABORT
#undef SERVER__LOG

/* end include guard */
#endif
