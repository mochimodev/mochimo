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

/* default timeout for incoming connections */
#define DEFAULT_TIMEOUT 1

static inline void server__cleanup_connection(SERVER *sp, CONNECTION *cp)
{
   if (cp == NULL) return;
   /* execute additional cleanup routines */
   if (sp->on_cleanup) sp->on_cleanup(cp);
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

   /* check server */
   if (sp == NULL) {
      set_errno(EINVAL);
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
   cp->data = data;

   /* (BLOCKING) lock and add node to server queue */
   if (mutex_lock(&(sp->mutex)) != 0) goto ERROR_CLEANUP;
   /* add node to server queue */
   if (dlnode_append(np, &(sp->queue)) != 0) {
      if (mutex_unlock(&(sp->mutex)) != 0) {
         goto DEADLOCK_CLEANUP;
      }
   }
   /* unlock exclusive hold on queue */
   if (mutex_unlock(&(sp->mutex)) != 0) {
      goto DEADLOCK_CLEANUP;
   }

   return 0;

   /* cleanup / error handling */
DEADLOCK_CLEANUP:
   sp->shutdown = 1;
ERROR_CLEANUP:
   server__cleanup(sp, np, NULL);

   return SOCKET_ERROR;
}  /* end server_queue() */

/**
 * Start a server on a provided socket.
 * @param sp Server structure pointer
 * @param sd Socket descriptor to use for listening
 * @param addr Socket address struct for binding
 * @param backlog Maximum number of pending connections
 * @return 0 on successful server completion, or non-zero on error.
 * Check errno for details.
 * @example
 * @code
 * SERVER server = SERVER_INITIALIZER;
 * struct sockaddr_in addr;
 * socklen_t len;
 *
 * server.fd = socket(AF_INET, SOCK_STREAM, 0);
 * if (server.fd == INVALID_SOCKET) return EXIT_FAILURE;
 * server.logger = log_handler;
 * server.on_accept = assign_data;
 * server.on_cleanup = cleanup_data;
 * server.on_io = io_processor;
 *
 * addr.sin_addr.s_addr = htonl(INADDR_ANY);
 * addr.sin_family = AF_INET;
 * addr.sin_port = htons(12345);
 *
 * len = (socklen_t) sizeof(addr);
 *
 * server(&server, &addr, len, 1024);
 *
 * @endcode
 * ... server() runs until it receives a shutdown signal, or fails.
 */
int server(SERVER *sp, struct sockaddr *addrp, socklen_t len, int backlog)
{
#define SERVER__LOG(SP, E, MSG) { if ((SP)->logger) (SP)->logger(E, MSG); }

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

   /* request winsock dll version 2.2 */
   if (wsa_startup(2, 2) != 0) {
      return SOCKET_ERROR;
   }

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
   if (listen(sp->sd, backlog) != 0) goto SHUTDOWN;

   /* loop while no shutdown trigger -- update currtime time */
   while (!sp->shutdown) {
      /* accept incoming connections -- balance backlog */
      while (queue.count < backlog) {
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
      }  /* end while (queue.count < backlog... */

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

      /* sleep or poll() with timeout */
      if (nfds == 0) millisleep(dynasleep);
      else if (nfds > 0) {
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
                  if (sp->on_io) sp->on_io(cp);
                  if (cp->pollfd.fd == INVALID_SOCKET) {
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
         if (dllist_append(&(sp->queue), &queue) != 0) goto FATAL;
         if (mutex_unlock(&(sp->mutex)) != 0) goto FATAL;
      }  /* end if (mutex_trylock... */
   }  /* end while (!sp->shutdown) */

SHUTDOWN:
   SERVER__LOG(sp, errno, "recv'd shutdown signal");
   /* set shutdown */
   sp->shutdown = 1;
   /* close listening socket */
   if (sp->sd != INVALID_SOCKET) closesocket(sp->sd);
   /* free internally allocated or held resources */
   if (server__cleanup(sp, anp, NULL) != 0) goto FATAL;
   while (( np = queue.next )) {
      if (server__cleanup(sp, np, &queue) != 0) goto FATAL;
   }

#ifdef _WIN32
   /* Windows exclusive cleanup */
   wsa_cleanup();

#endif

   return 0;

FATAL:
   SERVER__LOG(sp, errno, "FATAL SERVER ERROR!!!");
   /* kill server on fatal */
   sp->shutdown = 1;

   /* winsock cleanup */
   wsa_cleanup();

   return SOCKET_ERROR;

#undef SERVER__LOG
}  /* end server() */

/* end include guard */
#endif
