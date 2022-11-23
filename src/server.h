/**
 * @file server.h
 * @brief Mochimo server and related task handling support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from the extended-c extinet header
 * and requires the use of sock_startup() to activate socket support.
*/

/* include guard */
#ifndef MOCHIMO_SERVER_H
#define MOCHIMO_SERVER_H


/* internal support */
#include "types.h"

/* external support */
#include "extlib.h"

/** Asynchronous Work struct */
typedef struct {
   void *data;    /**< server work data pointer */
   time_t to;     /**< socket inactivity timeout time */
   SOCKET sd;     /**< socket descriptor of connection */
   short sio;     /**< socket IO wait type (IO_*) */
   short defer;   /**< flag indicating work should be deferred */
} AsyncWork;

/** Asynchronous Process function used on AsyncWork in various states */
typedef int (*AsyncProc)(AsyncWork *np);

/** Server context struct */
typedef struct {
   Mutex lock;                /**< mutually exclusive server lock */
   Mutex inlock;              /**< mutually exclusive "in" work lock */
   Mutex outlock;             /**< mutually exclusive "out" work lock */
   Condition alarm;           /**< condition variable server signals */
   Condition inalarm;         /**< condition variable "in" work signals */
   LinkedList active;         /**< list of active worker threads */
   LinkedList exited;         /**< list of exited worker threads */
   LinkedList inIO;           /**< list of "in" work, ready for io */
   LinkedList outIO;          /**< list of "out" work, waiting for io */
   /** work completion function called on completed work */
   AsyncProc donefn;
   /** work initialization function called on work received by accept() */
   AsyncProc initfn;
   /** work processing function called on work with active IO */
   AsyncProc procfn;
   struct sockaddr_in addr;   /**< internet socket server address data */
   int deferthreads;          /**< threads processing deferred work */
   int idlethreads;           /**< threads idling -- not processing */
   int numthreads;            /**< total server threads */
   int backlog;               /**< server backlog size (for listen()) */
   int running;               /**< flag indicating the server is ok */
   int lsd;                   /**< listen socket descriptor */
} Server;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int server_destroy(Server *sp);
int server_init(Server *sp, int af, int type, int proto);
int server_setprocess
   (Server *sp, AsyncProc donefn, AsyncProc initfn, AsyncProc procfn);
int server_setsockopt
   (Server *sp, int level, int optname, const char *optval, int optlen);
int server_shutdown(Server *sp);
int server_start(Server *sp, word32 addr, word16 port, int workers);
int server_work_create(Server *sp, void *data);
void server_work_cleanup(Server *sp, AsyncWork *wp);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
