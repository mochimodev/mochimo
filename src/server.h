/**
 * @file server.h
 * @brief Mochimo server support.
 * @details Provides support for socket connections by a server handler.
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from the extended-c extinet header and
 * requires the use of wsa_startup() to activate socket support on Windows.
*/

/* include guard */
#ifndef MOCHIMO_SERVER_H
#define MOCHIMO_SERVER_H


/* external support */
#include "exterrno.h"
#include "extinet.h"
#include "exttime.h"

/** Static initializer for a SERVER context struct. */
#define SERVER_INITIALIZER { 0, .mutex = MUTEX_INITIALIZER, \
   .cnd = CONDITION_INITIALIZER, .sd = INVALID_SOCKET }

/* IOWAIT indicates a connection "waiting" for I/O to become available on
 * the socket, typically returned from the "on_io" server event function
 */
#ifndef IOWAIT
   #define IOWAIT -2
#endif

/** Network connection handler struct. */
typedef struct connection {
   struct pollfd pollfd;   /* file descriptor and events */
   time_t to;              /* inactivity timeout trigger */
   void *data;             /* auxiliary connection data */
} CONNECTION;

/** Server context struct for managing server connections. */
typedef struct server {
   /**
    * Called when the server wishes to log information.
    * @param errnum Error number associated with log, or 0 for no error
    * @param message Log message string
    */
   void (*logger)(int errnum, const char *message);
   /**
    * Called when a new connection has been accept()'d by the server. This
    * helps the server to decide whether to accept the connection or not.
    * @param cp Pointer to a CONNECTION
    * @param addrp Pointer to a sockaddr struct containing connection info
    * @param len Length of the data in sockaddr struct
    * @return 0 to accept the connection, or non-zero on error. If an error
    * is indicated and errno is set to ENOMEM, the server will hold the
    * connection and retry the on_accept event function after processing
    * existing in progress connections in an attempt to free some memory.
    * For all other error conditions, the server will drop the connection.
    */
   int (*on_accept)(CONNECTION *cp, struct sockaddr *addrp, socklen_t len);
   /**
    * Called when a connection is ready for cleanup.
    * Auxiliary data should be handled by this event function.
    * @param cp Pointer to a CONNECTION
    * @return 0 to have the server cleanup the CONNECTION, or non-zero to
    * have the server skip CONNECTION cleanup to later resume network I/O
    */
   int (*on_cleanup)(CONNECTION *cp);
   /**
    * Called when a connection is ready for I/O, as indicated by pollfd.
    * @param cp Pointer to a CONNECTION
    * @return -2 to indicate the connection is waiting for network I/O.
    * POLL events should also be set to represent the desired I/O operation.
    * All other values will be interpreted as no longer waiting for I/O.
    */
   int (*on_io)(CONNECTION *cp);

   Mutex mutex;   /**< lock for connections list */
   Condition cnd; /**< condition for external signalling */
   DLLIST queue;  /**< connections being passed INTO the server */
   int shutdown;  /**< flag indicating the server to shutdown */
   int backlog;   /**< maximum number of pending (and queued) connections */
   SOCKET sd;     /**< server file descriptor */
} SERVER;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int server_queue(SERVER *sp, void *data, struct sockaddr *addrp);
int server_shutdown(SERVER *sp, int seconds);
int server_start(SERVER *sp, struct sockaddr *addrp, socklen_t len);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
