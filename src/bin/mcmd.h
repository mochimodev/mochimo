/**
 * @file mcmd.h
 * @brief Mochimo Server Daemon support header.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_DAEMON_H
#define MOCHIMO_DAEMON_H


/* internal support */
#include "block.h"
#include "chain.h"
#include "error.h"
#include "ledger.h"
#include "peer.h"
#include "protocol.h"
#include "trigg.h"

/* external support */
#include "extio.h"
#include "extlib.h"
#include "extmath.h"
#include "exttime.h"
#include <signal.h>
#include <string.h>

extern int Running;
extern int Nopush_opt;
extern word32 Quorum_opt;
extern word64 Trustblock_opt;
extern word16 Dstport_opt;
extern word16 Port_opt;

extern word32 Mfee[2];
extern word32 Myfee[2];

/* Recent peers lock */
extern RWLock RplistLock;

/* Server Task List's, and related Mutex's */
extern Condition ActiveIOAlarm;
extern Mutex ActiveIOLock;
extern Mutex InactiveIOLock;
extern Mutex SyncIOLock;
extern LinkedList ActiveIO;
extern LinkedList InactiveIO;
extern LinkedList SyncIO;

/* Server status flags */
extern int ServerSyncup;
extern int ServerInit;
extern int ServerOk;

/* cli options for directories */
extern char *Bcdir_opt;
extern char *Ltdir_opt;
extern char *Spdir_opt;
extern char *Txdir_opt;

/* cli options for peerlist filenames */
extern char *Coreip_opt;
extern char *Epinkip_opt;
extern char *Localip_opt;
extern char *Recentip_opt;
extern char *Startip_opt;

/* cli options for peerlist web address */
extern char *Starthttp_opt;

/* end include guard */
#endif
