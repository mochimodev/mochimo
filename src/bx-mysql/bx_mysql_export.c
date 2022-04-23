/** bx.c  Block Explorer - Database Export Feature
 *
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * NOTE: compile with:  cc -o bx bx.c trigg.o sha256.o -I/usr/include/mysql -lmysqlclient -DEXPORT_MYSQL
 *
 * Original v0.1 Author: Tim Cotten <tcotten@mochimo.org> - 06/20/2018
 * 
 * v0.2 by Stackoverflo - 09/20/2020
 *
 * Copyright (c) 2020 by Adequate Systems, LLC.  All Rights Reserved.
 */

#include "extprint.h"   /* print/logging support */

#include "../config.h"
#include "../mochimo.h"

#define EXCLUDE_NODES

word8 Errorlog = 1;  /* since not including data.c */
word8 Bgflag;
word8 Monitor;
word8 Running = 1;
word32 Trace = 1;
word32 Nsolved;
pid_t Mpid, Sendfound_pid;  /* in error.c */

#include "../algo/peach/peach.c"

#define MYSQL_CONF_NUM 4
#define MYSQL_CONF_HOSTNAME 0
#define MYSQL_CONF_DATABASE 1
#define MYSQL_CONF_USERNAME 2
#define MYSQL_CONF_PASSWORD 3

int get_mysql_conf(char *conf[])
{
  FILE *fp = fopen("../../src/bx-mysql/config/db.conf", "r");
  if (!fp) {
    return FALSE;
  }

  char buffer[1024];
  for (int i = 0; i < MYSQL_CONF_NUM; ++i) {
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
      return FALSE;
    }

    strtok(buffer, "\n");
    strtok(buffer, "\r");
    conf[i] = malloc(sizeof(char) * strlen(buffer) + 1);
    if (conf[i] == NULL) {
      return FALSE;
    }

    strcpy(conf[i], buffer);
  }

  fclose(fp);

  return TRUE;
}

const char *get_filename_ext(const char *filename)
{
  char *ext = strrchr(filename, '.');
  return (ext && ext != filename) ? ext+1 : "";
}

void db_export_address(word8 *addr_full, word8 *addr_hash, MYSQL *conn)
{
    // Create 32 byte hash of full address
    void* addr_full_raw = malloc(sizeof(byte) * TXADDRLEN);
    word8 addr_tag[TXTAGLEN];
    memcpy(addr_hash, addr_full, HASHLEN);
    memcpy(addr_full_raw, addr_full, TXADDRLEN);
    memcpy(addr_tag, addr_full + TXADDRLEN - TXTAGLEN, TXTAGLEN);

    // Call `address_insert` stored procedure
    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[3];
    my_bool is_null;
    my_bool is_null_tag;
    long unsigned int addr_hash_len = HASHLEN;
    long unsigned int addr_full_len = TXADDRLEN;
    long unsigned int TXTAGLEN  = TXTAGLEN;
    int status;

    stmt = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "CALL address_insert(?, ?, ?)", 28);
    memset(ps_params, 0, sizeof(ps_params));

    ps_params[0].buffer_type = MYSQL_TYPE_STRING;
    ps_params[0].buffer = addr_hash;
    ps_params[0].buffer_length = HASHLEN;
    ps_params[0].length = &addr_hash_len;
    ps_params[0].is_null = 0;

    ps_params[1].buffer_type = MYSQL_TYPE_BLOB;
    ps_params[1].buffer = addr_full_raw;
    ps_params[1].buffer_length = TXADDRLEN;
    ps_params[1].length = &addr_full_len;
    ps_params[1].is_null = 0;

    ps_params[2].buffer_type = MYSQL_TYPE_STRING;
    ps_params[2].buffer = addr_tag;
    ps_params[2].buffer_length = TXTAGLEN;
    ps_params[2].length = &TXTAGLEN;
    ps_params[2].is_null = &is_null_tag;

    is_null_tag = 0;
    if (addr_tag[0] == 0x42 || addr_tag[0] == 0x00) {
      is_null_tag = 1;
      TXTAGLEN = 0;
    }

    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);

    free(addr_full_raw);

    mysql_stmt_close(stmt);

}

void db_export_ledger_entry(LENTRY *le, word32 block_db_id, word32 block_num, MYSQL *conn)
{

// EXPORT address and get address hash
  word8 addr_hash[HASHLEN];
  db_export_address(le->addr, addr_hash, conn);

// Scrape Tag from Address
  void * le_addr_full = malloc(sizeof(byte) * TXADDRLEN);
  word8 le_addr_tag[TXTAGLEN];
  memcpy(le_addr_full, le->addr, TXADDRLEN);
  memcpy(le_addr_tag, le_addr_full + TXADDRLEN - TXTAGLEN, TXTAGLEN);
  free(le_addr_full);

  MYSQL_STMT *stmt;
  MYSQL_BIND ps_params[9];

  my_bool is_null = 0;

  long unsigned int hash_len = HASHLEN;
  long unsigned int addr_full_len = TXADDRLEN;
  long unsigned int addr_hash_len = HASHLEN;
  long unsigned int TXTAGLEN = TXTAGLEN;
  long unsigned int type_code = 9; /* Ledger Entry */

  word8 empty_hash[32];      /* All Zeroes Hash for Ledger Entries */
  word8 empty_tag[12];       /* All Zeroes DST Tag for Ledger Entries */

  memset(empty_hash, 0, 32);
  memset(empty_tag, 0, 12);
  
  int status;

  stmt = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL ledgersegment_insert(?, ?, ?, ?, ?, ?, ?, ?, ?)", 52);
  memset(ps_params, 0, sizeof(ps_params));

  ps_params[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params[0].buffer = &block_db_id;
  ps_params[0].is_null = &is_null;

  ps_params[1].buffer_type = MYSQL_TYPE_LONG;
  ps_params[1].buffer = &block_num;
  ps_params[1].is_null = &is_null;

  ps_params[2].buffer_type = MYSQL_TYPE_STRING;
  ps_params[2].buffer = (char *) empty_hash;
  ps_params[2].length = &hash_len;
  ps_params[2].is_null = &is_null;

  ps_params[3].buffer_type = MYSQL_TYPE_STRING;
  ps_params[3].buffer = addr_hash;
  ps_params[3].buffer_length = HASHLEN;
  ps_params[3].length = &addr_hash_len;
  ps_params[3].is_null = 0;

  ps_params[4].buffer_type = MYSQL_TYPE_STRING;
  ps_params[4].buffer = le_addr_tag;
  ps_params[4].length = &TXTAGLEN;
  ps_params[4].is_null = &is_null;
  
  ps_params[5].buffer_type = MYSQL_TYPE_LONG;
  ps_params[5].buffer = &type_code;
  ps_params[5].is_null = &is_null;

  ps_params[6].buffer_type = MYSQL_TYPE_STRING;
  ps_params[6].buffer = (char *) empty_hash;
  ps_params[6].length = &hash_len;
  ps_params[6].is_null = &is_null;

  ps_params[7].buffer_type = MYSQL_TYPE_STRING;
  ps_params[7].buffer = (char *) empty_tag;
  ps_params[7].length = &TXTAGLEN;
  ps_params[7].is_null = &is_null;

  ps_params[8].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[8].buffer = le->balance;
  ps_params[8].is_null = &is_null;

  status = mysql_stmt_bind_param(stmt, ps_params);
  status = mysql_stmt_execute(stmt);

  if(status) printf("%s\n", mysql_stmt_error(stmt));

  mysql_stmt_close(stmt);

}


void db_export_ledger(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_db_id, MYSQL *conn)
{
  word32 block_num = get32(bt->bnum);
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count = (header_len - sizeof(BHEADER)) / sizeof(LENTRY);
  int count;
  LENTRY le;

  fseek(fp, 4, SEEK_SET);
  for (word32 idx = 0; idx < tx_count; ++idx) {
    count = fread(&le, 1, sizeof(LENTRY), fp);
    if (count == sizeof(LENTRY)) {
      db_export_ledger_entry(&le, block_db_id, block_num, conn);
    } else {
      memset(&le, 0, sizeof(LENTRY));
    }
  }
}

void db_export_transaction_entry(TXQENTRY *txq, word32 block_db_id, word32 block_num, MYSQL *conn)
{
// Address hash buffers to pass to db_export_address()
  word8 src_addr_hash[HASHLEN];
  word8 dst_addr_hash[HASHLEN];
  word8 chg_addr_hash[HASHLEN];

// EXPORT address and hash pairs
  
  db_export_address(txq->src_addr, src_addr_hash, conn);
  db_export_address(txq->dst_addr, dst_addr_hash, conn);
  db_export_address(txq->chg_addr, chg_addr_hash, conn);

// Scrape Tags from Addresses
  void* src_addr_full = malloc(sizeof(byte) * TXADDRLEN);
  void* dst_addr_full = malloc(sizeof(byte) * TXADDRLEN);
  void* chg_addr_full = malloc(sizeof(byte) * TXADDRLEN);

  word8 src_addr_tag[TXTAGLEN];
  word8 dst_addr_tag[TXTAGLEN];
  word8 chg_addr_tag[TXTAGLEN];

  memcpy(src_addr_full, txq->src_addr, TXADDRLEN);
  memcpy(dst_addr_full, txq->dst_addr, TXADDRLEN);
  memcpy(chg_addr_full, txq->chg_addr, TXADDRLEN);

  memcpy(src_addr_tag, src_addr_full + TXADDRLEN - TXTAGLEN, TXTAGLEN);
  memcpy(dst_addr_tag, dst_addr_full + TXADDRLEN - TXTAGLEN, TXTAGLEN);
  memcpy(chg_addr_tag, chg_addr_full + TXADDRLEN - TXTAGLEN, TXTAGLEN);

  free(src_addr_full);
  free(dst_addr_full);
  free(chg_addr_full);

  int status;
  my_bool is_null;
  my_bool is_null_tag;
  long unsigned int hash_len = HASHLEN;
  long unsigned int signature_len = TXSIGLEN;
  long unsigned int addr_full_len = TXADDRLEN;
  long unsigned int addr_hash_len = HASHLEN;
  long unsigned int TXTAGLEN  = TXTAGLEN;


// Export Full Transaction Entry (type_code 0)

  MYSQL_STMT *stmt_0;
  MYSQL_BIND ps_params_0[12];

  word32 row_id = 0;
  word32 type_code = 0; /* Standard Full Transaction */

  stmt_0 = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt_0, "CALL transaction_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 59);
  memset(ps_params_0, 0, sizeof(ps_params_0));

  ps_params_0[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params_0[0].buffer = (char *)&block_db_id;
  ps_params_0[0].length = 0;
  ps_params_0[0].is_null = 0;

  ps_params_0[1].buffer_type = MYSQL_TYPE_STRING;
  ps_params_0[1].buffer = &(txq->tx_id);
  ps_params_0[1].buffer_length = HASHLEN;
  ps_params_0[1].length = &hash_len;
  ps_params_0[1].is_null = 0;

  ps_params_0[2].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_0[2].buffer = &src_addr_hash;  
  ps_params_0[2].buffer_length = HASHLEN;  
  ps_params_0[2].length = &addr_hash_len;  
  ps_params_0[2].is_null = 0;  

  ps_params_0[3].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_0[3].buffer = &dst_addr_hash;  
  ps_params_0[3].buffer_length = HASHLEN;  
  ps_params_0[3].length = &addr_hash_len;  
  ps_params_0[3].is_null = 0;  

  ps_params_0[4].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_0[4].buffer = &chg_addr_hash;  
  ps_params_0[4].buffer_length = HASHLEN;  
  ps_params_0[4].length = &addr_hash_len;  
  ps_params_0[4].is_null = 0;  

  ps_params_0[5].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_0[5].buffer = (char *)&(txq->send_total);
  ps_params_0[5].length = 0;
  ps_params_0[5].is_null = 0;

  ps_params_0[6].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_0[6].buffer = (char *)&(txq->change_total);
  ps_params_0[6].length = 0;
  ps_params_0[6].is_null = 0;

  ps_params_0[7].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_0[7].buffer = (char *)&(txq->tx_fee);
  ps_params_0[7].length = 0;
  ps_params_0[7].is_null = 0;

  ps_params_0[8].buffer_type = MYSQL_TYPE_STRING;
  ps_params_0[8].buffer = &(txq->tx_sig);
  ps_params_0[8].buffer_length = TXSIGLEN;
  ps_params_0[8].length = &signature_len;
  ps_params_0[8].is_null = 0;

  ps_params_0[9].buffer_type = MYSQL_TYPE_LONG;
  ps_params_0[9].buffer = (char *)&block_num;
  ps_params_0[9].length = 0;
  ps_params_0[9].is_null = 0;

  ps_params_0[10].buffer_type = MYSQL_TYPE_LONG;
  ps_params_0[10].buffer = (char *)&type_code;
  ps_params_0[10].length = 0;
  ps_params_0[10].is_null = 0;

  ps_params_0[11].buffer_type = MYSQL_TYPE_LONG;
  ps_params_0[11].buffer = (char *)&row_id;
  ps_params_0[11].length = 0;
  ps_params_0[11].is_null = 0;

  status = mysql_stmt_bind_param(stmt_0, ps_params_0);
  status = mysql_stmt_execute(stmt_0);

  mysql_stmt_close(stmt_0);


// Export Segment Entry: Receiving Coins (type_code 1)

  MYSQL_STMT *stmt_1;
  MYSQL_BIND ps_params_1[10];

  row_id = 0;
  type_code = 1; /* Address is Receiving Coins */

  stmt_1 = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt_1, "CALL txsegment_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 54);
  memset(ps_params_1, 0, sizeof(ps_params_1));

  ps_params_1[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params_1[0].buffer = (char *)&block_db_id;
  ps_params_1[0].length = 0;
  ps_params_1[0].is_null = 0;

  ps_params_1[1].buffer_type = MYSQL_TYPE_LONG;
  ps_params_1[1].buffer = (char *)&block_num;
  ps_params_1[1].length = 0;
  ps_params_1[1].is_null = 0;

  ps_params_1[2].buffer_type = MYSQL_TYPE_STRING;
  ps_params_1[2].buffer = &(txq->tx_id);
  ps_params_1[2].buffer_length = HASHLEN;
  ps_params_1[2].length = &hash_len;
  ps_params_1[2].is_null = 0;

  ps_params_1[3].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_1[3].buffer = &dst_addr_hash;  
  ps_params_1[3].buffer_length = HASHLEN;  
  ps_params_1[3].length = &addr_hash_len;  
  ps_params_1[3].is_null = 0;  

  ps_params_1[4].buffer_type = MYSQL_TYPE_STRING;
  ps_params_1[4].buffer = &dst_addr_tag;
  ps_params_1[4].buffer_length = TXTAGLEN;
  ps_params_1[4].length = &TXTAGLEN;
  ps_params_1[4].is_null = &is_null_tag;

  ps_params_1[5].buffer_type = MYSQL_TYPE_LONG;
  ps_params_1[5].buffer = (char *)&type_code;
  ps_params_1[5].length = 0;
  ps_params_1[5].is_null = 0;

  ps_params_1[6].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_1[6].buffer = &src_addr_hash;  
  ps_params_1[6].buffer_length = HASHLEN;  
  ps_params_1[6].length = &addr_hash_len;  
  ps_params_1[6].is_null = 0;  

  ps_params_1[7].buffer_type = MYSQL_TYPE_STRING;
  ps_params_1[7].buffer = &src_addr_tag;
  ps_params_1[7].buffer_length = TXTAGLEN;
  ps_params_1[7].length = &TXTAGLEN;
  ps_params_1[7].is_null = &is_null_tag;

  ps_params_1[8].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_1[8].buffer = (char *)&(txq->send_total);
  ps_params_1[8].length = 0;
  ps_params_1[8].is_null = 0;

  ps_params_1[9].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_1[9].buffer = (char *)&(txq->tx_fee);
  ps_params_1[9].length = 0;
  ps_params_1[9].is_null = 0;

  status = mysql_stmt_bind_param(stmt_1, ps_params_1);
  status = mysql_stmt_execute(stmt_1);

  if(status) printf("%s\n", mysql_stmt_error(stmt_1));

  mysql_stmt_close(stmt_1);


// Export Segment Entry: Sending Coins (type_code 2)

  MYSQL_STMT *stmt_2;
  MYSQL_BIND ps_params_2[10];

  row_id = 0;
  type_code = 2; /* Address is Sending Coins */

  stmt_2 = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt_1, "CALL txsegment_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 54);
  memset(ps_params_2, 0, sizeof(ps_params_2));

  ps_params_2[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params_2[0].buffer = (char *)&block_db_id;
  ps_params_2[0].length = 0;
  ps_params_2[0].is_null = 0;

  ps_params_2[1].buffer_type = MYSQL_TYPE_LONG;
  ps_params_2[1].buffer = (char *)&block_num;
  ps_params_2[1].length = 0;
  ps_params_2[1].is_null = 0;

  ps_params_2[2].buffer_type = MYSQL_TYPE_STRING;
  ps_params_2[2].buffer = &(txq->tx_id);
  ps_params_2[2].buffer_length = HASHLEN;
  ps_params_2[2].length = &hash_len;
  ps_params_2[2].is_null = 0;

  ps_params_2[3].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_2[3].buffer = &src_addr_hash;  
  ps_params_2[3].buffer_length = HASHLEN;  
  ps_params_2[3].length = &addr_hash_len;  
  ps_params_2[3].is_null = 0;  

  ps_params_2[4].buffer_type = MYSQL_TYPE_STRING;
  ps_params_2[4].buffer = &src_addr_tag;
  ps_params_2[4].buffer_length = TXTAGLEN;
  ps_params_2[4].length = &TXTAGLEN;
  ps_params_2[4].is_null = &is_null_tag;

  ps_params_2[5].buffer_type = MYSQL_TYPE_LONG;
  ps_params_2[5].buffer = (char *)&type_code;
  ps_params_2[5].length = 0;
  ps_params_2[5].is_null = 0;

  ps_params_2[6].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_2[6].buffer = &dst_addr_hash;  
  ps_params_2[6].buffer_length = HASHLEN;  
  ps_params_2[6].length = &addr_hash_len;  
  ps_params_2[6].is_null = 0;  

  ps_params_2[7].buffer_type = MYSQL_TYPE_STRING;
  ps_params_2[7].buffer = &dst_addr_tag;
  ps_params_2[7].buffer_length = TXTAGLEN;
  ps_params_2[7].length = &TXTAGLEN;
  ps_params_2[7].is_null = &is_null_tag;

  ps_params_2[8].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_2[8].buffer = (char *)&(txq->send_total);
  ps_params_2[8].length = 0;
  ps_params_2[8].is_null = 0;

  ps_params_2[9].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_2[9].buffer = (char *)&(txq->tx_fee);
  ps_params_2[9].length = 0;
  ps_params_2[9].is_null = 0;

  status = mysql_stmt_bind_param(stmt_2, ps_params_2);
  status = mysql_stmt_execute(stmt_2);

  if(status) printf("%s\n", mysql_stmt_error(stmt_2));

  mysql_stmt_close(stmt_2);

// Export Segment Entry: Receiving Change (type_code 3)

  MYSQL_STMT *stmt_3;
  MYSQL_BIND ps_params_3[10];

  row_id = 0;
  type_code = 3; /* Address is Receiving Change */

  stmt_3 = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt_1, "CALL txsegment_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 54);
  memset(ps_params_3, 0, sizeof(ps_params_3));

  ps_params_3[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params_3[0].buffer = (char *)&block_db_id;
  ps_params_3[0].length = 0;
  ps_params_3[0].is_null = 0;

  ps_params_3[1].buffer_type = MYSQL_TYPE_LONG;
  ps_params_3[1].buffer = (char *)&block_num;
  ps_params_3[1].length = 0;
  ps_params_3[1].is_null = 0;

  ps_params_3[2].buffer_type = MYSQL_TYPE_STRING;
  ps_params_3[2].buffer = &(txq->tx_id);
  ps_params_3[2].buffer_length = HASHLEN;
  ps_params_3[2].length = &hash_len;
  ps_params_3[2].is_null = 0;

  ps_params_3[3].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_3[3].buffer = &chg_addr_hash;  
  ps_params_3[3].buffer_length = HASHLEN;  
  ps_params_3[3].length = &addr_hash_len;  
  ps_params_3[3].is_null = 0;  

  ps_params_3[4].buffer_type = MYSQL_TYPE_STRING;
  ps_params_3[4].buffer = &chg_addr_tag;
  ps_params_3[4].buffer_length = TXTAGLEN;
  ps_params_3[4].length = &TXTAGLEN;
  ps_params_3[4].is_null = &is_null_tag;

  ps_params_3[5].buffer_type = MYSQL_TYPE_LONG;
  ps_params_3[5].buffer = (char *)&type_code;
  ps_params_3[5].length = 0;
  ps_params_3[5].is_null = 0;

  ps_params_3[6].buffer_type = MYSQL_TYPE_STRING;    
  ps_params_3[6].buffer = &src_addr_hash;  
  ps_params_3[6].buffer_length = HASHLEN;  
  ps_params_3[6].length = &addr_hash_len;  
  ps_params_3[6].is_null = 0;  

  ps_params_3[7].buffer_type = MYSQL_TYPE_STRING;
  ps_params_3[7].buffer = &src_addr_tag;
  ps_params_3[7].buffer_length = TXTAGLEN;
  ps_params_3[7].length = &TXTAGLEN;
  ps_params_3[7].is_null = &is_null_tag;

  ps_params_3[8].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_3[8].buffer = (char *)&(txq->change_total);
  ps_params_3[8].length = 0;
  ps_params_3[8].is_null = 0;

  ps_params_3[9].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params_3[9].buffer = (char *)&(txq->tx_fee);
  ps_params_3[9].length = 0;
  ps_params_3[9].is_null = 0;

  status = mysql_stmt_bind_param(stmt_3, ps_params_3);
  status = mysql_stmt_execute(stmt_3);

  if(status) printf("%s\n", mysql_stmt_error(stmt_3));

  mysql_stmt_close(stmt_3);

}

void db_export_transactions(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_db_id, MYSQL *conn)
{
  word32 block_num = get32(bt->bnum);
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count = get32(bt->tcount);
  int count;
  TXQENTRY txq;

  fseek(fp, header_len, SEEK_SET);
  for (word32 idx = 0; idx < tx_count; ++idx) {
    count = fread(&txq, 1, sizeof(TXQENTRY), fp);
    if (count == sizeof(TXQENTRY)) {
      db_export_transaction_entry(&txq, block_db_id, block_num, conn);
    } else {
      memset(&txq, 0, sizeof(TXQENTRY));
    }
  }
}

void obsolete_ledger_entries(MYSQL *conn)
{

    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[1];

    int status;

    stmt = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "CALL obsolete_ledger_entries()", 30);
    memset(ps_params, 0, sizeof(ps_params));
    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);
}


void db_export_entries(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_db_id, MYSQL *conn)
{
  word32 block_num = get32(bt->bnum);
  my_bool neogenesis = (block_num % 256 == 0);
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count = get32(bt->tcount);

  // Ledger entries or transaction entries
  if (neogenesis) { // ledger entries
    obsolete_ledger_entries(conn); /* Set all type_code 9 to 99 */
    db_export_ledger(bh, bt, fp, block_db_id, conn);
  } else { // transaction entries
    db_export_transactions(bh, bt, fp, block_db_id, conn);
  }
}

int check_block_exists(BTRAILER *bt, MYSQL *conn)
{
    printf("Hash: 0x"); bytes2hex(bt->bhash, HASHLEN); 

    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[1];
    my_bool is_null;
    long unsigned int hash_len = HASHLEN;
    word32 row_id = 0;
    int status;

    stmt = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "SELECT id FROM block WHERE hash = ? AND main_chain = 1", 54);
    memset(ps_params, 0, sizeof(ps_params));

    ps_params[0].buffer_type = MYSQL_TYPE_STRING;
    ps_params[0].buffer = &(bt->bhash);
    ps_params[0].buffer_length = HASHLEN;
    ps_params[0].length = &hash_len;
    ps_params[0].is_null = 0;

    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);

// Set output row_id, clear cursor
    do {
      MYSQL_FIELD *fields;
      MYSQL_BIND *rs_bind;

      if (mysql_stmt_field_count(stmt) > 0) {
        MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
        fields = mysql_fetch_fields(rs_metadata);
        rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
        memset(rs_bind, 0, sizeof (MYSQL_BIND));

        rs_bind[0].buffer_type = fields[0].type;
        rs_bind[0].is_null = &is_null;
        rs_bind[0].buffer = (char *)&row_id;
        rs_bind[0].buffer_length = sizeof(row_id);

        status = mysql_stmt_bind_result(stmt, rs_bind);
        while (1) {
          status = mysql_stmt_fetch(stmt);
          if (status == 1 || status == MYSQL_NO_DATA)
            break;
        }

        mysql_free_result(rs_metadata);
        free(rs_bind);
        fields = NULL;
      }

      status = mysql_stmt_next_result(stmt);
    } while (status == 0);

    mysql_stmt_close(stmt);

    if (row_id == 0) {
      return 0;
    }

    return 1;
}

void remove_block(word32 block_num, MYSQL *conn)
{

    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[1];

    int status;

    stmt = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "CALL remove_block(?)", 20);
    memset(ps_params, 0, sizeof(ps_params));

    ps_params[0].buffer_type = MYSQL_TYPE_LONG;
    ps_params[0].buffer = (char *)&block_num;
    ps_params[0].length = 0;
    ps_params[0].is_null = 0;

    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);
}


word32 db_export_block(BHEADER *bh, BTRAILER *bt, MYSQL *conn)
{
  word32 block_num = get32(bt->bnum);
  my_bool neogenesis = (block_num % 256 == 0);
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count = get32(bt->tcount);
  word32 difficulty = (word32)bt->difficulty[0];
  word32 solve_time = get32(bt->stime);
  word32 next_time = get32(bt->time0);
  char haiku[256];
  
  trigg_expand2(bt->nonce, haiku);


  if (neogenesis) {
    tx_count = (header_len - sizeof(BHEADER)) / sizeof(LENTRY); // # ledger entries
  }

  if (check_block_exists(bt, conn)) {
    printf("  Block on disk has same hash.  Skipping. %d.\n", block_num);
    return 0; /* No further processing required */
  }
  remove_block(block_num, conn);  /* If the block is in the DB, wipe it. */
  // Export the miner address and get hash
  word8 miner_addr_hash[HASHLEN];
  db_export_address(bh->maddr, miner_addr_hash, conn);

  // Export block entry
  // Call `block_insert` stored procedure
    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[16];
    my_bool is_null;
    long unsigned int addr_hash_len = HASHLEN;
    long unsigned int hash_len = HASHLEN;
    long unsigned int haiku_len = strlen(haiku);
    word32 row_id = 0;
    int status;

    stmt = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "CALL block_insert(?, 0, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 71);
    memset(ps_params, 0, sizeof(ps_params));

    ps_params[0].buffer_type = MYSQL_TYPE_LONG;
    ps_params[0].buffer = (char *)&block_num;
    ps_params[0].length = 0;
    ps_params[0].is_null = 0;

    ps_params[1].buffer_type = MYSQL_TYPE_TINY;
    ps_params[1].buffer = (char *)&neogenesis;
    ps_params[1].length = 0;
    ps_params[1].is_null = 0;

    ps_params[2].buffer_type = MYSQL_TYPE_LONG;
    ps_params[2].buffer = (char *)&header_len;
    ps_params[2].length = 0;
    ps_params[2].is_null = 0;

    ps_params[3].buffer_type = MYSQL_TYPE_STRING;
    ps_params[3].buffer = &(bt->bhash);
    ps_params[3].buffer_length = HASHLEN;
    ps_params[3].length = &hash_len;
    ps_params[3].is_null = 0;

    ps_params[4].buffer_type = MYSQL_TYPE_STRING;
    ps_params[4].buffer = &(bt->phash);
    ps_params[4].buffer_length = HASHLEN;
    ps_params[4].length = &hash_len;
    ps_params[4].is_null = 0;

    ps_params[5].buffer_type = MYSQL_TYPE_STRING;
    ps_params[5].buffer = &(bt->nonce);
    ps_params[5].buffer_length = HASHLEN;
    ps_params[5].length = &hash_len;
    ps_params[5].is_null = 0;

    ps_params[6].buffer_type = MYSQL_TYPE_STRING;
    ps_params[6].buffer = &(bt->mroot);
    ps_params[6].buffer_length = HASHLEN;
    ps_params[6].length = &hash_len;
    ps_params[6].is_null = 0;

    ps_params[7].buffer_type = MYSQL_TYPE_LONG;
    ps_params[7].buffer = (char *)&tx_count;
    ps_params[7].length = 0;
    ps_params[7].is_null = 0;

    ps_params[8].buffer_type = MYSQL_TYPE_STRING;    
    ps_params[8].buffer = &miner_addr_hash;  
    ps_params[8].buffer_length = HASHLEN;  
    ps_params[8].length = &addr_hash_len;  
    ps_params[8].is_null = 0;

    ps_params[9].buffer_type = MYSQL_TYPE_LONGLONG;
    ps_params[9].buffer = (char *)&(bh->mreward);
    ps_params[9].length = 0;
    ps_params[9].is_null = 0;

    ps_params[10].buffer_type = MYSQL_TYPE_LONGLONG;
    ps_params[10].buffer = (char *)&(bt->mfee);
    ps_params[10].length = 0;
    ps_params[10].is_null = 0;

    ps_params[11].buffer_type = MYSQL_TYPE_LONG;
    ps_params[11].buffer = (char *)&difficulty;
    ps_params[11].length = 0;
    ps_params[11].is_null = 0;

    ps_params[12].buffer_type = MYSQL_TYPE_LONG;
    ps_params[12].buffer = (char *)&solve_time;
    ps_params[12].length = 0;
    ps_params[12].is_null = 0;

    ps_params[13].buffer_type = MYSQL_TYPE_LONG;
    ps_params[13].buffer = (char *)&next_time;
    ps_params[13].length = 0;
    ps_params[13].is_null = 0;

    ps_params[14].buffer_type = MYSQL_TYPE_STRING;
    ps_params[14].buffer = haiku;
    ps_params[14].buffer_length = haiku_len;
    ps_params[14].length = 0;
    ps_params[14].is_null = 0;

    ps_params[15].buffer_type = MYSQL_TYPE_LONG;
    ps_params[15].buffer = (char *)&row_id;
    ps_params[15].length = 0;
    ps_params[15].is_null = 0;

    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);

    // Set output row_id, clear cursor
    do {
      MYSQL_FIELD *fields;
      MYSQL_BIND *rs_bind;

      if (mysql_stmt_field_count(stmt) > 0) {
        MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
        fields = mysql_fetch_fields(rs_metadata);
        rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
        memset(rs_bind, 0, sizeof (MYSQL_BIND));

        rs_bind[0].buffer_type = fields[0].type;
        rs_bind[0].is_null = &is_null;
        rs_bind[0].buffer = (char *)&row_id;
        rs_bind[0].buffer_length = sizeof(row_id);

        status = mysql_stmt_bind_result(stmt, rs_bind);
        while (1) {
          status = mysql_stmt_fetch(stmt);
          if (status == 1 || status == MYSQL_NO_DATA)
            break;
        }

        mysql_free_result(rs_metadata);
        free(rs_bind);
        fields = NULL;
      }

      status = mysql_stmt_next_result(stmt);
    } while (status == 0);

    mysql_stmt_close(stmt);

    if (row_id == 0 || row_id == -1) {
      return 0;
    }

    if(neogenesis) return row_id; /* No Further Processing */

// Export Segment Entry: Receiving Block Reward (type_code 4)

    MYSQL_STMT *stmt_4;
    MYSQL_BIND ps_params_4[10];

    word32 type_code = 4; /* Address is Receiving a Block Reward */

    my_bool is_null_tag = 1;
    long unsigned int TXTAGLEN  = TXTAGLEN;

    word32 zero = 0;
    word8 empty_tag[12];
    word8 empty_hash[32];

    memset(empty_tag, 0, 12);
    memset(empty_hash, 0, 32);

    stmt_4 = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt_4, "CALL txsegment_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 54);
    memset(ps_params_4, 0, sizeof(ps_params_4));

    ps_params_4[0].buffer_type = MYSQL_TYPE_LONG;
    ps_params_4[0].buffer = (char *)&row_id; /* From Block Export Above */
    ps_params_4[0].length = 0;
    ps_params_4[0].is_null = 0;

    ps_params_4[1].buffer_type = MYSQL_TYPE_LONG;
    ps_params_4[1].buffer = (char *)&block_num;
    ps_params_4[1].length = 0;
    ps_params_4[1].is_null = 0;

    ps_params_4[2].buffer_type = MYSQL_TYPE_STRING;
    ps_params_4[2].buffer = (char *)&empty_hash;
    ps_params_4[2].buffer_length = HASHLEN;
    ps_params_4[2].length = &hash_len;
    ps_params_4[2].is_null = 0;

    ps_params_4[3].buffer_type = MYSQL_TYPE_STRING;    
    ps_params_4[3].buffer = &miner_addr_hash;  
    ps_params_4[3].buffer_length = HASHLEN;  
    ps_params_4[3].length = &addr_hash_len;  
    ps_params_4[3].is_null = 0;

    ps_params_4[4].buffer_type = MYSQL_TYPE_STRING;
    ps_params_4[4].buffer = &empty_tag;
    ps_params_4[4].buffer_length = TXTAGLEN;
    ps_params_4[4].length = &TXTAGLEN;
    ps_params_4[4].is_null = &is_null_tag;

    ps_params_4[5].buffer_type = MYSQL_TYPE_LONG;
    ps_params_4[5].buffer = (char *)&type_code;
    ps_params_4[5].length = 0;
    ps_params_4[5].is_null = 0;

    ps_params_4[6].buffer_type = MYSQL_TYPE_STRING;
    ps_params_4[6].buffer = (char *)&empty_hash;
    ps_params_4[6].buffer_length = HASHLEN;
    ps_params_4[6].length = &hash_len;
    ps_params_4[6].is_null = 0;

    ps_params_4[7].buffer_type = MYSQL_TYPE_STRING;
    ps_params_4[7].buffer = &empty_tag; /* No Tag for Fund Source Either */
    ps_params_4[7].buffer_length = TXTAGLEN;
    ps_params_4[7].length = &TXTAGLEN;
    ps_params_4[7].is_null = &is_null_tag;

    ps_params_4[8].buffer_type = MYSQL_TYPE_LONGLONG;
    ps_params_4[8].buffer = (char *)&(bh->mreward);
    ps_params_4[8].length = 0;
    ps_params_4[8].is_null = 0;

    ps_params_4[9].buffer_type = MYSQL_TYPE_LONGLONG;
    ps_params_4[9].buffer = "0";
    ps_params_4[9].length = 0;
    ps_params_4[9].is_null = 0;

    status = mysql_stmt_bind_param(stmt_4, ps_params_4);
    status = mysql_stmt_execute(stmt_4);

    if(status) printf("%s\n", mysql_stmt_error(stmt_4));

    mysql_stmt_close(stmt_4);

    return row_id;
}

void export_block(char *filename, MYSQL *conn)
{
  FILE *fp;
  BHEADER bh;
  BTRAILER bt;
  word32 header_len;
  int count;

  printf("Exporting: %s\n", filename);

  // Open block file
  fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("  ERROR: Could not open block file: %s\n", filename);
    return;
  }

  // Get header length
  count = fread(&header_len, 1, 4, fp);
  if (count != 4) {
    printf("  ERROR: Unable to read header length\n");
err:
    fclose(fp);
    return;
  }

  // Read header
  memset(&bh, 0, sizeof(BHEADER));
  put32(bh.hdrlen, header_len);
  if (header_len == sizeof(BHEADER)) {
    fseek(fp, 0, SEEK_SET);
    if (fread(&bh, 1, sizeof(BHEADER), fp) != sizeof(BHEADER)) {
      printf("  ERROR: Unable to read header contents\n");
      goto err;
    }
  }

  // Read trailer
  if ((fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) ||
      (fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER))) {
    printf("  ERROR: Unable to read trailer contents\n");
    goto err;
  }

  // DB EXPORT Block contents
  word32 row_id = db_export_block(&bh, &bt, conn);
  if (row_id == 0) {
    printf("  ERROR: Failed to export block: %d\n", get32(bt.bnum));
  } else {
    // DB EXPORT Block ledger entries (neogenesis) or transactions (regular)
    db_export_entries(&bh, &bt, fp, row_id, conn);
  }

  // Close the block file
  fclose(fp);
}

int cstring_cmp(const void *a, const void *b) 
{ 
    if (a == b) return 0;

    return strcmp(b, a);
} 

int export(char *path)
{
  // Set path for block files
  if (path == NULL || strlen(path) == 0) {
    path = "./bc/";
  }

  printf("Exporting block files from: %s\n", path);

  // Open a database connection
  printf("MySQL client version: %s\n", mysql_get_client_info());
  MYSQL *conn = mysql_init(NULL);
  if (conn == NULL) {
    printf("ERROR: Could not create database connection\n");
    return 1;
  }

  char *conf[MYSQL_CONF_NUM];
  if (!get_mysql_conf(conf)) {
    printf("ERROR: Could not load database configuration file\n");
    return 1;
  }

  if (mysql_real_connect(conn, conf[MYSQL_CONF_HOSTNAME], conf[MYSQL_CONF_USERNAME], conf[MYSQL_CONF_PASSWORD], conf[MYSQL_CONF_DATABASE], 0, NULL, 0) == NULL) {
    printf("ERROR: Could not connect to the `mochimo_bx` database\n");
    return 1;
  }

  // Iterate over all available block files and export to the database
  char block_file_paths[32768][255];

  DIR *dir = opendir(path);
  int index = 0;
  for (int i = 0; i < 32768; i++) {
    strcpy(block_file_paths[i], "");
  }

  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
      char *filename = entry->d_name;
      char *filepath = malloc(strlen(path)+strlen(filename)+1);
      strcpy(filepath, path);
      strcat(filepath, filename);
      if (strncmp(get_filename_ext(filename), "bc", 3) == 0 && filename[0] == 'b') {
        strcpy(block_file_paths[index], filepath);
        index++;
        //export_block(filepath, conn);
      }
    }

    entry = NULL;
    closedir(dir);
  }

  qsort((void *)block_file_paths, sizeof(block_file_paths) / sizeof(block_file_paths[0]), sizeof(block_file_paths[0]), cstring_cmp);

  for (int i = 0; i < index; i++)
  {
    export_block(block_file_paths[i], conn);
  } 

  // Call `cache_update` stored procedure
  MYSQL_STMT *stmt;
  int status;

  printf("Export completing...\n");

  mysql_close(conn);

  return 0;
}
