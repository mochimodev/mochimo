/**
 * bx_mysql_export.c  Block Explorer - MySQL Database Export Feature
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * \author Tim Cotten <tcotten@mochimo.org> <tim@cotten.io>
 * \date 2018-06-20
 * \copyright Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 */

#define MYSQL_CONF_NUM 4
#define MYSQL_CONF_HOSTNAME 0
#define MYSQL_CONF_DATABASE 1
#define MYSQL_CONF_USERNAME 2
#define MYSQL_CONF_PASSWORD 3

/**
 * Helper func: get the filename extension if it has one
 */
const char *get_filename_ext(const char *filename)
{
  char *ext = strrchr(filename, '.');
  return (ext && ext != filename) ? ext+1 : "";
}

/**
 * Helper func: string comparison for reverse sorting
 */
int cstring_cmp(const void *a, const void *b) 
{ 
    if (a == b) return 0;

    return strcmp(b, a);
} 


/**
 * Requires a `config` folder and accompanying db.conf file at the same level
 * folder hierarchy level as `bin`
 */
int get_mysql_conf(char *conf[])
{
  // Exit if the config file isn't found
  FILE *fp = fopen("../../config/db.conf", "r");
  if (!fp) {
    return FALSE;
  }

  // Simple config file: each line represents a connection parameter
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

/**
 * Export an address (assume existence check already made)
 * Stored Procedure: address_insert(IN hash, IN full, IN tag, OUT insert_id)
 *
 * Returns row_id of inserted address
 */
word32 db_export_address(byte *addr_full, MYSQL *conn)
{
  byte  addr_hash[HASHLEN];                               // First 32 bytes of address
  void* addr_full_raw = malloc(sizeof(byte) * TXADDRLEN); // 2208 byte WOTS+, MYSQL_TYPE_BLOB 
  byte  addr_tag[ADDR_TAG_LEN];                           // 12 byte tag
  memcpy(addr_hash,     addr_full, HASHLEN);
  memcpy(addr_full_raw, addr_full, TXADDRLEN);
  memcpy(addr_tag,      addr_full + (TXADDRLEN - ADDR_TAG_LEN), ADDR_TAG_LEN);

  // Prepare `address_insert` parameters and bind them by reference
  MYSQL_STMT *stmt;
  MYSQL_BIND ps_params[4];
  my_bool    is_null;
  my_bool    is_null_tag = 0;
  word32     row_id = 0;
  int        status;
  long unsigned int addr_hash_len = HASHLEN;
  long unsigned int addr_full_len = TXADDRLEN;
  long unsigned int addr_tag_len  = ADDR_TAG_LEN;

  stmt   = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL address_insert(?, ?, ?, ?)", 31);
  memset(ps_params, 0, sizeof(ps_params));

  // IN: First 32 bytes of WOTS+ address
  ps_params[0].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[0].buffer_length = HASHLEN;
  ps_params[0].length        = &addr_hash_len;
  ps_params[0].buffer        = &addr_hash;
  ps_params[0].is_null       = 0;

  // IN: Full WOTS+ address
  ps_params[1].buffer_type   = MYSQL_TYPE_BLOB;
  ps_params[1].buffer_length = TXADDRLEN;
  ps_params[1].length        = &addr_full_len;
  ps_params[1].buffer        = addr_full_raw;
  ps_params[1].is_null       = 0;

  // IN: Tag
  ps_params[2].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[2].buffer_length = ADDR_TAG_LEN;
  ps_params[2].length        = &addr_tag_len;
  ps_params[2].buffer        = &addr_tag;
  ps_params[2].is_null       = &is_null_tag;

  // Tags that start with a 0x42 or 0x00 byte are NOT valid/registered tags
  if (addr_tag[0] == 0x42 || addr_tag[0] == 0x00) {
    is_null_tag = 1;
    addr_tag_len = 0;
  }

  // OUT: inserted row_id
  ps_params[3].buffer_type = MYSQL_TYPE_LONG;
  ps_params[3].length      = 0;
  ps_params[3].buffer      = (char *)&row_id;
  ps_params[3].is_null     = 0;

  // Execute
  status = mysql_stmt_bind_param(stmt, ps_params);
  status = mysql_stmt_execute(stmt);

  free(addr_full_raw); // clean up the void pointer

  // Results
  do {
    MYSQL_FIELD *fields;
    MYSQL_BIND  *rs_bind;

    if (mysql_stmt_field_count(stmt) > 0) {
      MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
      fields  = mysql_fetch_fields(rs_metadata);
      rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
      memset(rs_bind, 0, sizeof (MYSQL_BIND));

      rs_bind[0].buffer_type   = fields[0].type;
      rs_bind[0].is_null       = &is_null;
      rs_bind[0].buffer        = (char *)&row_id;
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

  return row_id;
}

/**
 * Export a ledger entry (assume existence check already made)
 * Stored Procedure: ledger_insert(IN block_id, IN addr_id, IN balance, OUT insert_id)
 *
 * Ledger entries are stored as unique indexes on [block_id, addr_id]
 */

void db_export_ledger_entry(LENTRY *le, word32 block_id, MYSQL *conn)
{
  // Export address or get existing row_id
  word32 addr_id = db_export_address(le->addr, conn);

  if (addr_id == 0) {
    printf("  ERROR: Could not create ledger entry.\n");
    return;
  }

  // Prepare `ledger_insert` parameters and bind them by reference
  MYSQL_STMT *stmt;
  MYSQL_BIND ps_params[4];
  my_bool    is_null;
  word32     row_id = 0;
  int        status;

  stmt   = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL ledger_insert(?, ?, ?, ?)", 30);
  memset(ps_params, 0, sizeof(ps_params));

  // IN: Block ID
  ps_params[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params[0].length      = 0;
  ps_params[0].buffer      = (char *)&block_id;
  ps_params[0].is_null     = 0;

  // IN: Address ID
  ps_params[1].buffer_type = MYSQL_TYPE_LONG;
  ps_params[1].length      = 0;
  ps_params[1].buffer      = (char *)&addr_id;
  ps_params[1].is_null     = 0;

  // IN: Balance
  ps_params[2].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[2].length      = 0;
  ps_params[2].buffer      = (char *)&(le->balance);
  ps_params[2].is_null     = 0;

  // OUT: inserted row_id
  ps_params[3].buffer_type = MYSQL_TYPE_LONG;
  ps_params[3].length      = 0;
  ps_params[3].buffer      = (char *)&row_id;
  ps_params[3].is_null     = 0;

  // Execute
  status = mysql_stmt_bind_param(stmt, ps_params);
  status = mysql_stmt_execute(stmt);

  // Results
  do {
    MYSQL_FIELD *fields;
    MYSQL_BIND  *rs_bind;

    if (mysql_stmt_field_count(stmt) > 0) {
      MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
      fields  = mysql_fetch_fields(rs_metadata);
      rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
      memset(rs_bind, 0, sizeof (MYSQL_BIND));

      rs_bind[0].buffer_type   = fields[0].type;
      rs_bind[0].is_null       = &is_null;
      rs_bind[0].buffer        = (char *)&row_id;
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
}

/**
 * Export the ledger of a neogenesis (aeon) block containing the amounts of all
 * non-zero balance addresses/tags
 */
void db_export_ledger(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_id, MYSQL *conn)
{
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count = (header_len - sizeof(BHEADER)) / sizeof(LENTRY);
  int count;
  LENTRY le;

  // Iterate over all ledger entries in the neogenesis block
  fseek(fp, 4, SEEK_SET);
  for (word32 idx = 0; idx < tx_count; ++idx) {
    count = fread(&le, 1, sizeof(LENTRY), fp);
    if (count == sizeof(LENTRY)) {
      db_export_ledger_entry(&le, block_id, conn);
    } else {
      memset(&le, 0, sizeof(LENTRY));
    }
  }
}

/**
 * Export a transaction entry
 * Stored Procedure: transaction_insert(IN block_id,
 *                                      IN tx_id,
 *                                      IN src_addr_id, IN dst_addr_id, IN chg_addr_id,
 *                                      IN send_total, IN chg_total, IN tx_fee,
 *                                      IN tx_sig,
 *                                      OUT insert_id)
 *
 * Transaction entries are stored as unique indexes on [block_id, tx_id]
 */
void db_export_transaction_entry(TXQENTRY *txq, word32 block_id, MYSQL *conn)
{
  // Export addresses or get row_id if they already exist
  word32 src_addr_id = db_export_address(txq->src_addr, conn);
  word32 dst_addr_id = db_export_address(txq->dst_addr, conn);
  word32 chg_addr_id = db_export_address(txq->chg_addr, conn);

  if (src_addr_id == 0 || dst_addr_id == 0 || chg_addr_id == 0) {
    printf("  ERROR: Could not create transaction entry.\n");
    return;
  }

  // Prepare `ledger_insert` parameters and bind them by reference
  MYSQL_STMT *stmt;
  MYSQL_BIND ps_params[10];
  my_bool    is_null;
  word32     row_id = 0;
  int        status;
  long unsigned int hash_len      = HASHLEN;
  long unsigned int signature_len = TXSIGLEN;

  stmt   = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL transaction_insert(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 53);
  memset(ps_params, 0, sizeof(ps_params));

  ps_params[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params[0].length      = 0;
  ps_params[0].buffer      = (char *)&block_id;
  ps_params[0].is_null     = 0;

  ps_params[1].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[1].buffer_length = HASHLEN;
  ps_params[1].length        = &hash_len;
  ps_params[1].buffer        = &(txq->tx_id);
  ps_params[1].is_null       = 0;

  ps_params[2].buffer_type = MYSQL_TYPE_LONG;
  ps_params[2].length      = 0;
  ps_params[2].buffer      = (char *)&src_addr_id;
  ps_params[2].is_null     = 0;

  ps_params[3].buffer_type = MYSQL_TYPE_LONG;
  ps_params[3].length      = 0;
  ps_params[3].buffer      = (char *)&dst_addr_id;
  ps_params[3].is_null     = 0;

  ps_params[4].buffer_type = MYSQL_TYPE_LONG;
  ps_params[4].length      = 0;
  ps_params[4].buffer      = (char *)&chg_addr_id;
  ps_params[4].is_null     = 0;

  ps_params[5].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[5].length      = 0;
  ps_params[5].buffer      = (char *)&(txq->send_total);
  ps_params[5].is_null     = 0;

  ps_params[6].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[6].length      = 0;
  ps_params[6].buffer      = (char *)&(txq->change_total);
  ps_params[6].is_null     = 0;

  ps_params[7].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[7].length      = 0;
  ps_params[7].buffer      = (char *)&(txq->tx_fee);
  ps_params[7].is_null     = 0;

  ps_params[8].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[8].buffer_length = TXSIGLEN;
  ps_params[8].length        = &signature_len;
  ps_params[8].buffer        = &(txq->tx_sig);
  ps_params[8].is_null       = 0;

  ps_params[9].buffer_type = MYSQL_TYPE_LONG;
  ps_params[9].length      = 0;
  ps_params[9].buffer      = (char *)&row_id;
  ps_params[9].is_null     = 0;

  // Execute
  status = mysql_stmt_bind_param(stmt, ps_params);
  status = mysql_stmt_execute(stmt);

  // Results
  do {
    MYSQL_FIELD *fields;
    MYSQL_BIND  *rs_bind;

    if (mysql_stmt_field_count(stmt) > 0) {
      MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
      fields  = mysql_fetch_fields(rs_metadata);
      rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
      memset(rs_bind, 0, sizeof (MYSQL_BIND));

      rs_bind[0].buffer_type   = fields[0].type;
      rs_bind[0].is_null       = &is_null;
      rs_bind[0].buffer        = (char *)&row_id;
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
}

/**
 * Export all of the transactions of a given block
 */
void db_export_transactions(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_id, MYSQL *conn)
{
  word32 header_len = get32(bh->hdrlen);
  word32 tx_count   = get32(bt->tcount);
  int count;
  TXQENTRY txq;

  // Iterate over all transactions (these are found after the block header)
  fseek(fp, header_len, SEEK_SET);
  for (word32 idx = 0; idx < tx_count; ++idx) {
    count = fread(&txq, 1, sizeof(TXQENTRY), fp);
    if (count == sizeof(TXQENTRY)) {
      db_export_transaction_entry(&txq, block_id, conn);
    } else {
      memset(&txq, 0, sizeof(TXQENTRY));
    }
  }
}

/**
 * Export either the transactions or ledger entries of a given block
 */
void db_export_entries(BHEADER *bh, BTRAILER *bt, FILE *fp, word32 block_id, MYSQL *conn)
{
  word32 block_num   = get32(bt->bnum);
  my_bool neogenesis = (block_num % 256 == 0);
  word32 header_len  = get32(bh->hdrlen);
  word32 tx_count    = get32(bt->tcount);

  // Ledger entries or transaction entries
  if (neogenesis) { // ledger entries
    db_export_ledger(bh, bt, fp, block_id, conn);
  } else { // transaction entries
    db_export_transactions(bh, bt, fp, block_id, conn);
  }
}

/**
 * Discover whether a given block already exists in the db with the same hash
 */
int check_block_exists(BTRAILER *bt, MYSQL *conn)
{
    printf("Hash: 0x"); bytes2hex(bt->bhash, HASHLEN); 

    MYSQL_STMT *stmt;
    MYSQL_BIND ps_params[1];
    my_bool    is_null;
    word32     row_id = 0;
    int        status;
    long unsigned int hash_len = HASHLEN;

    stmt   = mysql_stmt_init(conn);
    status = mysql_stmt_prepare(stmt, "SELECT id FROM block WHERE hash = ? AND main_chain = 1", 54); // TODO: WARN: network_id not specified
    memset(ps_params, 0, sizeof(ps_params));

    ps_params[0].buffer_type   = MYSQL_TYPE_STRING;
    ps_params[0].buffer        = &(bt->bhash);
    ps_params[0].buffer_length = HASHLEN;
    ps_params[0].length        = &hash_len;
    ps_params[0].is_null       = 0;

    status = mysql_stmt_bind_param(stmt, ps_params);
    status = mysql_stmt_execute(stmt);

    // Results
    do {
      MYSQL_FIELD *fields;
      MYSQL_BIND *rs_bind;

      if (mysql_stmt_field_count(stmt) > 0) {
        MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
        fields  = mysql_fetch_fields(rs_metadata);
        rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
        memset(rs_bind, 0, sizeof (MYSQL_BIND));

        rs_bind[0].buffer_type   = fields[0].type;
        rs_bind[0].is_null       = &is_null;
        rs_bind[0].buffer        = (char *)&row_id;
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

/**
 * Try exporting an entire block to the database
 */
word32 db_export_block(BHEADER *bh, BTRAILER *bt, MYSQL *conn)
{
  word32  block_num  = get32(bt->bnum);
  my_bool neogenesis = (block_num % 256 == 0);
  word32  header_len = get32(bh->hdrlen);
  word32  tx_count   = get32(bt->tcount);
  word32  difficulty = (word32)bt->difficulty[0];
  word32  solve_time = get32(bt->stime);
  word32  next_time  = get32(bt->time0);
  char    *haiku     = trigg_check(bt->mroot, bt->difficulty[0], bt->bnum);

  if (neogenesis) {
    tx_count = (header_len - sizeof(BHEADER)) / sizeof(LENTRY); // # ledger entries
  }

  if (haiku == NULL) {
    haiku = "";
  }

  if (check_block_exists(bt, conn)) {
    printf("  SKIP: Already imported %d.\n", block_num);
    return 0;
  }

  // Export the miner address and get the row_id
  word32 miner_addr_id = db_export_address(bh->maddr, conn);
  if (miner_addr_id == 0) {
    printf("  ERROR: Unable to insert miner address.\n");
    return 0;
  }

  // Call `block_insert` stored procedure
  MYSQL_STMT *stmt;
  MYSQL_BIND ps_params[16];
  my_bool    is_null;
  word32     row_id = 0;
  int        status;
  long unsigned int hash_len = HASHLEN;
  long unsigned int haiku_len = strlen(haiku);
 
  stmt   = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL block_insert(?, 0, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 71);
  memset(ps_params, 0, sizeof(ps_params));

  ps_params[0].buffer_type = MYSQL_TYPE_LONG;
  ps_params[0].buffer      = (char *)&block_num;
  ps_params[0].length      = 0;
  ps_params[0].is_null     = 0;

  ps_params[1].buffer_type = MYSQL_TYPE_TINY;
  ps_params[1].buffer      = (char *)&neogenesis;
  ps_params[1].length      = 0;
  ps_params[1].is_null     = 0;

  ps_params[2].buffer_type = MYSQL_TYPE_LONG;
  ps_params[2].buffer      = (char *)&header_len;
  ps_params[2].length      = 0;
  ps_params[2].is_null     = 0;

  ps_params[3].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[3].buffer        = &(bt->bhash);
  ps_params[3].buffer_length = HASHLEN;
  ps_params[3].length        = &hash_len;
  ps_params[3].is_null       = 0;

  ps_params[4].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[4].buffer        = &(bt->phash);
  ps_params[4].buffer_length = HASHLEN;
  ps_params[4].length        = &hash_len;
  ps_params[4].is_null       = 0;

  ps_params[5].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[5].buffer        = &(bt->nonce);
  ps_params[5].buffer_length = HASHLEN;
  ps_params[5].length        = &hash_len;
  ps_params[5].is_null       = 0;

  ps_params[6].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[6].buffer        = &(bt->mroot);
  ps_params[6].buffer_length = HASHLEN;
  ps_params[6].length        = &hash_len;
  ps_params[6].is_null       = 0;

  ps_params[7].buffer_type = MYSQL_TYPE_LONG;
  ps_params[7].buffer      = (char *)&tx_count;
  ps_params[7].length      = 0;
  ps_params[7].is_null     = 0;

  ps_params[8].buffer_type = MYSQL_TYPE_LONG;
  ps_params[8].buffer      = (char *)&miner_addr_id;
  ps_params[8].length      = 0;
  ps_params[8].is_null     = 0;

  ps_params[9].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[9].buffer      = (char *)&(bh->mreward);
  ps_params[9].length      = 0;
  ps_params[9].is_null     = 0;

  ps_params[10].buffer_type = MYSQL_TYPE_LONGLONG;
  ps_params[10].buffer      = (char *)&(bt->mfee);
  ps_params[10].length      = 0;
  ps_params[10].is_null     = 0;

  ps_params[11].buffer_type = MYSQL_TYPE_LONG;
  ps_params[11].buffer      = (char *)&difficulty;
  ps_params[11].length      = 0;
  ps_params[11].is_null     = 0;

  ps_params[12].buffer_type = MYSQL_TYPE_LONG;
  ps_params[12].buffer      = (char *)&solve_time;
  ps_params[12].length      = 0;
  ps_params[12].is_null     = 0;

  ps_params[13].buffer_type = MYSQL_TYPE_LONG;
  ps_params[13].buffer      = (char *)&next_time;
  ps_params[13].length      = 0;
  ps_params[13].is_null     = 0;

  ps_params[14].buffer_type   = MYSQL_TYPE_STRING;
  ps_params[14].buffer        = haiku;
  ps_params[14].buffer_length = haiku_len;
  ps_params[14].length        = 0;
  ps_params[14].is_null       = 0;

  ps_params[15].buffer_type = MYSQL_TYPE_LONG;
  ps_params[15].buffer      = (char *)&row_id;
  ps_params[15].length      = 0;
  ps_params[15].is_null     = 0;

  // Execute
  status = mysql_stmt_bind_param(stmt, ps_params);
  status = mysql_stmt_execute(stmt);

  // Results
  do {
    MYSQL_FIELD *fields;
    MYSQL_BIND *rs_bind;

    if (mysql_stmt_field_count(stmt) > 0) {
      MYSQL_RES *rs_metadata = mysql_stmt_result_metadata(stmt);
      fields  = mysql_fetch_fields(rs_metadata);
      rs_bind = (MYSQL_BIND *) malloc(sizeof (MYSQL_BIND));
      memset(rs_bind, 0, sizeof (MYSQL_BIND));

      rs_bind[0].buffer_type   = fields[0].type;
      rs_bind[0].is_null       = &is_null;
      rs_bind[0].buffer        = (char *)&row_id;
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

  return row_id;
}

/**
 * Try loading a blockfile and exporting it
 */
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

/**
 * Export blocks
 */
int export(char *path)
{
  // Set path for block files (called from ./d like ../bx -e)
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

  // Load database configuration
  char *conf[MYSQL_CONF_NUM];
  if (!get_mysql_conf(conf)) {
    printf("ERROR: Could not load database configuration file\n");
    return 1;
  }

  // Connect to the database
  if (mysql_real_connect(conn, conf[MYSQL_CONF_HOSTNAME], conf[MYSQL_CONF_USERNAME], conf[MYSQL_CONF_PASSWORD], conf[MYSQL_CONF_DATABASE], 0, NULL, 0) == NULL) {
    printf("ERROR: Could not connect to the `mochimo_bx` database\n");
    return 1;
  }

  // Iterate over all available block files and export to the database
  char block_file_paths[32768][255];

  DIR *dir = opendir(path);
  int index = 0;
  for (int i = 0; i < 1024; i++) {
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

  // Update cache (current aeon blocks + balances)
  printf("Updating cache\n");
  stmt = mysql_stmt_init(conn);
  status = mysql_stmt_prepare(stmt, "CALL cache_update()", 19);
  status = mysql_stmt_execute(stmt);
  mysql_stmt_close(stmt);

  mysql_close(conn);

  return 0;
}
