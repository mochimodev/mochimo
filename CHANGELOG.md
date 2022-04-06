# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



# Changelog

<!--

   CHANGELOG.md - Mochimo Changelog
   Copyright 2018-2022 Adequate Systems, LLC. All Rights Reserved.

   Changelog format is based on <https://keepachangelog.com/>

-->

## [Unreleased]
Repository restructure and implementation of the Adequate Systems Build-C repository for streamlined CI/CD processes and updated build utilities.

### Added
- build-c utilities (build-c-1.1.1) for automated building testing and coverage
  - includes LICENSE.md (so Github detects the existance of a license)
### Changed
- moved/revised github specific templates and standards
- moved `LICENSE.PDF` to `.github/` directory

## [2.4.1] Mochimo Patch Level 37

### Added
- byte Insyncup;  /* non-zero when syncup() runs */ to data.c
- Insyncup to syncup() in syncup.c
- addrlen to le_find() in ledger.c
- Tagidx[] to tag.c
- tag_free() to tag.c, update.c
- proper time_t vtime for vstart.lck check in server.c
- Insyncup to update() in update.c to help plog()'s
- char *solvestr to update() to reckon pushed, solved, and updated blocks
- Syncup Function, Removing Contention
- mochimo/bin/d/split directory
- send_found() to syncup.c
- Send found message to low weight peer in refresh_ipl()
- MTXTRIGGER to bval.c mtxval.c txclean.c txval.c config.h
- system call to init-external.sh on initial system sync
- system call to update-external.sh on successful block update
- advanced support for block-explorer export functionality
- support for third-party utility triggered system restart()
- directories mochimo/src/test and mochimo/src/old for testing this build
- weight checks to contention() and checkproof()
- functions sub_weight(), and past_weight() to proof.c
- return code to send_found() in update.c
- sftimer for send_found() in server.c
- vstart.lck restart trigger (Verisimility) to server.c

### Changed
- PATCHLEVEL to 37 in mochimo.c and minertest.c
- VERSIONSTR to "37" in mochimo.c and minertest.c
- tag_find() in tag.c, txclean.c, and mtxval.c
- pval.c server.c, and config.h to allow 0xff pseudo-blocks
- simplified parameter logic in tag_valid() and calls from bval and txval.c
- bx.c to indicate tags not found in MTX
- replaced 100 with NR_DST in bval.c mtxval.c txclean.c
- proof.c comments
- syncup.c fprintf's to plog's
- server.c ipltime from 600 to 300
- checkproof() in proof.c
- comments and plog's in gettx.c
- put back V23TRIGGER check in checkproof() in proof.c
- tried to improve comments and plog() messages.
- swapped Bail(1) and Bail(2) to avoid recomputing past_weight() if first trailer doesn't match
- Improved Code readability / comments for checkproof() bail conditions 6 & 7

### Removed
- bigwait from server.c
- FILE *rlog from syncup()

### Fixed
- previous hash check in checkproof() in proof.c
- missing error return lines in bval2() in gettx.c
- Bugs in checkproof()
- bug in past_weight() to skip NG blocks
- bug in checkproof() to skip Difficulty check on init
- bug in syncup() w/first previous NG block
- bug in syncup() to skip NG blocks

## [2.4.0] Mochimo Patch Level 34

### Added
- FPGA-Resistant Algorithm
  - Extensible High-Memory Algo v24()
- Multi-Destination Transactions
  - Scales up high volume third-party payment systems

### Security

- fixes and networking tweaks

## [2.3.0] Mochimo Patch Level 33

### Added
- Pseudoblocks for mid-block difficulty adjustment
  - Impossible for blocks to exceed 15m49s after v2.3
- security fixes to MROOT creation
  - Removes certain spoofing attack vectors
- support for ZERO-tag OP_BAL Requests
  - Allows address lookup and balance query without TAG
  - Allows wallet recovery from seed phrase
- TXCLEAN Queue Re-validation following block updates
  - Prevents known cornercase block validation failures
  - Prevents all attack vectors involving poisoned TXs in the TXCLEAN queue
- low-balance pruning consensus mechanism (CAROUSEL)
  - Allows the community to clean low balances out of the ledger by consensus
  - Keeps the blockchain free of bloat, and recovers from spam attacks
- optional mining fee adjustment per miner
  - Allows future miners to create a mining fee market after mining rewards are gone
- TFILE PoPOW Chain in OP_FOUND
  - Allows nodes to definitively confirm an advertising node really solved a block
- Watchdog timer to restart and resync if no blocks solved or updated in 30 minutes +/- 10 minutes
  - Prevents Block of Death, Stuck at 0x0 events caused by temporary internet outage for nodes
  - Allows nodes to dynamically recover from any number of possible failure cases
- Upload Bandwidth limit (default = 5MB/s Upload, user configurable)
  - Prevents certain kinds of spam attacks against nodes
- NOMINER feature, initialized with -n at runtime
  - Allows a node to run in relay mode only without mining blocks
- command line compile options for CPU or GPU, merging both development branches
  - Involed with ./makeunx bin -DCPU  -or-  ./makeunx bin -DCUDA

### Changed
- inbound TX uniqueness test (CRC) with address-based validation
  - Allows the system to scale past 65,536 TXs per block at some indeterminate future date

## [2.2.0] Mochimo Patch Level 32

### Added
- support for diverse nVidia GPU Models
- OPCODE 15 & 16 (pull candidate / push solved blocks)
- support for Headless Miners and Mining Pools
- node capability bits to identify server capabilities during handshake
- new execute and dispatch functions for handling headless miner requests
- new reaping function for terminating stale child processes related to headless miners

### Changed
- CUDA Code optimizations for average HP/s +10-20%
- improved sanity checking in get_eon (prep for ring signatures)
- adjusted wait time up to 300 seconds from 180 if no quorum found

### Fixed
- random seed issue in rand.c
- various community requested patches


## [2.0.0] Mochimo Patch Level 31

October 27th, 2018

### Added
- added new open source license
- added trigger block for new weight calculation as 17185 (0x4321)
- added trigger block for new reward calculation as 17185 (0x4321)
- added trigger block for new difficulty calculation as 17185 (0x4321)
- added trigger block for tag system validation checks as 17185 (0x4321)
- added dynamic start nodes list download from mochimap.net
- added tag.c, tag related fixes throughout
- added wallet Build 31 with tag support

### Changed
- update system version number to 2
- updated default coreip.lst
- reorganized source code distro in prep for Github
- Adjusted TXVAL to insist src addresses must fully spent
- Enabled balance forwarding to change address
- bup.c
  - patch a bunch of stuff
  - balances debit '-' first, then credit 'A'
- bval.c
  - trancodes: '-' and 'A'
  - enforces no tag on bh.maddr
  - tag mods
  - permanent future time fix
- bupdata.c
  - new set_difficulty() with preset trigger
  - new set_difficulty() block trigger = 16383
- gomochi
  - sleep set to 1 second
  - added dynamic startnodes.lst download
- init.c
  - new add_weight() improved block weight fork on block trigger
  - new add_weight() -DNEWWEIGHT forks chain on block trigger
  - get_eon(): timeout set to 180
  - modified read_coreipl() and init_coreipl()
  - permanent future time fix
- data.c: #define CORELISTLEN 16
- gettx.c: added contention(), catchup(), and bval2()
- server.c: removed LULL timer
- txclean.c: fixed unlink(argv parameter 1) bug
- util.c: new get_mreward() on block trigger

### Removed
- removed default maddr.dat
- removed txq1.lck (process_tx() is now synchronous)

[Unreleased]: https://github.com/adequatesystems/build-c/compare/v2.4.1...HEAD
[2.4.1]: https://github.com/adequatesystems/build-c/compare/v2.4...v2.4.1
[2.4.0]: https://github.com/adequatesystems/build-c/compare/v2.3...v2.4
[2.3.0]: https://github.com/adequatesystems/build-c/compare/v2.2...v2.3
[2.2.0]: https://github.com/adequatesystems/build-c/compare/v2.1...v2.2
[2.1.0]: https://github.com/adequatesystems/build-c/compare/v2.0...v2.1
[2.0.0]: https://github.com/adequatesystems/build-c/releases/tag/v2.0
