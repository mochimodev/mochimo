<div align="center">
<a href="https://mochimo.org/">
<img width="75%" src=".github/media/logo-banner.png" />
</a>

[![MPL 2.0 Derivative License](https://img.shields.io/badge/_License-MPL_2.0_Derivative-%23.svg?logo=open%20source%20initiative&labelColor=2d3339&color=0059ff)](.github/LICENSE.pdf)
[![GitHub release (latest by date)](https://img.shields.io/github/release/mochimodev/mochimo.svg?logo=github&logoColor=lightgrey&&labelColor=2d3339&label=Latest%20%27main%27&color=%230059ff)](https://github.com/mochimodev/mochimo/releases)
[![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/mochimodev/mochimo/latest/dev?logo=github&logoColor=lightgrey&label=Latest%20%27dev%27&labelColor=2d3339&color=%230059ff)](https://github.com/mochimodev/mochimo/tree/dev)<br/>
*You must read and agree to the [LICENSE](https://mochimo.org/license.pdf)
prior to running the code.*

**This repository is home to the Mochimo Cryptocurrency Engine code (main-net).**<br/>
It includes a fully functional cryptocurrency network node and a text-based developer's wallet. The full node, and developer's wallet, will compile without issue on most 64-bit Linux-based machines with the GNU Makefile provided under the "src" directory. However, please note that the developer's wallet is provided for development use only. It is recommended to use [Mojo](https://github.com/mochimodev/mojo-java-wallet/releases) as your main wallet software.

|     | `master` | `dev` |
| --: | :------: | :---: |
| Unit: | [![Tests workflow](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml/badge.svg?branch=master)](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml) | [![Tests workflow](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml) |
| Software: | [![Builds workflow](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml/badge.svg?branch=master)](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml) | [![Builds workflow](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml) |
| Static Analysis: | [![CodeQL workflow](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml/badge.svg?branch=master)](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml) | [![CodeQL workflow](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml) |
| Test Coverage: | [![CodeCov code coverage](https://codecov.io/gh/mochimodev/mochimo/graph/badge.svg)](https://codecov.io/gh/mochimodev/mochimo) | [![codecov](https://codecov.io/gh/mochimodev/mochimo/branch/dev/graph/badge.svg)](https://codecov.io/gh/mochimodev/mochimo/branch/dev) |
| Coverage Graph:| <a href="https://codecov.io/gh/mochimodev/mochimo/graphs/sunburst.svg"><img width="128px" alt="CodeCov graph" src="https://codecov.io/gh/mochimodev/mochimo/graphs/sunburst.svg" /></a> | <a href="https://codecov.io/gh/mochimodev/mochimo/branch/dev/graphs/sunburst.svg"><img width="128px" alt="CodeCov graph" src="https://codecov.io/gh/mochimodev/mochimo/branch/dev/graphs/sunburst.svg" /></a> |

</div>

<hr><hr>
<h1 align="center"><strong>REQUIREMENTS</strong></h1>
<sup><strong>Please Note: support cannot be guaranteed for systems that do not meet the recommended requirements.</strong></sup>

## Recommended ~ <sub>![Ubuntu 20.04 LTS](https://img.shields.io/badge/Ubuntu-20.04_LTS-E95420?style=flat&logo=ubuntu&logoColor=white)
- (OS) Ubuntu 20.04 LTS
- (CPU) Dual-core Processor
- (RAM) 2GB of Random Access Memory
- (SSD) 64GB of Solid State Drive Storage
- (NETWORK) Port 2095 incoming TCP/IPv4 access
  - *may require router [port forwarding](https://portforward.com/)*

## Prerequisites
To download and build source from Github, ensure you have appropriate packages installed:
```sh
sudo apt install -y build-essential git-all
```
... or, to use the Quick-Setup script:
```sh
sudo apt-get install -y curl
```

(Optionally) For building GPU mining nodes, you will require appropriate NVIDIA drivers and CUDA Toolkit:<br/>
*Note: the Mochimo Cryptocurrency Engine's GPU mining capability is currently only compatible with NVIDIA GPU's.*
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

<hr><hr>
<h1 align="center"><strong>USAGE</strong></h1>

## Quick Setup/Update (relay-node)
```sh
bash <(curl -sL mochimo.org/setup/node)
```
... or for a specific branch:
```sh
bash <(curl -sL mochimo.org/setup/node) <branch>
```

## Mochimo Server (relay/mining-node)
1) Clone the repository: `git clone https://github.com/mochimodev/mochimo`
2) Enter repository directory: `cd mochimo`
3) Build and install the mining-node<br/>
   a) Make CPU mining node: `make install-mochimo`, or<br/>
   b) Make CUDA mining node: `make install-mochimo CFLAGS=-DCUDA`
4) Enter the binary directory: `cd bin`
5) Copy (or create) your mining address `maddr.dat` in this directory
6) Run Mochimo Node (server)<br/>
   a) Mining mode: `./gomochi`, or<br/>
   b) Relay mode: `./gomochi -n`

## Mochimo Miner
1) Install appropriate NVIDIA Drivers and Cuda Toolkit (see above)
2) Clone the repository: `git clone https://github.com/mochimodev/mochimo`
3) Enter repository directory: `cd mochimo`
4) Build and install the Cuda Miner: `make install-cudaminer`
4) Enter the binary directory: `cd bin`
5) Copy (or create) your mining address `maddr.dat` in this directory
6) Run Mochimo Miner<br/>
   a) Headless Mining: `./miner`, or<br/>
   b) Solo Mining (requires Mochimo Server): `./miner --host <ip>`, or<br/>
   c) Pool Mining (requires Pool): `./miner --pool <ip> --port <num>`

## Additional Instructions
See the [Mochimo Wiki](http://www.mochiwiki.com).

<hr><hr>
<h1 align="center"><strong>LICENSE</strong></h1>

<sup>**The license to use versions of the code prior to v2.0 expired on December 31st, 2018. Use of the old code is strictly prohibited.**</sup><br/>
The current version of the code is released under an MPL2.0 derivative Open Source license.<br/>
The community is free to develop and change the code with the caveat that any changes must be for the benefit of the Mochimo cryptocurrency network (with a number of exclusions).<br/>
Please read the [LICENSE](https://mochimo.org/license.pdf) for more details on limitations and restrictions.

The Mochimo Package (main-net) is copyright 2022 Adequate Systems, LLC.<br/>
Please read the license file in the package for additional restrictions.

Contact: support@mochimo.org

<hr><hr>
<h1 align="center"><strong>COMMUNITY</strong></h1>

Discord is our most active social forum where you can discuss Mochimo with the rest of the developer and beta testing community.
- [![Discord](https://img.shields.io/discord/460867662977695765?label=Discord%20Mochimo%20Official&logo=discord&style=social)](https://discord.mochimap.com)
- [![Twitter Follow](https://img.shields.io/twitter/follow/mochimocrypto?style=social)](https://twitter.com/mochimocrypto)
- [![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/mochimo?style=social)](https://www.reddit.com/r/mochimo/)
- [![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCFW0_JZR32gMvEtJQ3YE0KA?style=social)](https://www.youtube.com/channel/UCFW0_JZR32gMvEtJQ3YE0KA)
- [![Telegram](https://img.shields.io/badge/Telegram-_-white?style=social&logo=telegram)](https://t.me/mochimocrypto)
- [![Medium Articles](https://img.shields.io/badge/Medium-_-white?style=social&logo=medium&logoColor=12100E)](https://medium.com/mochimo-official)

<hr><hr>
<h1 align="center"><strong>BLOCKCHAIN</strong></h1>

<div align="center">
	<h2><strong>Overview</strong></h2>

| | | | | |
| --: | :-- | --: | :-- | :-- |
| **Coin Name (Ticker)** | Mochimo (MCM) | **Maximum Supply** | 76,493,180 MCM (deflationary) | [API](https://new-api.mochimap.com/chain/maxsupply) |
| **Launch Date** | June 25, 2018 15:43:45 UTC | **Transaction Fee** | 0.0000005 MCM | [API](https://new-api.mochimap.com/chain/mfee) |
| **Mining Duration** | ~22 years |

</div>
<div align="center">
	<h2><strong>Reward Distribution</strong></h2>

| | | | |
| --: | :-: | :-: | :--: |
| | **Phase 1** | **Phase 2** | **Phase 3 (current)**|
| **Block Range** | 0 - 17,185 | 17,185 - 373,761 | 373,761 - 2,097,152 |
| **Base Reward** | 5.0 MCM | 5.917392 MCM | 59.523942 MCM |
| **(Block) Delta** | +0.000056 MCM | +0.00015 MCM | -0.000028488 MCM |
| **Last Reward** | 5.962248 MCM | 59.403642 MCM | 10.427979192 MCM |

</div>
