<div align="center">
<a href="https://mochimo.org/">
<img width="75%" src=".github/media/logo-banner.png" />
</a>

*You must read and agree to the [LICENSE](https://mochimo.org/license.pdf)
prior to running the code.*

**This repository is home to the Mochimo Cryptocurrency Engine code (main-net).**<br/>
It includes a fully functional cryptocurrency network node and a text-based developer's wallet. The full node, and developer's wallet, will compile without issue on most 64-bit Linux-based machines with the GNU Makefile provided under the "src" directory. However, please note that the developer's wallet is provided for development use only. It is recommended to use [Mojo](https://github.com/mochimodev/mojo-java-wallet/releases) as your main wallet software.

|     | `master` | `dev` |
| --: | :------: | :---: |
| Version: | [![GitHub release (latest by date)](https://img.shields.io/github/release/mochimodev/mochimo.svg?logo=github&logoColor=lightgrey&&labelColor=2d3339&label=Release&color=%230059ff)](https://github.com/mochimodev/mochimo/releases) | [![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/mochimodev/mochimo/latest/dev?logo=github&logoColor=lightgrey&label=Commits&labelColor=2d3339&color=%230059ff)](https://github.com/mochimodev/mochimo/tree/dev) |
| Unit Tests: | [![Tests workflow](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml/badge.svg)](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml) | [![Tests workflow](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/tests.yaml) |
| Software Builds: | [![Builds workflow](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml/badge.svg)](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml) | [![Builds workflow](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/builds.yaml) |
| Static Analysis: | [![CodeQL workflow](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml/badge.svg)](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml) | [![CodeQL workflow](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml/badge.svg?branch=dev)](https://github.com/mochimodev/mochimo/actions/workflows/codeql.yaml) |
| Test Coverage: | [![CodeCov code coverage](https://codecov.io/gh/mochimodev/mochimo/graph/badge.svg)](https://codecov.io/gh/mochimodev/mochimo) | [![codecov](https://codecov.io/gh/mochimodev/mochimo/branch/dev/graph/badge.svg)](https://codecov.io/gh/mochimodev/mochimo/branch/dev) |

</div>

<hr><hr>
<h1 align="center"><strong>REQUIREMENTS</strong></h1>

## Recommended ~ <sub>![Ubuntu 20.04 LTS](https://img.shields.io/badge/Ubuntu-20.04_LTS-E95420?style=flat&logo=ubuntu&logoColor=white)
- (OS) Ubuntu 20.04 LTS
- (CPU) Dual-core Processor
- (RAM) 2GB of Random Access Memory
- (SSD) 64GB of Solid State Drive Storage
- (NETWORK) Port 2095 incoming TCP/IPv4 access
  - *may require router [port forwarding](https://portforward.com/)*

<hr><hr>
<h1 align="center"><strong>USAGE</strong></h1>

## Quick Setup/Update (relay-node)
The quick setup/update script can be used to quickly provision or update a Mochimo Server on a Ubuntu Machine. To use, simply run:
```sh
sudo apt-get install -y curl # if not already installed
bash <(curl -sL mochimo.org/setup.x)
```
... or to install a specific branch, run:
```sh
bash <(curl -sL mochimo.org/setup.x) <branch>
```

## Other Guides and Information
<sup><i>NOTE: mining guides for v2.4.2 and above, are on the Community Wiki</i></sup>
* [Mochimo Community Wiki](http://github.com/mochimodev/mochimo/wiki)
* [Mochimo Official Wiki](http://www.mochiwiki.com)

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
