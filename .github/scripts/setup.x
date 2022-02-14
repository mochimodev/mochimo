#!/bin/bash

##
# setup.x - Mochimo Node setup script
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#
# Date: 1 November 2021
#

### Update/Install Dependencies
while
   apt update && apt install -y build-essential git-all
do test $? -eq 0 && break || (printf "\n   Retrying...\n\n" && sleep 2); done

### Ensure latest service file is installed
cat <<EOF >/etc/systemd/system/mochimo.service
[Unit]
Description=Mochimo Relay Node
After=network.target
[Service]
User=mochimo-node
Group=mochimo-node
WorkingDirectory=/home/mochimo-node/mochimo/bin/
ExecStart=/bin/bash /home/mochimo-node/mochimo/bin/gomochi d -n -D
[Install]
WantedBy=multi-user.target
EOF

### Reload service daemon and enable Mochimo Relaynode Service
systemctl daemon-reload && systemctl enable mochimo.service

### Create mochimo user
if test -z "$(getent passwd mochimo-node)"; then
   if ! useradd -m -d /home/mochimo-node -s /bin/bash mochimo-node; then
      printf "\n   Failed to create 'mochimo-node' user."
      printf "\n   Try creating user 'mochimo-node' manually."
      printf "\n   Then rerun setup.\n\n"
      exit
   fi
fi

### Ensure correct ownership of existing mochimo directory
if test -d /home/mochimo-node/mochimo; then
   chown -R mochimo-node:mochimo-node /home/mochimo-node/mochimo
fi

## store latest commit for later checking (if any)
PREVCOMMIT=$(git -C /home/mochimo-node/mochimo/ rev-parse HEAD 2>/dev/null)

### Update or clone as mochimo-node user
sudo -u mochimo-node sh<<EOC

### Download or Update Mochimo Software
if test -d ~/mochimo; then
   printf "\n   MOCHIMO DIRECTORY DETECTED."
   printf "\n   Performing mochimo update...\n\n"
   cd ~/mochimo && git pull
else
   cd ~ && git clone --single-branch https://github.com/mochimodev/mochimo.git
fi

EOC

### Check mochimo installation and (re)start service (only if updated)
if test -d /home/mochimo-node/mochimo; then
   CURRCOMMIT=$(git -C /home/mochimo-node/mochimo rev-parse HEAD 2>/dev/null)
   if test ! "$PREVCOMMIT" = "$CURRCOMMIT"; then
      ### (re)Compile software as mochimo-node user
      sudo -u mochimo-node sh<<EOC

### After successful compile and install
cd ~/mochimo/src && ./makeunx bin -DCPU && ./makeunx install && \
   cp ~/mochimo/bin/maddr.mat ~/mochimo/bin/maddr.dat

EOC
      printf "\n   (re)Starting Mochimo service."
      printf "\n   This can take up to 90 seconds...\n\n"
      service mochimo restart
   fi
fi

### Check mochimo installation and (re)start service (only if updated)
if test -e /home/mochimo-node/mochimo/bin/mochimo; then
   printf "\n   SETUP COMPLETE!\n\n"
else
   printf "\n   SETUP FAILED!!!\n\n"
fi