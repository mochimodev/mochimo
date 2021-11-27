#!/bin/bash

##
# setup.x - Mochimo Node setup script
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#
# Date: 1 November 2021
# Revised: 27 November 2021
#

### Update/Install Dependencies
while
   DEBIAN_FRONTEND=noninteractive apt update && apt install -y build-essential git-all
do test $? -eq 0 && break || (echo -e "\n   Retry in 3 seconds...\n" && sleep 3); done

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
      echo -e "\n   Failed to create 'mochimo-node' user."
      echo -e   "   Try creating user 'mochimo-node' manually and rerun script.\n"
      exit
   fi
fi

### Switch to mochimo-node user
su mochimo-node || (echo -e "\n   Failed to switch to user 'mochimo-node'.\n" && exit)

### Change directory to $HOME and download Mochimo Software
if test -d ~/mochimo; then
   echo -e "\n   MOCHIMO DIRECTORY DETECTED.\n   Performing mochimo update...\n"
   cd ~/mochimo && git pull
else
   cd ~ && git clone --single-branch https://github.com/mochimodev/mochimo.git
fi

### After successful compile and install, switch user back
cd ~/mochimo/src && ./makeunx bin -DCPU && ./makeunx install && \
   cp ~/mochimo/bin/maddr.mat ~/mochimo/bin/maddr.dat

### Return to previous user
exit

### (Re)Start Mochimo Service
echo -e "\n   (re)Starting Mochimo service.\n   This can take up to 90 seconds\n"
service mochimo restart
