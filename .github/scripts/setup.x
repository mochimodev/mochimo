#!/bin/bash
##
# setup.x - Mochimo Node setup script
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#
# Date: 1 November 2021
#

### Update/Install dependencies
apt update && apt install -y build-essential git-all

### Check existence of service
if test ! -f "/etc/systemd/system/mochimo.service"; then

### Create Mochimo Relaynode Service
cat <<EOF >/etc/systemd/system/mochimo.service
# Contents of /etc/systemd/system/mochimo.service
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
systemctl daemon-reload
systemctl enable mochimo.service

### End existence of service
fi

### Create mochimo user
if test -z "$(getent passwd mochimo-node); then
   useradd -m -d /home/mochimo-node -s /bin/bash mochimo-node
fi

### Switch to mochimo-node user
su mochimo-node

### Change directory to $HOME and download Mochimo Software
if test -d "~/mochimo"; then
   echo "EXISTING MOCHIMO DIRECTORY DETECTED. Updating..."
   cd ~/mochimo && git pull
else
   cd ~ && git clone --single-branch https://github.com/mochimodev/mochimo.git
fi

### After successful compile and install, switch user back and reboot
cd ~/mochimo/src && ./makeunx bin -DCPU && ./makeunx install && \
   cp ~/mochimo/bin/maddr.mat ~/mochimo/bin/maddr.dat && exit && reboot
