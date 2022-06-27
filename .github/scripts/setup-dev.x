#!/bin/bash

##
# setup.x - Mochimo Node setup script
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#
# Date: 1 November 2021
#

### Update/Install Dependencies (ONLY on first install)
if test ! -d /home/mochimo-node/mochimo; then
   while
      apt update && apt install -y build-essential git-all
   do test $? -eq 0 && break || (printf "\n   Retrying...\n\n" && sleep 2); done
fi

### Ensure latest service file is installed
cat <<EOF >/etc/systemd/system/mochimo.service
[Unit]
Description=Mochimo Relay Node
After=network.target
[Service]
User=mochimo-node
Group=mochimo-node
WorkingDirectory=/home/mochimo-node/mochimo/bin/
ExecStart=/bin/bash /home/mochimo-node/mochimo/bin/gomochi -n
Restart=on-failure
RestartSec=3s
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

### Stop service if running
printf "\n   Stopping existing Mochimo service."
printf "\n    * This can take up to 90 seconds...\n\n"
service mochimo stop

### Update or clone as mochimo-node user
if test -d /home/mochimo-node/mochimo; then
   printf "\n   MOCHIMO DIRECTORY DETECTED."
   printf "\n   Performing mochimo update...\n\n"

### START USER mochimo-node
sudo -u mochimo-node sh<<EOC
cd ~/mochimo && make uninstall && git pull && make install-mochimo
EOC
### END USER mochimo-node

else
   printf "\n   Performing mochimo CLEAN install...\n\n"

### START USER mochimo-node
sudo -u mochimo-node sh<<EOC
git clone -b dev https://github.com/mochimodev/mochimo.git ~/mochimo \
 && cd ~/mochimo && make install-mochimo
EOC
### END USER mochimo-node

fi

### Check mochimo installation and (re)start service (only if updated)
if test -e /home/mochimo-node/mochimo/bin/mochimo; then
   printf "\n   SETUP COMPLETE!\n\n"
   service mochimo start
else
   printf "\n   SETUP FAILED!!!\n\n"
fi
