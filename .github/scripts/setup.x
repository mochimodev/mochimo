#!/bin/bash
## setup.x - Mochimo Node setup script
#
# Copyright (c) 2022 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#

### Defaults
MOCHIMO_USER="mochimo"
MOCHIMO_DIR="/home/$MOCHIMO_USER/main"
BRANCH_OPT=

### Check for branch option
if test ! -z $1; then
   MOCHIMO_DIR="/home/$MOCHIMO_USER/$1"
   BRANCH_OPT="-b $1"
fi

### Create mochimo user (if not exists)
if test -z "$(getent passwd $MOCHIMO_USER)"; then
   if ! useradd -m -d /home/$MOCHIMO_USER -s /bin/bash $MOCHIMO_USER; then
      echo "   Failed to create '$MOCHIMO_USER' user."
      echo "   Try creating user '$MOCHIMO_USER' manually."
      echo "   Then rerun setup." && echo
      exit
   fi
fi

### Check for existing installation
if test -d $MOCHIMO_DIR; then echo
   echo "   MOCHIMO DIRECTORY DETECTED."
   echo "   Performing mochimo update..." && echo
   ## store latest commit for later checking (if any)
   PREVCOMMIT=$(su $MOCHIMO_USER -c "git -C $MOCHIMO_DIR/ rev-parse HEAD 2>/dev/null")
   ### Ensure correct ownership of existing mochimo directory
   chown -R $MOCHIMO_USER:$MOCHIMO_USER $MOCHIMO_DIR
   ### Perform update on git repo
   su $MOCHIMO_USER -c "git -C $MOCHIMO_DIR pull"
   ### Check for effective updates
   CURRCOMMIT=$(su $MOCHIMO_USER -c "git -C $MOCHIMO_DIR rev-parse HEAD 2>/dev/null")
   if test ! "$PREVCOMMIT" = "$CURRCOMMIT"; then echo
      echo "   Stopping Mochimo service."
      echo "   This can take up to 90 seconds..." && echo
      systemctl stop mochimo.service
      ### Rebuild from source
      su $MOCHIMO_USER -c "make -C $MOCHIMO_DIR cleanall install"
      if test $? -ne 0; then
         echo "   FAILED TO REBUILD UPDATE FROM SOURCE!!!" && echo
         ### Restart existing service and exit
         systemctl start mochimo.service
         exit 1
      fi
   fi
else
   ### Update/Install Dependencies (ONLY on first install)
   while apt-get update && apt-get install -y build-essential git
   do test $? -eq 0 && break || \
      (echo "   Failed to install deps, retrying..." && sleep 2)
   done
   ### Clone mochimo <branch> into directory
   su $MOCHIMO_USER -c "git clone $BRANCH_OPT https://github.com/mochimodev/mochimo.git $MOCHIMO_DIR"
   ### Build from source
   su $MOCHIMO_USER -c "make -C $MOCHIMO_DIR install"
   if test $? -ne 0; then
      echo "   FAILED TO BUILD MOCHIMO FROM SOURCE!!!" && echo
      exit 1
   fi
fi

### Create or update systemd service file
cat <<EOF >/etc/systemd/system/mochimo.service
[Unit]
Description=Mochimo Relay Node
After=network.target
[Service]
User=$MOCHIMO_USER
Group=$MOCHIMO_USER
WorkingDirectory=$MOCHIMO_DIR/bin/
ExecStartPre=/bin/sh -c 'until ping -c1 1.1.1.1; do sleep 1; done;'
ExecStart=/bin/bash $MOCHIMO_DIR/bin/mcmd -ll 5
[Install]
WantedBy=multi-user.target
EOF

### Reload service daemon, enable Mochimo Service and start
systemctl daemon-reload
systemctl enable mochimo.service
systemctl restart mochimo.service

### Wait a bit...
sleep 1 && echo

### Check mochimo installation
if systemctl is-active --quiet mochimo; then
   echo "   SETUP COMPLETE!"
else
   echo "   SETUP FAILED!!!"
   echo "   For more details, run:"
   echo "      systemctl status mochimo.service"
fi

echo
