#!/bin/bash
## setup.x - Mochimo Node setup script
# Designed primarily for one-line curl execution
#    curl -L <remote>/setup.x | sudo bash -s -- [branch]
#
# Copyright (c) 2022-2025 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#

### Check permissions
if test $(id -u) -ne 0; then
   echo "This script requires superuser privileges."
   echo "Please run with sudo or as root."
   exit 1
fi

### Defaults
BRANCH="${1:-master}"
BRANCH_OPT="${1:+-b $1}"
USE_LAST_TAG=$(test -z $1 && echo 1 || echo 0)
USER_ACTUAL=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$USER_ACTUAL" | cut -d: -f6)
USER_HOME=${USER_HOME:-$HOME}
USER_MCM_DIR="$USER_HOME/.mcm"
USER_WORKING_DIR="$USER_MCM_DIR/repo"

### Functions
check_deps() {
   command -v git >/dev/null 2>&1 || return 1
   command -v make >/dev/null 2>&1 || return 1
}

clean_exit() {
   # fix permissions
   chown -R $USER_ACTUAL:$USER_ACTUAL "$USER_MCM_DIR/"
   exit $1
}

fail_exit() {
   echo
   echo "   $@"
   echo
   clean_exit 1
}

fail_restart() {
   echo
   echo "   $@"
   echo
   systemctl start mochimo.service
}

ok_exit() {
   echo
   echo "   $@"
   echo
   clean_exit 0
}

ok_restart() {
   echo
   echo "   $@"
   echo
   systemctl start mochimo.service
}

install_deps() {
   apt update && apt install -y build-essential git
   test $? -ne 0 && fail_exit "Failed to install dependencies!!!"
}

git_C() {
   git -C "$USER_WORKING_DIR/" $@
}

git_update() {
   # obtain the latest branch state
   git_C fetch && git_C checkout $BRANCH && git_C pull || \
      return $? # return failures

   # rewind to last tag if not branch request
   if test $USE_LAST_TAG -eq 1; then
      git_C checkout $(git_C describe --tags --abbrev=0) || \
         return $? # return failures
   fi
}

make_C() {
   make -C "$USER_WORKING_DIR/" $@
}

### Ensure dependencies are installed
check_deps || install_deps

### Check for existing installation
if test -d "$USER_WORKING_DIR/"; then echo
   echo "   MOCHIMO DIRECTORY DETECTED."
   echo "   Performing mochimo update..." && echo
   # perform update and check commit changes
   git_update || fail_exit "FAILED TO UPDATE MOCHIMO REPOSITORY!!!"
   echo "   Stopping Mochimo service."
   echo "   This can take up to 90 seconds..." && echo
   systemctl stop mochimo.service
   # rebuild from source
   make_C clean mochimo || fail_restart "FAILED TO REBUILD MOCHIMO SOURCE!!!"
   make_C install service || fail_restart "FAILED TO INSTALL MOCHIMO SERVICE!!!"
   ok_restart "MOCHIMO UPDATE COMPLETE!"
else
   # clone and build source
   git clone $BRANCH_OPT https://github.com/mochimodev/mochimo.git "$USER_WORKING_DIR/"
   make_C mochimo || fail_exit "FAILED TO BUILD MOCHIMO FROM SOURCE!!!"
   make_C install service || fail_exit "FAILED TO INSTALL MOCHIMO SERVICE!!!"
   ok_restart "MOCHIMO INSTALLATION COMPLETE!"
fi

### DONE
clean_exit 0
