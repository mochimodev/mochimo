#!/bin/bash
## setup.x - Mochimo Node setup script
#
# Copyright (c) 2022-2025 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#

### Defaults
BRANCH="${1:-master}"
BRANCH_OPT="${1:+-b $1}"
WORKING_DIR="$HOME/.mcm/repo"
USE_LAST_TAG=$(test -z $1 && echo 1 || echo 0)

# fn for sudo where appropriate
use_sudo() {
   if command -v sudo >/dev/null 2>&1 \
         && sudo -n true >/dev/null 2>&1; then
      sudo $@
   else
      $@
   fi
}

check_deps() {
   command -v git >/dev/null 2>&1 || return 1
   command -v make >/dev/null 2>&1 || return 1
}

fail_exit() {
   echo
   echo "   $@"
   echo
   exit 1
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
   exit 0
}

ok_restart() {
   echo
   echo "   $@"
   echo
   systemctl start mochimo.service
}

install_deps() {
   use_sudo apt update && use_sudo apt install -y build-essential git
   test $? -ne 0 && fail_exit "Failed to install dependencies!!!"
}

git_C() {
   git -C "$WORKING_DIR/" $@
}

# fn for repository update
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
   make -C "$WORKING_DIR/" $@
}

sudo_make_C() {
   use_sudo make -C "$WORKING_DIR/" $@
}

### Ensure dependencies are installed
check_deps || install_deps

### Check for existing installation
if test -d "$WORKING_DIR/"; then echo
   echo "   MOCHIMO DIRECTORY DETECTED."
   echo "   Performing mochimo update..." && echo
   ## perform update and check commit changes
   git_update || fail_exit "FAILED TO UPDATE MOCHIMO REPOSITORY!!!"
   echo "   Stopping Mochimo service."
   echo "   This can take up to 90 seconds..." && echo
   use_sudo systemctl stop mochimo.service
   ### Rebuild from source
   make_C clean mochimo || fail_restart "FAILED TO REBUILD MOCHIMO SOURCE!!!"
   sudo_make_C install service || fail_restart "FAILED TO INSTALL MOCHIMO SERVICE!!!"
   ok_restart "MOCHIMO UPDATE COMPLETE!"
else
   # clone and build source
   git clone $BRANCH_OPT https://github.com/mochimodev/mochimo.git "$WORKING_DIR/"
   make_C mochimo || fail_exit "FAILED TO BUILD MOCHIMO FROM SOURCE!!!"
   sudo_make_C install service || fail_exit "FAILED TO INSTALL MOCHIMO SERVICE!!!"
   ok_restart "MOCHIMO INSTALLATION COMPLETE!"
fi
