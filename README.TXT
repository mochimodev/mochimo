/*
 * Mochimo Beginner's Guide.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
*/

Welcome to the Mochimo Main Net!

GETTING STARTED
You must read and agree to the LICENSE.PDF before proceeding.

Recommended System:
Ubuntu 16.04 LTS 64-bit
2+ Processor Threads (if running as VM)
1+ GB of System Memory
TCP Port 2095

Notes:
The mochimo node is a "server" in the conventional sense, and must have direct access to the internet. The port that Mochimo operates on is TCP 2095. If you are running the software behind a firewall, you will need to permit inbound connections on TCP 2095. If you are running the software behind a router performing NAT, such as with a home internet connection, you will need to configure a static private IP on your Linux box, and Port Forwarding of port TCP 2095 to the static private IP.

Configuring your Miner:
The recommended setup is an installation of Ubuntu 16.04LTS. This can be accomplished either by Virtualization software or a dedicated install. The choices is yours.  The followin paragraph will go over configuring a static IP address on your Miner. 

Once Ubuntu is installed, open a terminal windows and type ifconfig to find your IP address (it should look something like 192.168.0.24 or 10.0.0.45) and remember it.  Now, find the Settings "gear" icon on the quick launch bar on the screen and click it. In the System Settings window, click on the Network folder icon to get to the Network Settings window. Select the interface you are using to connect. (Either Wired or Wireless) then select Options. 

On the Editing Connection window, select IPv4 Settings. In the drop-down window under Method: Select Manual. Then to the right of the Addresses window, Click on Add. This will allow you to type in the Addresses window. Type the IP address you found earlier in the terminal window. Hit the Tab key and type the number 24 (assuming your private network mask is 255.255.255.0), hit Tab again and put in your router's IP address (this is usually the same as your IP address if you replace the last set of digits with a 1 - e.x. IP Address=192.168.0.24; Router=192.168.1.1, some systems use .254 instead of .1) 

You will also need to enter a DNS Server. If you don't know what your ISP uses, you could always use Google's DNS Servers: 8.8.8.8 and 9.9.9.9

Port Forwarding Instructions:

These are the basic instrucions for how to forward a port through you router to your mining rig. The details will vary widely from router to router. Below is a website that has specific instructions for MANY devices including screenshots. 

First make sure that if your rig is a VM, that it is set to Bridged Mode in the network settings. Your mining rig needs to have an IP address that your router can see. Once that is done, open a terminal on your rig (Ubuntu:Ctrl-Alt-T) and type: ifconfig (and then hit Enter.) Somewhere in that output (depending on how you are setup) is your IP address. It should look something like: 192.168.0.24 or 10.0.0.45 which is what we "programed" it to be in the previous step. 

From now on your local IP address will be referred to as: XX.XX.XX.XX Once you have this, it's time to find your router! AKA, your gateway. Here is the website where you can find instructions on how to forward a port for your specific router:

https://portforward.com/

The basic steps are that you have to tell the router the IP address (XX.XX.XX.XX) should have port 2095 forwarded to it.  The Internal and External port that traffic should be directed through (2095) and the protocol to use (TCP or BOTH). Once you THINK you are done, visit this site to see if it worked:

https://portchecker.co/

Enter 2095 in the port dialog box to see if you are open.  WARNING: this might take a while, so be patient if it says closed at first. If you really are closed, you may still be able to see the network, but you won'tbe able to mine, and your server will continually restart...If everything checks out, you are ready to go!


First, unpack the tarball like this:tar zxvf main-net*.tgz && cd mochi

Compiling Instructions:
The software includes a script for compiling the software, and installing the binaries in the ./bin directory.  

To use it, please confirm you have gcc installed:
   gcc --version

   If not installed, install with: 
      sudo apt install gcc-6 g++-6 

Install Nvidia drivers if necessary: 
   sudo apt install nvidia-xxx Xxx -> driver version you want to install 390.77 is currently the newest driver for 10xx series 

Install kernel headers and development packages for the currently running kernel:  
   sudo apt-get install linux-headers-$(uname -r) 

Download Cuda:
   wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux 

   Make it executable:
     chmod +x cuda_9.2.148_396.37_linux 

Install:
   ./cuda_9.2.148_396.37_linux.run --verbose --silent --toolkit --override


To compile: 
   cd ./src
   ./makeunx bin -DCUDA
   ./makeunx install

Initial Wallet Instructions:
   cd ../bin

   
IMPORTANT:  You should create an address "maddr.dat" using the wallet beforeyou start your miner, or you won't earn any mining rewards.

When you compile the source code using "makeunx install" it will also compile
the text-based wallet software.  Here follows basic instructions on how to
use the text-based wallet.  Note, Mochimo currently has a gorgeouss GUI
wallet under production, called "Mojo", and you can see an example of that
on the website.  However, participants will have to just use the basic text
interface for now.

Create Your Wallet File:

To start, from the ./bin directory, invoke the wallet with the following
command:

./wallet -p2095 -n

1. You will be prompted to choose a name for your wallet, we suggest "mining".
2. You will be prompted for a password.  Your wallet will be encrypted 100% of
   the time. This password will be used to decrypt the wallet, don't lose it.
3. You will be asked to enter some random text to help randomize your wallet
   seed.  You do not need to memorize this information.
4. You will be asked to pick a filename.  We recommend mining.wal.
5. The wallet file is saved to disk and the wallet software exits.

Open Your Wallet

If you named your wallet "mining.wal", then open the wallet with the following
command:

./wallet -p2095 mining.wal

You will be prompted to enter your password.  If you enter this incorrectly, the system will not tell you, but the name at the top of the wallet when you open it will not say "Mining", but rather some garbled gibberish.  The reason for this is the password is a decryption key and is not stored in the wallet itself.  If you enter it wrong, the wallet will still decrypt using the wrong key, and the result will be garbage.  You know you entered it correctly if the
wallet name you chose displays at the top of the screen.

Creating a mining address:

1. From the main menu, select option 4 to create an address.  You will be prompted to name it.  We suggest calling it "mining address".  

WARNING: YOU MUST *NOT* ADD A TAG TO A MINING ADDRESS.

2. Select option 2, to display your address entries.

3. Select option 6, to check the balance on this new address.  You won't have any coins yet.

4. Now let's save this address to a file, so that we can start mining and have our block rewards go to this address.

5. Select option 7, to export.  You will be prompted to select the index number of the address you want to export.  In this case it will be index 1, since it's the only address you have.  Type 1 and hit enter.

6. You will be prompted to select a file name.  The file has to be named: maddr.dat

7. You will be asked where to save the Balance.  Please select "N".

8. You will be told that maddr already exists if you've done everything correctly.  Enter Y to overwrite the default mining address with your new mining address.

9. Now type 0 to exit, and go invoke your miner.  

You are ready to mine!

Starting the Node/Miner:

Change to the ./bin directory, and invoke the miner with the following command:
   ./gomochi d

Stopping the Node/Miner:

Type CTRL+C and wait a moment for the monitor to appear.  Hit ? and <ENTER> to see options.  Type q and <ENTER> to exit.  Wait a few moments for the software to exit gracefully.

Resuming the Miner:

You may resume from where you left off (so that you don't have to resync your blocks) by typing:
   ./resume d
   
   
Good luck!  Come visit us on the Discord channel (Devs are there all day every day), or send us an email at support@mochimo.org if you have any questions!
