#!/bin/bash
DIR=`pwd`
sudo mkdir /data
sudo chmod -R 777 /data
sudo chmod -R 777 /opt/conda
cp install.sh /data
cd /data
sh install.sh
cd $DIR