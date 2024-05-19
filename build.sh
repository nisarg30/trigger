#!/bin/bash

# Download and extract TA-Lib source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz

# Change to the TA-Lib directory
cd ta-lib

# Configure, build, and install TA-Lib
./configure --prefix=/usr
make
sudo make install

# Return to the project root directory
cd ..

# Install Python dependencies
pip install TA-Lib
pip install -r requirements.txt
