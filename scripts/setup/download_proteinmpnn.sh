#!/bin/bash  
# Download ProteinMPNN weights
echo "Downloading ProteinMPNN weights..."
git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN 2>/dev/null || echo "Already cloned"
