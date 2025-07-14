#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai


#### Will install ollama and llama3 LLM needed for this project currently.
# Will install user ollama on the system unless you modify this script.
# Will install service to run ollama server as ollama user


## Blah some strange script, lets do it more simply in this script 
#curl -fsSL https://ollama.com/install.sh | sh

if [ ! -d /usr/share/ollama ];then
	echo "Make user"	
        sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
	sudo usermod -a -G ollama $(whoami)
fi

if [ ! -f /etc/systemd/system/ollama.service ];then
echo "Add service"
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=\$PATH"

[Install]
WantedBy=multi-user.target
EOF

echo "Reload daemon"
sudo systemctl daemon-reload
echo "Enable"
sudo systemctl enable --now ollama.service
fi

read -p "Install Ollama Package" ANS
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ./ollama-linux-amd64.tgz
echo "Install the package"
sudo tar -C /usr -xzf ./ollama-linux-amd64.tgz
echo "Make sure ownership is good... ollama of /usr/share/ollama"
sudo chown -R ollama:ollama /usr/share/ollama
echo "Restart ollama.service"
sudo systemctl restart ollama.service

echo "Almost done..."
sleep 3
echo "Install the LLM package"
ollama pull llama3
echo "Test the new born LLM"
ollama run llama3 "Write a Python function to print Hello World."
