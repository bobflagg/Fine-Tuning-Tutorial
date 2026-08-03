#!/usr/bin/env bash
set -euo pipefail
log() { echo -e "\n[bootstrap] $*\n"; }

mkdir -p /root/.cache/huggingface

apt-get update
apt-get install -y tree
apt-get install -y htop


pip install -r /workspace/Fine-Tuning-Tutorial/requirements.txt


# Install Claude Code:
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
cat << 'EOF'
Set up Claude Code:
>> claude
  /plugin marketplace add huggingface/skills
  /plugin install hf-cli@huggingface/skills
  hf skills add huggingface-trackio
EOF

