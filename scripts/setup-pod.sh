mkdir -p /root/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/.cache/huggingface

apt-get update
apt-get install -y tree
apt-get install -y htop

curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
curl -sSL https://app.mcpmarket.com/install/9sqROjSriagzhBAv5S9KXrylNQvCaUfIutMGpeu7p30 | bash
cat << 'EOF'
Set up Claude Code:
>> claude
  /plugin marketplace add huggingface/skills
  /plugin install hf-cli@huggingface/skills
  hf skills add huggingface-trackio
EOF

