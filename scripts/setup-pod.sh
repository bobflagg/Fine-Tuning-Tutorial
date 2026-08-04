#!/usr/bin/env bash
set -euo pipefail
log() { echo -e "\n[bootstrap] $*\n"; }

INSTALL_CLAUDE_CODE=false
while getopts "c" opt; do
  case $opt in
    c)
      INSTALL_CLAUDE_CODE=true
      ;;
    \?)
      echo "Invalid option" >&2
      exit 1
      ;;
  esac
done

# Set credentials:
KEY_PATH="/root/.ssh/id_ed25519_bobflagg"
if [ ! -f "$KEY_PATH" ]; then
  mkdir -p "$(dirname "$KEY_PATH")"
  echo "Paste the contents from <<cat ~/.ssh/id_ed25519_bobflagg>>, then press Ctrl-D on a new line:"
  cat > "$KEY_PATH"
  chmod 600 "$KEY_PATH"
fi
cat << 'EOF'
eval `ssh-agent -s`
ssh-add ~/.ssh/id_ed25519_bobflagg 
EOF

mkdir -p /root/.cache/huggingface

apt-get update
apt-get install -y tree
apt-get install -y htop

pip install -r /workspace/Fine-Tuning-Tutorial/requirements.txt

# 4. Evaluate the boolean variable
if [ "$INSTALL_CLAUDE_CODE" = true ]; then
  echo "Installing Claude Code."
  # Install Claude Code:
  curl -fsSL https://claude.ai/install.sh | bash
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
else
  echo "Verbose mode is DISABLED."
fi


