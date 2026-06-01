import os, shutil, tempfile

zshrc_path = os.path.expanduser("~/.zshrc")
content = open(zshrc_path).read()

config = """
# PATH
export PATH="$HOME/local/bin:$PATH"

# Proxy
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# DeepSeek API for Claude Code
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="sk-ae1249616ebc4fd79e94c0ae52827284"
export ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"
"""

tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zshrc", dir=os.path.dirname(zshrc_path))
with os.fdopen(tmp_fd, "w") as f:
    f.write(content + config)
os.replace(tmp_path, zshrc_path)
print("Done writing .zshrc")

zprofile_path = os.path.expanduser("~/.zprofile")
pcontent = open(zprofile_path).read()

if "ANTHROPIC_BASE_URL" in pcontent:
    print("DeepSeek config already in .zprofile")
else:
    tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".zprofile", dir=os.path.dirname(zprofile_path))
    with os.fdopen(tmp_fd2, "w") as f:
        f.write(pcontent + """
# DeepSeek API for Claude Code
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="sk-ae1249616ebc4fd79e94c0ae52827284"
export ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"
""")
    os.replace(tmp_path2, zprofile_path)
    print("Done writing .zprofile")
