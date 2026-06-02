# local-claude-code
Run the most popular coding agent locally !

# Run Claude Code Locally with llama.cpp and Qwen3.5

This tutorial documents a working setup for running **Claude Code** against a local **llama.cpp `llama-server`** instance using **Qwen3.5 GGUF** models.

The recommended default is **Qwen3.5-9B** for weaker machines. A **Qwen3.5-35B-A3B** command is included as an optional larger-model setup.

The final working architecture is:

```text
Claude Code CLI
  -> Anthropic-compatible HTTP request
  -> local llama.cpp llama-server
  -> Qwen3.5 GGUF model
```

## Index

- [Tested outcome](#tested-outcome)
- [Important notes](#important-notes)
- [1. Install system dependencies](#1-install-system-dependencies)
- [2. Add CUDA to your shell environment](#2-add-cuda-to-your-shell-environment)
  - [WSL note](#wsl-note)
- [3. Clone llama.cpp and pin a release tag](#3-clone-llamacpp-and-pin-a-release-tag)
- [4. Build llama.cpp with CUDA](#4-build-llamacpp-with-cuda)
  - [CPU-only fallback](#cpu-only-fallback)
- [5. Start llama-server with Qwen3.5-9B](#5-start-llama-server-with-qwen35-9b)
  - [Optional KV cache memory reduction](#optional-kv-cache-memory-reduction)
- [6. Optional: Qwen3.5-35B-A3B command](#6-optional-qwen35-35b-a3b-command)
  - [Speed-oriented first attempt](#speed-oriented-first-attempt)
  - [VRAM-saving variant](#vram-saving-variant)
- [7. Test the local Anthropic-compatible endpoint](#7-test-the-local-anthropic-compatible-endpoint)
- [8. Install Claude Code](#8-install-claude-code)
- [9. Configure Claude Code for the local model](#9-configure-claude-code-for-the-local-model)
- [10. Configure Claude Code settings](#10-configure-claude-code-settings)
- [11. Start Claude Code](#11-start-claude-code)
- [12. Optional launcher script](#12-optional-launcher-script)
- [13. Troubleshooting](#13-troubleshooting)
  - [Error: auth conflict](#error-auth-conflict)
  - [Error: system message must be at the beginning](#error-system-message-must-be-at-the-beginning)
  - [Error: no CUDA compiler found](#error-no-cuda-compiler-found)
  - [Build killed with cc1plus or nvcc terminated](#build-killed-with-cc1plus-or-nvcc-terminated)
  - [Connection refused](#connection-refused)
  - [Claude Code still answers as Claude Opus](#claude-code-still-answers-as-claude-opus)
  - [Out of memory while running the model](#out-of-memory-while-running-the-model)
  - [Return to normal Claude Code](#return-to-normal-claude-code)
- [14. Minimal working recipe](#14-minimal-working-recipe)
- [15. Notes on Homebrew](#15-notes-on-homebrew)
- [References](#references)

## Tested outcome

A working validation inside Claude Code should look like this:

```text
/model
Set model to unsloth/Qwen3.5-9B and saved as your default for new sessions

Say one sentence confirming which model name you are using.
I am using the unsloth/Qwen3.5-9B model.
```

## Important notes

- `nvidia-smi` working only proves that the NVIDIA driver is visible. Building llama.cpp with CUDA also requires the CUDA compiler `nvcc`.
- Claude Code must use only one local auth variable. Do not set both `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`.
- Qwen3.5 with Claude Code may need llama.cpp's chat parser workaround: `--skip-chat-parsing`.
- For weaker machines, start with Qwen3.5-9B and `--ctx-size 8192` or `16384`.
- For CUDA builds, prefer building llama.cpp from source. Homebrew is not recommended for NVIDIA CUDA builds.

---

## 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y git cmake build-essential curl libcurl4-openssl-dev ccache
```

For CUDA acceleration, install the NVIDIA CUDA Toolkit appropriate for your distro.

Check whether `nvcc` is available:

```bash
which nvcc
nvcc --version
```

If `nvidia-smi` works but `nvcc` is missing, the NVIDIA driver is installed but the CUDA Toolkit is not correctly installed or is not in your shell `PATH`.

---

## 2. Add CUDA to your shell environment

If `nvcc` exists under `/usr/local/cuda/bin/nvcc` but is not found by your shell, add CUDA to `~/.bashrc`:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

Verify again:

```bash
which nvcc
nvcc --version
```

If you installed a versioned CUDA path such as `/usr/local/cuda-13.3`, either make sure `/usr/local/cuda` points to it or replace `/usr/local/cuda` in the commands above with the versioned path.

### WSL note

On WSL, install the NVIDIA driver on Windows. Inside WSL, install the CUDA Toolkit, not the Linux NVIDIA display driver. If `nvidia-smi` works in WSL but `nvcc` does not, you still need the toolkit inside WSL.

---

## 3. Clone llama.cpp and pin a release tag

`llama.cpp` moves quickly. For reproducibility, use a release tag instead of `master`.

At the time this tutorial was finalized, `b9478` was the latest checked tag used for this setup.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

git fetch --tags
git checkout b9478

git describe --tags --always
```

Expected output:

```text
b9478
```

If this tag no longer exists or a newer tag is preferred, check the llama.cpp releases page and use a recent release tag.

---

## 4. Build llama.cpp with CUDA

Use a clean build directory:

```bash
cd ~/llama.cpp
rm -rf build
```

Configure with CUDA:

```bash
cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Build with low parallelism to avoid RAM exhaustion:

```bash
cmake --build build --config Release --parallel 2 --target llama-cli llama-server
```

If the build is killed with an error like this:

```text
c++: fatal error: Killed signal terminated program cc1plus
nvcc: Terminated
```

then the OS likely killed the compiler due to insufficient RAM or swap. Retry with one build job:

```bash
cmake --build build --config Release --parallel 1 --target llama-cli llama-server
```

Confirm an out-of-memory kill with:

```bash
sudo dmesg -T | grep -Ei 'killed process|out of memory|oom|cc1plus|nvcc' | tail -n 50
```

Optional temporary swap:

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

Make swap persistent only if you want it permanently:

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### CPU-only fallback

If you do not want CUDA:

```bash
cd ~/llama.cpp
rm -rf build

cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF

cmake --build build --config Release --parallel 2 --target llama-cli llama-server
```

CPU-only will work but will be much slower for coding-agent use.

---

## 5. Start llama-server with Qwen3.5-9B

This is the recommended command for weaker machines.

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --alias unsloth/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8001 \
  --ctx-size 16384 \
  --parallel 1 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --jinja \
  --skip-chat-parsing \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00
```

For lower VRAM, reduce context:

```bash
--ctx-size 8192
```

If the model does not fit fully on GPU, reduce GPU offload:

```bash
--n-gpu-layers 20
```

or let llama.cpp decide:

```bash
--n-gpu-layers auto
```

### Optional KV cache memory reduction

If VRAM is tight, add:

```bash
--cache-type-k q8_0 \
--cache-type-v q8_0
```

This mainly reduces KV cache memory. It may or may not improve speed; benchmark both default KV cache and `q8_0` KV cache on your machine.

---

## 6. Optional: Qwen3.5-35B-A3B command

Use this only if you have enough RAM/VRAM.

### Speed-oriented first attempt

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  --model unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --alias unsloth/Qwen3.5-35B-A3B \
  --host 127.0.0.1 \
  --port 8001 \
  --ctx-size 16384 \
  --parallel 1 \
  --n-gpu-layers all \
  --flash-attn on \
  --jinja \
  --skip-chat-parsing \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00
```

### VRAM-saving variant

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  --model unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --alias unsloth/Qwen3.5-35B-A3B \
  --host 127.0.0.1 \
  --port 8001 \
  --ctx-size 8192 \
  --parallel 1 \
  --n-gpu-layers all \
  --flash-attn on \
  --jinja \
  --skip-chat-parsing \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00 \
  --kv-unified \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

Notes:

- `--n-gpu-layers all` is usually the biggest speed win if the model fits in VRAM.
- `--flash-attn on` is generally worth trying for long context.
- `--parallel 1` is appropriate for a single Claude Code session.
- `q8_0` KV cache is mostly for memory reduction; test whether it helps speed on your GPU.
- If you used `hf download`, make sure the `--model` path matches your actual file path.

---

## 7. Test the local Anthropic-compatible endpoint

In a second terminal, test `llama-server` directly:

```bash
curl http://127.0.0.1:8001/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "unsloth/Qwen3.5-9B",
    "max_tokens": 64,
    "messages": [
      {
        "role": "user",
        "content": "Say one sentence confirming the local model works."
      }
    ]
  }'
```

If this fails, fix `llama-server` before starting Claude Code.

---

## 8. Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

Restart your shell if `claude` is not found.

---

## 9. Configure Claude Code for the local model

Use only `ANTHROPIC_AUTH_TOKEN`. Do not also set `ANTHROPIC_API_KEY`.

```bash
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8001"
export ANTHROPIC_AUTH_TOKEN="sk-local"
export ANTHROPIC_CUSTOM_MODEL_OPTION="unsloth/Qwen3.5-9B"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen3.5 9B Local"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Qwen3.5-9B served locally by llama.cpp"
```

Remove conflicting API key exports from `~/.bashrc`:

```bash
sed -i '/ANTHROPIC_API_KEY/d' ~/.bashrc
```

Persist the local configuration:

```bash
cat >> ~/.bashrc <<'EOF'

# Claude Code local llama.cpp / Qwen3.5-9B
export ANTHROPIC_BASE_URL="http://127.0.0.1:8001"
export ANTHROPIC_AUTH_TOKEN="sk-local"
export ANTHROPIC_CUSTOM_MODEL_OPTION="unsloth/Qwen3.5-9B"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen3.5 9B Local"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Qwen3.5-9B served locally by llama.cpp"
EOF

source ~/.bashrc
```

If switching to Qwen3.5-35B-A3B, change the model variables:

```bash
export ANTHROPIC_CUSTOM_MODEL_OPTION="unsloth/Qwen3.5-35B-A3B"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen3.5 35B A3B Local"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Qwen3.5-35B-A3B served locally by llama.cpp"
```

---

## 10. Configure Claude Code settings

Create or update `~/.claude/settings.json`:

```bash
mkdir -p ~/.claude

cat > ~/.claude/settings.json <<'JSON'
{
  "env": {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
JSON
```

This reduces nonessential traffic and avoids extra attribution content that can interfere with local gateway prompt handling.

---

## 11. Start Claude Code

Make sure `llama-server` is already running in another terminal.

```bash
cd ~/Teachess
claude --model unsloth/Qwen3.5-9B
```

Inside Claude Code, run:

```text
/model
```

Select or confirm:

```text
unsloth/Qwen3.5-9B
```

Then test:

```text
Say one sentence confirming which model name you are using.
```

Expected answer:

```text
I am using the unsloth/Qwen3.5-9B model.
```

If the first answer still reports Claude Opus, use `/model` again and explicitly select the Qwen model. After selection, repeat the test.

---

## 12. Optional launcher script

Create a helper script:

```bash
mkdir -p ~/bin

cat > ~/bin/claude-qwen35-local <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8001"
export ANTHROPIC_AUTH_TOKEN="sk-local"
export ANTHROPIC_CUSTOM_MODEL_OPTION="unsloth/Qwen3.5-9B"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen3.5 9B Local"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Qwen3.5-9B served locally by llama.cpp"

exec claude --model unsloth/Qwen3.5-9B "$@"
EOF

chmod +x ~/bin/claude-qwen35-local
```

Run:

```bash
cd ~/Teachess
claude-qwen35-local
```

---

## 13. Troubleshooting

### Error: auth conflict

Example:

```text
Auth conflict: Both a token (ANTHROPIC_AUTH_TOKEN) and an API key (ANTHROPIC_API_KEY) are set.
```

Fix:

```bash
unset ANTHROPIC_API_KEY
sed -i '/ANTHROPIC_API_KEY/d' ~/.bashrc
source ~/.bashrc
```

Keep only:

```bash
export ANTHROPIC_AUTH_TOKEN="sk-local"
```

### Error: system message must be at the beginning

Example:

```text
API Error: 400 Unable to generate parser for this template.
Error: Jinja Exception: System message must be at the beginning.
```

Fix by restarting `llama-server` with:

```bash
--jinja \
--skip-chat-parsing \
--chat-template-kwargs '{"enable_thinking":false}'
```

If it still fails, try the ChatML fallback:

```bash
--jinja \
--chat-template chatml \
--skip-chat-parsing \
--chat-template-kwargs '{"enable_thinking":false}'
```

### Error: no CUDA compiler found

Example:

```text
No CMAKE_CUDA_COMPILER could be found.
```

Check:

```bash
which nvcc
nvcc --version
ls -l /usr/local/cuda/bin/nvcc
```

If `nvcc` exists, export CUDA paths:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Then rebuild from a clean directory:

```bash
cd ~/llama.cpp
rm -rf build

cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Build killed with cc1plus or nvcc terminated

Example:

```text
c++: fatal error: Killed signal terminated program cc1plus
nvcc: Terminated
```

Use fewer build jobs:

```bash
cmake --build build --config Release --parallel 1 --target llama-cli llama-server
```

Optionally add swap:

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

### Connection refused

Check that `llama-server` is running and listening on the expected port:

```bash
curl http://127.0.0.1:8001/v1/models
```

If you used another port, update:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:<port>"
```

### Claude Code still answers as Claude Opus

Inside Claude Code, run:

```text
/model
```

Then explicitly select:

```text
unsloth/Qwen3.5-9B
```

Repeat:

```text
Say one sentence confirming which model name you are using.
```

Also watch the terminal running `llama-server`; a local request should appear there.

### Out of memory while running the model

Try these changes in order:

```bash
--ctx-size 8192
```

```bash
--cache-type-k q8_0 \
--cache-type-v q8_0
```

```bash
--n-gpu-layers 20
```

For very limited machines, use Qwen3.5-9B rather than Qwen3.5-35B-A3B.

### Return to normal Claude Code

Unset the local gateway variables:

```bash
unset ANTHROPIC_BASE_URL
unset ANTHROPIC_AUTH_TOKEN
unset ANTHROPIC_API_KEY
unset ANTHROPIC_CUSTOM_MODEL_OPTION
unset ANTHROPIC_CUSTOM_MODEL_OPTION_NAME
unset ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION
```

Then run:

```bash
claude
```

---

## 14. Minimal working recipe

Terminal 1:

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --alias unsloth/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8001 \
  --ctx-size 16384 \
  --parallel 1 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --jinja \
  --skip-chat-parsing \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Terminal 2:

```bash
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8001"
export ANTHROPIC_AUTH_TOKEN="sk-local"
export ANTHROPIC_CUSTOM_MODEL_OPTION="unsloth/Qwen3.5-9B"

cd ~/Teachess
claude --model unsloth/Qwen3.5-9B
```

Inside Claude Code:

```text
/model
Say one sentence confirming which model name you are using.
```

Expected:

```text
I am using the unsloth/Qwen3.5-9B model.
```

---

## 15. Notes on Homebrew

`brew install llama.cpp` can be useful for a quick CPU or macOS install, but it is not the recommended path for an NVIDIA CUDA build. For CUDA, build from source with:

```bash
-DGGML_CUDA=ON
```

and point CMake at `nvcc` if needed:

```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

---

## References

- Claude Code model configuration: <https://code.claude.com/docs/en/model-config>
- Claude Code LLM gateway configuration: <https://code.claude.com/docs/en/llm-gateway>
- Claude Code environment variables: <https://code.claude.com/docs/en/env-vars>
- llama.cpp build documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp server documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- llama.cpp releases: <https://github.com/ggml-org/llama.cpp/releases>
- Unsloth Qwen3.5 documentation: <https://unsloth.ai/docs/models/qwen3.5>
- Unsloth Claude Code local LLM guide: <https://unsloth.ai/docs/basics/claude-code>
- NVIDIA CUDA Linux installation guide: <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>
- NVIDIA CUDA on WSL guide: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- CMake build parallelism: <https://cmake.org/cmake/help/latest/envvar/CMAKE_BUILD_PARALLEL_LEVEL.html>
