# LLM Agent — Autonomous File System Agent

> **AI-powered CLI tool** that autonomously navigates, reads, edits, and manages your file system using natural language instructions via any OpenAI-compatible LLM API.

[![PyPI version](https://img.shields.io/pypi/v/llm-agent?color=green)](https://pypi.org/project/llm-agent/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is this?

**LLM Agent** is a terminal-based AI agent that transforms natural language commands into file system operations. Point it at a directory and ask it to refactor code, find bugs, migrate data, or organize files — it will explore, plan, and execute autonomously until the task is complete.

Built on top of any OpenAI-compatible API (OpenAI, Ollama, LM Studio, vLLM, Groq, Cohere, etc.), so you can use local models or cloud providers with zero vendor lock-in.

## Features

- **🔍 Deep exploration** — Recursively lists, searches, and reads files to understand your codebase before acting
- **✏️ Autonomous editing** — Creates, modifies, replaces, and deletes files based on your instructions
- **🤖 Model agnostic** — Works with any OpenAI-compatible API endpoint (local or cloud)
- **📦 Named profiles** — Saves and switches between multiple API keys, base URLs, and model configurations
- **🔄 Token streaming** — Real-time streaming of model output for transparency
- **🛡️ Sandboxed execution** — Operates within a confined root directory to prevent accidental system damage
- **🔧 Rich tool suite** — 20+ file system tools: read, write, edit, grep, glob, search, tree, diff, word count, permissions, and more
- **📊 Live stats** — Progress bar, iteration tracking, error counts, and elapsed time displayed per step

## Installation

```bash
git clone https://github.com/unn-Known1/llm-agent.git
cd llm-agent
pip install -e .
```

Then run:

```bash
llm-agent              # Interactive REPL
llm-agent "task here"   # One-shot mode
llm-agent --help        # Show all options
```

## Quick Start

```bash
# Interactive REPL (first time: configure via /set commands)
llm-agent

# One-shot mode
llm-agent "Find all TODO comments and summarize them" --root ./myproject

# With verbose output
llm-agent "Refactor all database queries to use transactions" --root ./backend --verbose
```

## Configuration

```bash
# Interactive REPL
llm-agent

# Set up your profile
/set key sk-your-api-key
/set url https://api.openai.com/v1
/set model gpt-4o
/root /path/to/project

# Run tasks
"Fix all TypeError exceptions in the auth module"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `list_dir` | List directory contents |
| `read_file` | Read file with line ranges |
| `write_file` | Create or overwrite a file |
| `edit_file` | Modify file content with old/new text |
| `replace_in_file` | Regex-based find and replace |
| `delete_file` | Delete a file (with confirmation) |
| `make_dir` | Create directories |
| `remove_dir` | Remove directories |
| `glob` | Find files by glob pattern |
| `search` | Grep-style content search |
| `grep` | Find files containing regex |
| `count_matches` | Count regex matches in file |
| `tree` | Display directory tree |
| `file_info` | Get file metadata |
| `head` / `tail` | First/last N lines |
| `word_count` | Count lines, words, chars |
| `diff` | Show differences between files |
| `file_type` | Detect MIME type |
| `file_permissions` | Get/set file permissions |
| `run_command` | Execute shell commands |

## Command Reference

```bash
llm-agent [task] [flags]

Flags:
  --root, -r <path>         Set sandbox root directory
  --model, -m <model>       Override active model
  --api-key <key>           Override API key
  --api-url <url>           Override base URL
  --max-iter, -n <n>        Max iterations (default: 30)
  --read-only               Disable write operations
  --dry-run                 Simulate writes without executing
  --no-stream               Disable token streaming
  --show-thoughts, -t       Show model reasoning in real-time
  --no-stats                Hide live stats bar
  --verbose, -v             Enable debug logging
  --quiet, -q               Minimal output
  --output, -o <file>       Write result JSON to file
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/config` | Show active profile settings |
| `/profiles` | List all saved profiles |
| `/profile use <name>` | Switch to a profile |
| `/profile save <name>` | Save current settings as a profile |
| `/set key <value>` | Set API key |
| `/set url <value>` | Set base URL |
| `/set model <name>` | Set active model |
| `/set iter <n>` | Set max iterations |
| `/models` | List models in profile |
| `/root [path]` | Show or change sandbox root |
| `/history` | Show recent tasks |
| `/help` | Show all commands |
| `/quit` | Exit |

## Architecture

```
User Task → LLM (GPT-4o / Ollama / etc.)
              ↓
         Tool Call (JSON)
              ↓
         Sandbox (file system ops within root)
              ↓
         Result → LLM (reasoning loop)
              ↓
         Iteration until "done"
```

The agent maintains full conversation history across turns, giving the model complete visibility of all prior tool calls and results — not just the last one.

## Supported APIs

Works with any **OpenAI-compatible API endpoint**:

- OpenAI (gpt-4o, gpt-4o-mini, etc.)
- Ollama (llama3, qwen, mistral, etc.)
- LM Studio
- vLLM
- Groq
- Cohere
- Azure OpenAI
- Custom proxies

## License

MIT License — see [LICENSE](LICENSE)