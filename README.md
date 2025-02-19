# BeagleMind
<img src="beaglemind.webp" width="100" height="100" />
Chat with BeagleBoard docs using Qwen2.5-Instruct and RAG. Ask about BeagleBone, BeaglePlay, or anything in `docs.beagleboard.io`.

## Setup

1. **Clone the Docs**
   ```bash
   git clone https://github.com/beagleboard/docs.beagleboard.io
   ```

2. **Install uv**
   - **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - More options: [uv docs](https://docs.astral.sh/uv/getting-started/installation/)

3. **Sync Dependencies**
   ```bash
   uv sync
   ```

4. **Get a Model**
   Download Qwen2.5-Instruct (e.g., `Qwen/Qwen2.5-7B-Instruct`) from Hugging Face and note its path.

## Usage

Run the chat:
```bash
uv run python inference.py -c path/to/qwen2.5/model
```

- Ask away: `What’s the BeagleBone AI?`
- Commands:
  - `:h` - Help
  - `:q` - Quit
  - `:cl` - Clear screen
  - `:clh` - Clear history
  - `:his` - Show history
  - `:seed <N>` - Set seed
  - `:conf` - View config
  - `:conf <key>=<value>` - Tweak config
  - `:reset-conf` - Reset config

**Options:**
- `-c <path>` - Model path (default: `model`)
- `-s <number>` - Seed (default: 1234)
- `--cpu-only` - Run on CPU

## Tips
- First run embeds `docs.beagleboard.io`—takes a sec.
- Adjust `k` in `inference.py`’s `retrieve(query, k=2)` for more context.

## License
Follows Qwen2.5’s license—keep it legit. See `model/LICENSE`.
