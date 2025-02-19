# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A BeagleBoard-powered command-line chat demo with RAG."""

import argparse
import os
import platform
import shutil
from copy import deepcopy
from embeddings import retrieve  # Import the retrieval function
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = "model"

_WELCOME_MSG = """\
Welcome to the BeagleBoard Chat Demo! Type your question to dive into BeagleBoard docs, or :h for help.
Powered by Qwen2.5-Instruct and BeagleBoard documentation via RAG.
"""
_HELP_MSG = """\
Commands:
    :help / :h              Show this help message
    :exit / :quit / :q      Exit the demo
    :clear / :cl            Clear screen
    :clear-history / :clh   Clear chat history
    :history / :his         Show chat history
    :seed                   Show current random seed
    :seed <N>               Set random seed to <N>
    :conf                   Show generation config
    :conf <key>=<value>     Tweak generation config
    :reset-conf             Reset generation config
"""
_ALL_COMMAND_NAMES = [
    "help", "h", "exit", "quit", "q", "clear", "cl",
    "clear-history", "clh", "history", "his", "seed", "conf", "reset-conf"
]


def _setup_readline():
    try:
        import readline
    except ImportError:
        return

    _matches = []

    def _completer(text, state):
        nonlocal _matches
        if state == 0:
            _matches = [cmd for cmd in _ALL_COMMAND_NAMES if cmd.startswith(text)]
        return _matches[state] if 0 <= state < len(_matches) else None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, resume_download=True)
    device_map = "cpu" if args.cpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, torch_dtype="auto", device_map=device_map, resume_download=True
    ).eval()
    model.generation_config.max_new_tokens = 2048  # Plenty for BeagleBoard chats
    model.generation_config.temperature = 0.6  # Keep it coherent, Beagle-style
    return model, tokenizer


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    os.system("cls" if platform.system() == "Windows" else "clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f"Chat History ({len(history)})".center(terminal_width, "="))
    for i, (query, response) in enumerate(history):
        print(f"You[{i}]: {query}")
        print(f"BeagleBot[{i}]: {response}")
    print("=" * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input("You> ").strip()
        except UnicodeDecodeError:
            print("[ERROR] Input encoding goofed up")
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print("[ERROR] Don’t leave me hanging—type something!")


def _chat_stream(model, tokenizer, query, history):
    # Grab BeagleBoard docs with RAG
    retrieved_docs = retrieve(query, k=2)  # Adjust k for more/less context
    if retrieved_docs:
        context = "\n\n".join(
            f"Source: {path}\nContent: {doc[:1000]}"  # Cap at 1000 chars per doc
            for doc, path in retrieved_docs
        )
    else:
        context = "No BeagleBoard docs found for that one."

    # Build convo with history
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    
    # Craft a BeagleBoard-flavored prompt
    augmented_query = (
        f"Using BeagleBoard docs as context:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Give me a solid answer based on the docs where it fits."
    )
    conversation.append({"role": "user", "content": augmented_query})

    # Prep the model input
    input_text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    # Stream the response
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {**inputs, "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def main():
    parser = argparse.ArgumentParser(description="BeagleBoard Chat Demo with Qwen2.5-Instruct.")
    parser.add_argument(
        "-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
        help="Path to Qwen2.5 model (default: 'model')"
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU only")
    args = parser.parse_args()

    history, response = [], ""
    model, tokenizer = _load_model_tokenizer(args)
    orig_gen_config = deepcopy(model.generation_config)

    _setup_readline()
    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed
    while True:
        query = _get_input()

        if query.startswith(":"):
            command_words = query[1:].strip().split()
            command = command_words[0] if command_words else ""

            if command in ["exit", "quit", "q"]:
                break
            elif command in ["clear", "cl"]:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ["clear-history", "clh"]:
                print(f"[INFO] Wiped {len(history)} chat entries")
                history.clear()
                _gc()
                continue
            elif command in ["help", "h"]:
                print(_HELP_MSG)
                continue
            elif command in ["history", "his"]:
                _print_history(history)
                continue
            elif command == "seed":
                if len(command_words) == 1:
                    print(f"[INFO] Current seed: {seed}")
                else:
                    try:
                        seed = int(command_words[1])
                        print(f"[INFO] Seed set to {seed}")
                    except ValueError:
                        print(f"[WARNING] {command_words[1]} ain’t a number, bro")
                continue
            elif command == "conf":
                if len(command_words) == 1:
                    print(model.generation_config)
                else:
                    for kv in command_words[1:]:
                        if "=" not in kv:
                            print("[WARNING] Use <key>=<value> format")
                            continue
                        key, value = kv.split("=", 1)
                        try:
                            setattr(model.generation_config, key, eval(value))
                            print(f"[INFO] Set {key} = {value}")
                        except Exception as e:
                            print(f"[ERROR] {e}")
                continue
            elif command == "reset-conf":
                model.generation_config = deepcopy(orig_gen_config)
                print("[INFO] Config reset to default")
                print(model.generation_config)
                continue

        set_seed(seed)
        _clear_screen()
        print(f"\nYou: {query}")
        print(f"\nBeagleBot: ", end="")
        try:
            partial_text = ""
            for new_text in _chat_stream(model, tokenizer, query, history):
                print(new_text, end="", flush=True)
                partial_text += new_text
            response = partial_text
            print()
            history.append((query, response))
        except KeyboardInterrupt:
            print("[WARNING] Chat interrupted—hit me again!")
            continue


if __name__ == "__main__":
    main()
