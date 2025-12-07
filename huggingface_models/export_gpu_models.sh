#!/bin/bash

# Ensure PROJECT_BASE is set
if [ -z "$PROJECT_BASE" ]; then
    echo "Error: PROJECT_BASE environment variable is not set!"
    exit 1
fi

# First command
SRC1="$PROJECT_BASE/huggingface_models/Llama-3.2-3B-Instruct"
DEST="$PROJECT_BASE/gpu_models"

if [ -d "$SRC1" ]; then
    echo "Copying $SRC1 to $DEST ..."
    cp -r "$SRC1" "$DEST"
else
    echo "Warning: Directory $SRC1 does not exist. Skipping copy."
fi

# Second command
MODEL2="$PROJECT_BASE/huggingface_models/Llama-2-7b-chat-hf"
if [ -d "$MODEL2" ]; then
    echo "Saving model Llama-2-7b-chat-hf ..."
    python3 "$PROJECT_BASE/ServerlessLLM_modify/examples/sllm_store/save_vllm_model.py" \
        --local_model_path "$MODEL2" \
        --model_name Llama-2-7b-chat-hf \
        --storage_path "$DEST" \
        --tensor_parallel_size 1
else
    echo "Warning: Directory $MODEL2 does not exist. Skipping save."
fi

# Third command
MODEL3="$PROJECT_BASE/huggingface_models/Llama-2-13b-chat-hf"
if [ -d "$MODEL3" ]; then
    echo "Saving model Llama-2-13b-chat-hf ..."
    python3 "$PROJECT_BASE/ServerlessLLM_modify/examples/sllm_store/save_vllm_model.py" \
        --local_model_path "$MODEL3" \
        --model_name Llama-2-13b-chat-hf \
        --storage_path "$DEST" \
        --tensor_parallel_size 1
else
    echo "Warning: Directory $MODEL3 does not exist. Skipping save."
fi

echo "Script finished."
