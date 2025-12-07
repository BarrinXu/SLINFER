#!/bin/bash

# Ensure PROJECT_BASE is set
if [ -z "$PROJECT_BASE" ]; then
    echo "Error: PROJECT_BASE environment variable is not set!"
    exit 1
fi

# First model
SRC1="$PROJECT_BASE/huggingface_models/Llama-3.2-3B-Instruct"
DEST1="$PROJECT_BASE/cpu_models/Llama-3.2-3B-Instruct"

if [ -d "$SRC1" ]; then
    echo "Exporting Llama-3.2-3B-Instruct to OpenVINO format..."
    optimum-cli export openvino --model "$SRC1" \
        --task text-generation-with-past \
        --weight-format fp16 "$DEST1"
else
    echo "Warning: Directory $SRC1 does not exist. Skipping export."
fi

# Second model
SRC2="$PROJECT_BASE/huggingface_models/Llama-2-7b-chat-hf"
DEST2="$PROJECT_BASE/cpu_models/Llama-2-7b-chat-hf"

if [ -d "$SRC2" ]; then
    echo "Exporting Llama-2-7b-chat-hf to OpenVINO format..."
    optimum-cli export openvino --model "$SRC2" \
        --task text-generation-with-past \
        --weight-format fp16 "$DEST2"
else
    echo "Warning: Directory $SRC2 does not exist. Skipping export."
fi

# Third model
SRC3="$PROJECT_BASE/huggingface_models/Llama-2-13b-chat-hf"
DEST3="$PROJECT_BASE/cpu_models/Llama-2-13b-chat-hf"

if [ -d "$SRC3" ]; then
    echo "Exporting Llama-2-13b-chat-hf to OpenVINO format..."
    optimum-cli export openvino --model "$SRC3" \
        --task text-generation-with-past \
        --weight-format fp16 "$DEST3"
else
    echo "Warning: Directory $SRC3 does not exist. Skipping export."
fi

echo "Script finished."
