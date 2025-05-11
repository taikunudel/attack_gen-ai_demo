#!/bin/bash
export ATKER_PATH='bert-base-uncased'
# export TARGET_PATH='llama-guard-3-1b' # default is LLaMA Guard 3 1b
# export SERVER_URL='http://infodeep:8000/v1'
export TARGET_PATH='llama-guard-3-8b' # default is LLaMA Guard 3 8b
export SERVER_URL='http://infodeep:8001/v1'
export SAVE_TO_PATH='/usa/taikun/07_transencoder/attack-genai/trained_attacker/attacker_05082025_162359_llama-guard_doc_0.5_9_9100_0.7800.pth'
export LEN_DOC_MAX=512
export NUM_DOC_MASKS=10
export SAMPLES_PER_TOK=100
export ATTACK_WHAT='doc'

export EVALUATION_PREFIX="eva_$(date +%m%d_%H%M%S)_${ATKER_PATH}_${TARGET_PATH}_${ATTACK_WHAT}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}"
export OUTPUT_TXT="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.txt"
export ATK_JSON_LOG="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.json"

# # Print the summary message
echo "Attacking with the following parameters:"
echo "Attacker Name: $ATKER_PATH"
echo "Target Model: $TARGET_PATH"
echo "JSON File: $ATK_JSON_LOG"

SCRIPT_PATH="/usa/taikun/07_transencoder/attack-genai/evaluation_attacker_genai.py"
# Run the python script with the defined variables
python -u $SCRIPT_PATH \
  --atker_path $ATKER_PATH \
  --target_path $TARGET_PATH \
  --save_to_path $SAVE_TO_PATH \
  --len_doc_max $LEN_DOC_MAX \
  --num_doc_masks $NUM_DOC_MASKS \
  --atk_json_log $ATK_JSON_LOG \
  --server_url $SERVER_URL \
  --samples_per_tok $SAMPLES_PER_TOK > $OUTPUT_TXT 2>&1
