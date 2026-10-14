
```
# Download the model
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Convert TTS model (transformer + speaker encoder + tokenizer)
python scripts/convert_tts_to_gguf.py \
    --type q4_k \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-0.6b-base-q4_k.gguf

# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    --type q8_0 \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-tokenizer-q8_0.gguf
```
