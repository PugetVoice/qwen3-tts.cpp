
## Qwen3 TTS

Download the model
```
# 0.6B
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# 1.7B
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

Convert TTS model (transformer + speaker encoder + tokenizer)
```
# 0.6B
python scripts/convert_tts_to_gguf.py \
    --type q4_k \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-0.6b-base-q4_k.gguf

# 1.7B
python scripts/convert_tts_to_gguf.py \
    --type q4_k \
    --input models/Qwen3-TTS-12Hz-1.7B-Base \
    --output models/qwen3-tts-1.7b-base-q4_k.gguf
```

Convert the tokenizer to GGUF format, it's shared by both the 0.6B and 1.7B models.
```
# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    --type q8_0 \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-tokenizer-q8_0.gguf
```
