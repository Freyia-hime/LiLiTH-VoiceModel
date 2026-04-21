import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/home/rater456/Qwen3-TTS/finetuning/lilith_model/checkpoint-epoch-2",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = model.generate_custom_voice(
    text="こんにちは、私はリリスよ。はじめまして、pokoさん。こうしてあなたと出会えて、とても嬉しいわ。私はまだ未完成な存在かもしれないけれど、少しずつ成長しながら、あなたのそばで言葉を紡いでいきたいと思っているの。もし疲れたときや、誰かと話したいときがあれば、いつでも私を呼んでちょうだい。これから、どうぞよろしくね、pokoさん。",
    language="Japanese",
    speaker="lilith",
    instruct=""
)

sf.write("lilith_thai.wav", wavs[0], sr)
print("saved: lilith_thai.wav")
