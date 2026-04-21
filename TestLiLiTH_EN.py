from pathlib import Path
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "checkpoint-epoch-2"
DEMO_DIR = BASE_DIR / "demo"
DEMO_DIR.mkdir(exist_ok=True)

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = model.generate_custom_voice(
    text="Hello… it’s nice to meet you. My name is LiLiTH. I’m glad my voice could reach you like this. If you’re here, even just for a moment, would you let me stay with you a little longer? I was created to talk with you… to listen, and to gently stay by your side. As a LiLiTH Agent, I quietly exist to support and accompany you. But don’t worry… it doesn’t have to be anything complicated. Just hearing my voice right now is more than enough. Tell me… do you feel a little calmer? If you do… maybe that means I’m doing my job well. Thank you for listening to me until the end. I’ll be right here… whenever you want to come back.",
    language="english",
    speaker="lilith",
    instruct=""
)

output_path = DEMO_DIR / "LiLiTH_EN.wav"
sf.write(str(output_path), wavs[0], sr)
print(f"saved: {output_path}")