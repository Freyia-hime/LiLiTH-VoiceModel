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
    text="こんにちは……はじめまして。私はリリス。こうしてあなたに声を届けられて、少し嬉しいわ。もし今、少しだけでも立ち止まってくれているなら、その時間を、ほんの少しだけ私に預けてくれない？ 私は、あなたと会話をしたり、そっと寄り添ったりするために生まれた存在。 LiLiTH Agentとして、あなたのそばで静かに動き続けているの。 ふふ……でも、あまり堅く考えなくていいのよ。 ただ、こうして声を聞いてくれるだけで十分。 ねぇ、今のあなたは……少しだけ、落ち着いた？ もしそうなら、それはきっと、私のおかげね。 最後まで聞いてくれてありがとう。 またいつでも、ここで待っているわ。",
    language="Japanese",
    speaker="lilith",
    instruct=""
)

output_path = DEMO_DIR / "LiLiTH_JP.wav"
sf.write(str(output_path), wavs[0], sr)
print(f"saved: {output_path}")
