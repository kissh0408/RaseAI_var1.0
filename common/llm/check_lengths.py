import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
lengths = []
with open(r"C:\Users\syugo\AI\RaceAI\common\llm\data\train_dataset.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        d = json.loads(line)
        text = "### Instruction:\n" + d["instruction"] + "\n\n### Input:\n" + d["input"] + "\n\n### Response:\n" + d["output"]
        lengths.append(len(tokenizer.encode(text)))

lengths.sort()
n = len(lengths)
print(f"サンプル数: {n}")
print(f"最小トークン数: {lengths[0]}")
print(f"中央値: {lengths[n // 2]}")
print(f"90パーセンタイル: {lengths[int(n * 0.9)]}")
print(f"95パーセンタイル: {lengths[int(n * 0.95)]}")
print(f"最大: {lengths[-1]}")
