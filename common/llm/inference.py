"""
inference.py — LFM-8B + LoRA アダプタを使った推論スクリプト (transformers バックエンド)

使用法:
    from inference import load_model, predict_race
    tokenizer, model = load_model(base_model_id, lora_adapter_path)
    scores = predict_race(race_text, model, tokenizer)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time

import torch

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER_PATH = os.path.join(_THIS_DIR, "models", "lora_adapters")
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_INSTRUCTION = (
    "競馬レースを分析し、各馬のrank_scoreとev_scoreをJSON形式で出力してください。"
    '出力形式: {"scores":[{"horse_num":int,"rank_score":float,"ev_score":float}]}'
)


def _check_vram(required_gb: float = 8.0) -> None:
    """VRAM が required_gb 以上空いているか確認する。不足時は警告のみ（クラッシュしない）。"""
    if not torch.cuda.is_available():
        logger.warning("CUDA が利用できません。CPU 推論にフォールバックします（非常に低速）。")
        return
    free_bytes = torch.cuda.mem_get_info()[0]
    free_gb = free_bytes / 1024 ** 3
    logger.info("VRAM 空き: %.1f GB (必要推定: %.1f GB)", free_gb, required_gb)
    if free_gb < required_gb:
        logger.warning(
            "VRAM 不足の可能性 (空き %.1f GB < 必要 %.1f GB)。"
            "batch_size を下げるか、他の GPU プロセスを終了してください。",
            free_gb, required_gb,
        )


def load_model(
    base_model_id: str = BASE_MODEL_ID,
    lora_adapter_path: str = DEFAULT_ADAPTER_PATH,
    device: str = "cuda",
):
    """
    LoRAアダプタを適用したモデルをロードする。

    Args:
        base_model_id: HuggingFace モデルID
        lora_adapter_path: LoRA アダプタの保存先ディレクトリ
        device: 推論デバイス ("cuda" or "cpu")

    Returns:
        (tokenizer, model) タプル
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _check_vram(required_gb=8.0)
    logger.info("モデルロード: %s", base_model_id)
    logger.info("LoRAアダプタ: %s", lora_adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path if os.path.exists(
            os.path.join(lora_adapter_path, "tokenizer_config.json")
        ) else base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()

    logger.info("モデルロード完了")
    return tokenizer, model


def _build_prompt(race_text: str) -> str:
    """推論用プロンプトを構築する。"""
    return (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Input:\n{race_text}\n\n"
        f"### Response:\n"
    )


def _extract_json(text: str) -> dict | None:
    """
    生成テキストから JSON を抽出する。
    モデルが余分なテキストを出力してもパースできるよう、
    最初の { ... } ブロックを正規表現で探す。
    """
    # ネストされた {} を含む最初の JSON オブジェクトを探す
    brace_count = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = text[start_idx : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 不完全なJSONの場合、次の候補を探す
                    start_idx = None
                    brace_count = 0
    return None


def predict_race(
    race_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> list[dict]:
    """
    1レース分のスコア予測を行う。

    Args:
        race_text: race_to_text.race_to_prompt() の出力
        model: LoRAアダプタ適用済みモデル
        tokenizer: トークナイザ
        max_new_tokens: 最大生成トークン数
        temperature: 生成温度 (0.1推奨: 安定した構造化出力のため)

    Returns:
        [{"horse_num": int, "rank_score": float, "ev_score": float}, ...]
        パース失敗時は空リストを返す。
    """
    prompt = _build_prompt(race_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # 入力部分を除いた生成テキスト
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    # JSON パース
    parsed = _extract_json(generated)
    if parsed is None:
        return []

    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        return []

    # スキーマ検証
    result = []
    for item in scores:
        if not isinstance(item, dict):
            continue
        if "horse_num" not in item or "rank_score" not in item or "ev_score" not in item:
            continue
        horse_num = item["horse_num"]
        rank_score = item["rank_score"]
        ev_score = item["ev_score"]
        if not isinstance(horse_num, int):
            try:
                horse_num = int(horse_num)
            except (ValueError, TypeError):
                continue
        if rank_score is None or ev_score is None:
            continue
        result.append({
            "horse_num": horse_num,
            "rank_score": float(rank_score),
            "ev_score": float(ev_score),
        })

    return result


def _validate_scores(scores: list) -> list[dict]:
    """scores リストのスキーマ検証。"""
    result = []
    for item in scores:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("horse_num", "rank_score", "ev_score")):
            continue
        try:
            result.append({
                "horse_num": int(item["horse_num"]),
                "rank_score": float(item["rank_score"]),
                "ev_score": float(item["ev_score"]),
            })
        except (ValueError, TypeError):
            continue
    return result


def _generate_one_batch(
    prompts: list[str],
    model,
    tokenizer,
    max_new_tokens: int,
) -> list[str]:
    """1バッチ分の生成。OOM 時は例外を再送出する（呼び出し元でハンドル）。"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1300,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    return [
        tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
        for j in range(len(prompts))
    ]


def predict_batch(
    race_texts: list[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 768,
    race_ids: list[str] | None = None,
) -> list[list[dict]]:
    """複数レースを GPU バッチ処理で一括予測する（左パディング）。

    OOM 発生時はバッチサイズを自動で半減して再試行し、最終的に 1 件ずつに縮退する。
    JSON パース失敗時はそのレース分のみ空リストとして記録しスキップする（全体クラッシュしない）。

    Args:
        race_texts: race_to_prompt() の出力リスト
        batch_size: 初期バッチサイズ（VRAM 16GB なら 4 推奨）
        max_new_tokens: 最大生成トークン数（18頭 × 42 トークン ≈ 756 なので 768 以上必要）
        race_ids: ログ用レース ID リスト（省略時はインデックスで代替）

    Returns:
        各レースのスコアリスト（len == len(race_texts)）
    """
    from tqdm import tqdm

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    ids = race_ids if race_ids is not None else [str(i) for i in range(len(race_texts))]
    all_results: list[list[dict]] = []
    n_batches = (len(race_texts) + batch_size - 1) // batch_size
    parse_failures = 0
    t0 = time.monotonic()

    for batch_idx, i in enumerate(
        tqdm(range(0, len(race_texts), batch_size), total=n_batches, desc="バッチ推論")
    ):
        chunk_texts = race_texts[i : i + batch_size]
        chunk_ids   = ids[i : i + batch_size]
        prompts = [_build_prompt(t) for t in chunk_texts]

        # OOM 自動縮退: batch_size → batch_size//2 → 1 まで試みる
        current_bs = len(prompts)
        generated_texts: list[str] = []
        while current_bs >= 1:
            try:
                sub_results = []
                for sub_i in range(0, len(prompts), current_bs):
                    sub_prompts = prompts[sub_i : sub_i + current_bs]
                    sub_results.extend(
                        _generate_one_batch(sub_prompts, model, tokenizer, max_new_tokens)
                    )
                generated_texts = sub_results
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                prev = current_bs
                current_bs = max(1, current_bs // 2)
                logger.warning(
                    "OOM 発生 (batch=%d/%d, bs=%d→%d): VRAM キャッシュクリア後に再試行",
                    batch_idx + 1, n_batches, prev, current_bs,
                )
                if prev == 1:
                    logger.error(
                        "bs=1 でも OOM。レース %s をスキップします。", chunk_ids
                    )
                    generated_texts = [""] * len(prompts)
                    break

        # JSON パース & バリデーション
        for j, (gen_text, rid) in enumerate(zip(generated_texts, chunk_ids)):
            parsed = _extract_json(gen_text)
            if parsed is None:
                parse_failures += 1
                logger.warning(
                    "JSON パース失敗 (race_id=%s, batch=%d, item=%d): %s",
                    rid, batch_idx + 1, j,
                    gen_text[:120].replace("\n", " "),
                )
                all_results.append([])
                continue
            scores = parsed.get("scores", [])
            validated = _validate_scores(scores) if isinstance(scores, list) else []
            if not validated:
                logger.warning(
                    "スキーマ不正またはスコア空 (race_id=%s): raw=%s",
                    rid, str(scores)[:120],
                )
            all_results.append(validated)

    elapsed = time.monotonic() - t0
    logger.info(
        "推論完了: %d レース / %.1f 秒 (%.2f 秒/レース) / パース失敗 %d 件",
        len(race_texts), elapsed, elapsed / max(1, len(race_texts)), parse_failures,
    )
    if parse_failures > 0:
        failure_rate = parse_failures / len(race_texts) * 100
        level = logging.ERROR if failure_rate > 30 else logging.WARNING
        logger.log(level, "パース失敗率: %.1f%% (%d/%d)", failure_rate, parse_failures, len(race_texts))

    tokenizer.padding_side = orig_padding_side
    return all_results


if __name__ == "__main__":
    import argparse

    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    from race_to_text import load_race_data, race_to_prompt

    parser = argparse.ArgumentParser(description="LFM-8B レース予測推論")
    parser.add_argument("--year", type=int, default=2026, help="対象年")
    parser.add_argument("--race_id", type=str, help="特定のレースID")
    parser.add_argument(
        "--adapter_path",
        default=DEFAULT_ADAPTER_PATH,
        help="LoRAアダプタパス",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(lora_adapter_path=args.adapter_path)
    df_se, df_ra = load_race_data(args.year)
    ra_indexed = df_ra.set_index("race_id")

    if args.race_id:
        race_ids = [args.race_id]
    else:
        race_ids = list(df_se["race_id"].unique())[:3]

    for rid in race_ids:
        if rid not in ra_indexed.index:
            print(f"[WARN] {rid} が race_ra に見つかりません")
            continue
        horses = df_se[df_se["race_id"] == rid]
        ra_row = ra_indexed.loc[rid]
        race_text = race_to_prompt(ra_row, horses)
        print(f"\n=== {rid} ===")
        print(race_text)
        scores = predict_race(race_text, model, tokenizer)
        print("予測スコア:", json.dumps(scores, ensure_ascii=False, indent=2))
