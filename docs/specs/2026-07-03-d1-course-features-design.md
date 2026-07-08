# 実装仕様書: D-1 コース系特徴量（弱点条件対策） — 2026-07-03

作成: planner
対象: `docs/2026-07-02-system-issues-and-improvements.md` セクション D-1
バージョン系: **v39**（v39_course / v39_waku / v39_pace の3実験）

---

## 0. 禁止事項の確認（実装前後に必ず参照）

- [ ] **オッズ・人気・市場情報を一切使わない**（単勝/複勝/馬連オッズ、人気順位、market_prob 等）
- [ ] **テストデータ（2025+）の結果で特徴量・閾値を後付け調整しない**。
      エラー分析（error_analysis_v38_best_time.json）は「どの軸を探すか」の指針としてのみ使用し、
      閾値・フラグ定義・センタリング定数などの設計判断はすべて**公知のコース形態情報**または
      **学習期間（〜2023-12-31）/ バリデーション（〜2024-12-31）のデータ**のみで行う
- [ ] **リーク停止閾値**: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluator へ報告
- [ ] 実装前後に混入チェックを実行:
      `grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"`
- [ ] `features_v38_best_time.parquet` と `pure_rank/models/` を**上書き前にバックアップ**（禁止事項 9）

---

## 1. 目的と根拠データ

### 1.1 現行正式ベースライン

| 指標 | v38_best_time（131列） |
|------|----------------------|
| Top-1 | **30.16%**（4,775レース、2025-01-05〜2026-05-24） |
| Top-3 | 61.63% |
| NDCG@3 | 0.5353 |
| Spearman | 0.5041 |

### 1.2 弱点条件（error_analysis_v38_best_time.json、全体 30.16% 比）

| 条件 | Top-1 | 差分 | n |
|------|-------|------|---|
| **course_code=3（福島）** | **22.44%** | **-7.7pp** | 312 |
| **course_code=10（小倉）** | **23.24%** | **-6.9pp** | 383 |
| 17頭以上 | 25.65% | -4.5pp | 421 |
| 短距離（〜1400m） | 27.52% | -2.6pp | 1,675 |
| 13-16頭 | 28.62% | -1.5pp | 2,977 |
| （参考・強い側）5-8頭 | 40.43% | +10.3pp | 277 |
| （参考・強い側）稍重 | 33.21% | +3.1pp | 834 |

v33 時点と同じ弱点分布が v38 でも持続しており、一過性ではなく構造的な弱点である。
福島+小倉で 695 レース = テストの 14.6%。ここが +5pp 改善すれば全体 +0.7pp 相当。

### 1.3 推定原因（レポート D-1）

1. **コース実績のサンプル不足**: 福島・小倉は開催期間が短く、
   `hist_same_course_win_rate` / `hist_jockey_course_win_rate` の NaN 率が高い
   （コース単位の集計ではローカル場の適性情報が薄い）
2. **小回りコースの先行有利バイアス**: 直線が短いコースでは先行馬が残りやすい。
   市場は織り込んでいるがモデルには「コースの物理形態」が未提供
   （course_code はカテゴリとして存在するが、場をまたいだ一般化ができない）
3. **多頭数・短距離の展開/枠の影響**: テン争いの激化・枠バイアスの情報が不足

---

## 2. 設計方針

### 2.1 course_code カテゴリとの冗長性について（正直な評価）

コース静的属性（直線長・小回りフラグ）は **course_code の決定的関数**であり、
情報理論的には既存のカテゴリ特徴量 `course_code` に対して新情報を持たない。
それでも追加する価値は次の2点にある：

1. **場をまたいだ証拠のプーリング**: 木が `course_code ∈ {1,2,3,10}` という分割を
   学習するには各場のサンプルが必要だが、福島・小倉はまさにサンプルが薄い。
   「小回りフラグ」「直線長」という共有軸を与えれば、札幌・函館・福島・小倉の
   4場分の証拠を1つの分割で共有できる（= 弱点の根本原因であるデータ疎性への直接対策）
2. **交互作用の明示化**: 「小回り × 先行傾向」は depth-2 の分割で表現可能ではあるが、
   LambdaRank の勾配はレース内相対比較に基づくため、積として明示した方が発見が容易

したがって**静的属性単独では効果を主張しない**。静的属性 + 交互作用 + プーリング勝率を
1つの仮説バンドル（実験1）として検証する。

### 2.2 Phase A 失敗パターンとの照合（重複リスク管理）

Phase A の教訓: 既存特徴量と r=0.74 以上の候補は全滅
（脚質系は `hist_front_running_pref` と r=0.74〜0.79 で飽和）。

本仕様の全候補は「既存列との積・ゲーティング・別ソース集計」であり生の脚質再集計ではないが、
**積特徴量は片方の因子と高相関になりやすい**ため、以下の相関ゲートを必須とする：

> **相関ゲート**: 学習+バリデーション期間（race_date <= 2024-12-31）の行のみで、
> 新特徴量と既存131列の Pearson 相関を計算し、**|r| >= 0.7 の列が1つでもあれば
> 学習に進まず planner に報告**する（設計を修正するか候補を破棄する）。
> 2025+ の行はこの計算に使わない。

各候補の想定相関は候補ごとの節に記載する。

### 2.3 リーク防止の区別（明記）

| 種別 | 該当特徴量 | リーク防止 |
|------|-----------|-----------|
| 静的定数テーブル | course_straight_len, course_is_small | **定数なのでリークと無関係**（レース前に確定している公知の物理情報） |
| 既存列同士の積 | front_pref_x_small, front_pref_x_pace | 因子が既にリークセーフ（shift(1) 済み / レース前確定情報）なら積もセーフ |
| 時系列集計 | hist_track_size_win_rate | groupby + `shift(1).expanding()`（既存 hist_same_course_win_rate と同型） |
| 時系列集計（日次） | hist_waku_course_bias_ts | **日次集計 → cumsum → shift(1)** → merge（既存 J-4: hist_jockey_course_win_rate と同型）。同日の他レース結果を含めない |
| 学習期間定数 | front_pref_x_pace のセンタリング定数 c | **race_date <= 2023-12-31（train_end）の行のみ**から計算し manifest に記録 |

---

## 3. コース形態定数テーブル（実験1で使用）

JRA 公表の公知情報（直線長）。市場情報ではない。
`create_features.py` のモジュール定数 `COURSE_GEOMETRY` として定義する
（チューニング対象ではないため train_config.json ではなくコード内定数でよい。docstring に本仕様書を参照させる）。

| course_code | 場 | 芝直線(内) [m] | 芝直線(外) [m] | ダ直線 [m] | is_small |
|---|---|---|---|---|---|
| 1 | 札幌 | 266.1 | — | 264.3 | **1** |
| 2 | 函館 | 262.1 | — | 260.3 | **1** |
| 3 | 福島 | 292.0 | — | 295.7 | **1** |
| 4 | 新潟 | 358.7 | 658.7 | 353.9 | 0 |
| 5 | 東京 | 525.9 | — | 501.6 | 0 |
| 6 | 中山 | 310.0 | 310.0 | 308.0 | 0 |
| 7 | 中京 | 412.5 | — | 410.7 | 0 |
| 8 | 京都 | 328.4 | 403.7 | 329.1 | 0 |
| 9 | 阪神 | 356.5 | 473.6 | 352.7 | 0 |
| 10 | 小倉 | 293.0 | — | 291.3 | **1** |

- **is_small の定義基準（後付け調整の余地を残さないため固定）**: 「芝直線長 < 300m」。
  該当は札幌・函館・福島・小倉の4場のみ。中山(310m)・新潟内(358.7m)は該当しない。
  この基準は幾何情報のみに基づく（テスト成績を見て場を選んだものではない）
- 直線長の小数は概算値でよい（モデルは順序情報しか使わない）。
  実装時に JRA 公式サイトの値と大きく食い違う場合のみ修正する

### 内回り/外回りの判別（track_code の利用）

- `track_code` は RA 由来で merged df に存在する（現在 `FORBIDDEN_COLS` によりメタ列として
  特徴量から除外されているが、**派生特徴量の計算ソースとして使うことは問題ない**。
  レース前確定情報であり市場情報でもない。派生列は別名で作るので除外リストとは干渉しない）
- 判別ルール（芝のみ。ダートに内外の別はない）:
  - `track_code ∈ {12, 18}`（左外 / 右外）→ 外回り値を使用（新潟・京都・阪神で有効）
  - `track_code == 10`（芝・直線）→ `course_straight_len = distance`（新潟直線1000m）
  - それ以外（11, 17 等の内回り・判別不能コード 13/14/19/20 等）→ 内回り値にフォールバック
- **実装時の検証**: track_code の value_counts をログ出力し、想定外コード
  （20〜22 は現行 `surface_code = track_code // 10` の導出では芝なのにダート扱いになる
  既知の懸念がある）の件数を確認する。想定外コードが多数あれば planner に報告

---

## 4. 候補特徴量仕様

### 実験1: v39_course — 小回りコース×先行バイアス（4列、優先度1）

**仮説**: 直線が短い小回りコースでは先行馬が残りやすく、かつローカル4場の
コース適性はプーリングしないとサンプル不足になる。

| # | 特徴量名 | 定義 | 種別 |
|---|---------|------|------|
| 1 | `course_straight_len` | セクション3のテーブル + track_code 外回り判別。surface_code ∉ {1,2} は NaN | 静的 |
| 2 | `course_is_small` | course_code ∈ {1,2,3,10} → 1、他 0（int8。カテゴリ指定不要） | 静的 |
| 3 | `front_pref_x_small` | `hist_front_running_pref × course_is_small`（pref が NaN なら NaN のまま。fillna しない） | 交互作用 |
| 4 | `hist_track_size_win_rate` | `df.groupby(["ketto_num", "course_is_small"])["is_win"].transform(lambda x: x.shift(1).expanding().mean())`（既存 hist_same_course_win_rate と同型。小回り4場/主要6場をプールした馬のコースサイズ適性） | 時系列 |

**実装位置**: #1, #2, #3 は `_build_current_features`（course_is_small を先に作る必要があるため
関数冒頭で静的列を生成 → #3 は hist_front_running_pref 生成後なので SECTION 4 の位置で問題ない）。
#4 は `_build_hist_features` 内の馬場適性系ブロックに追加するが、**course_is_small に依存する**ため、
静的列の生成を `_build_hist_features` より前（例: `_load_data` 直後のフィルタ後）に移すのが簡潔。
関数分割は implementer の裁量（依存順序だけ守ること）。

**想定相関（実装時に相関ゲートで実測）**:
- course_straight_len vs course_code: カテゴリなので Pearson 対象外。数値列とは低相関見込み
- front_pref_x_small vs hist_front_running_pref: 小回りレースは全体の約15〜20%なので
  ゲーティング積の相関は r≈0.4〜0.5 見込み（Phase A の r=0.74 とは構造が異なる）。ゲートで確認
- hist_track_size_win_rate vs hist_same_course_win_rate / hist_win_rate: r=0.5〜0.65 見込み。
  **|r| >= 0.7 なら本列のみ落として3列で学習**（バンドル全体は破棄しない）

**期待改善**: Top-1 +0.1〜0.3pp（福島・小倉 22〜23% の底上げが主経路）

---

### 実験2: v39_waku — コース×距離帯×枠の時系列枠バイアス（1列、優先度2）

**仮説**: 短距離・多頭数で効く「コース別の枠有利不利」は現行の
`wakuban_surface`（符号付き枠番）と `relative_post_position`（枠/頭数）では表現できない。
実際の過去の結果から推定した枠バイアスは別ソースのシグナルである。

| 特徴量名 | `hist_waku_course_bias_ts` |
|---|---|
| 集計キー | `(course_code, surface_code, distance_category, wakuban)` |
| 集計値 | **超過勝率** = `is_win − 1/horse_count` の累積平均（頭数の違いでベースレート 1/n が変わるため、生の勝率ではなく期待値との差を使う） |
| パターン | 日次集計 → cumsum → shift(1) → merge（既存 J-4 `hist_jockey_course_win_rate` と完全同型） |
| 最小サンプル | `cum_races_prev < 50` → NaN（モジュール定数 `MIN_WAKU_SAMPLES = 50`。MIN_JOCKEY_RACES と同じ流儀） |

実装擬似コード（J-4 の写経で足りる）:

```python
wk_daily = (
    df.assign(_excess=df["is_win"] - 1.0 / df["horse_count"])
    .groupby(["course_code", "surface_code", "distance_category", "wakuban", "race_date"],
             observed=True)
    .agg(d_excess=("_excess", "sum"), d_races=("_excess", "count"))
    .reset_index()
    .sort_values([...キー..., "race_date"])
)
grp = wk_daily.groupby([...キー...], observed=True)
wk_daily["cum_excess_prev"] = grp["d_excess"].cumsum().pipe(lambda s: grp_shift...)  # J-4 と同型
# hist_waku_course_bias_ts = cum_excess_prev / cum_races_prev（閾値未満 NaN）→ キー+race_date で merge
```

**セル数の見込み**: 10場 × 2馬場 × 4距離帯 × 8枠 = 640 セル。学習期間 40万行超に対して
十分に密（1セル平均 600行超）。ただし距離帯×場の組み合わせに存在しないセルがあるのは正常。

**想定相関**: wakuban / wakuban_surface / relative_post_position と r=0.3〜0.5 見込み
（枠バイアスは枠番に対して単調とは限らない。ダートの内枠不利・芝短距離の内枠有利など非単調）。
ゲートで確認。

**期待改善**: Top-1 +0〜0.2pp（短距離 27.5%・17頭+ 25.7% の底上げが主経路）

---

### 実験3: v39_pace — テン争い激化ペナルティ（1列、優先度3）

**仮説**: 先行馬が多いレースでは先行脚質の期待値が下がる（ペース激化）。
17頭以上のレースで特に効く。

| 特徴量名 | `front_pref_x_pace` |
|---|---|
| 定義 | `hist_front_running_pref × (density_others − c)` |
| density_others | 自馬を除いたレース内の先行傾向平均: `(Σ_j pref_filled_j − pref_filled_i) / (horse_count − 1)`、pref_filled = `hist_front_running_pref.fillna(0)`（既存 field_front_runner_density の自馬除外版） |
| センタリング定数 c | **race_date <= 2023-12-31（train_end）の行のみ**の density_others 平均。生成時に manifest / ログへ記録する |
| NaN 規則 | 自馬の hist_front_running_pref が NaN なら本列も NaN |

**なぜ自馬除外とセンタリングが必須か（Phase A 教訓の直接適用)**:
素朴な積 `pref × field_front_runner_density` は、density にレース間分散が小さく
かつ自馬の pref が density に含まれるため、**pref 自体と r > 0.9 になる**ことがほぼ確実
（Phase A 失敗パターンの再演）。自馬除外で自己相関成分を消し、センタリングで積の符号を
反転可能にする（先行馬過多レースで負、先行馬手薄レースで正）ことで pref との相関を落とす。

**想定相関**: hist_front_running_pref と r=0.2〜0.5、field_front_runner_density と r=0.3〜0.5 見込み。
**ゲート必須**。|r| >= 0.7 なら破棄（これ以上の変換は複雑化に見合わない）。

**期待改善**: Top-1 +0〜0.15pp（17頭+ の底上げが主経路。レポート D-1 / R-4 でも
「D-1 の実験に相乗りさせず別実験として」と指定済み）

---

### 不採用候補（検討済み）

| 候補 | 判断 | 理由 |
|------|------|------|
| 右/左回りフラグ・坂の有無 | 見送り | is_small・直線長と情報が重複し、course_code カテゴリで代替可能。列数増加に見合わない。実験1が合格し弱点が残る場合に再検討 |
| 馬×コース勝率の再細分化 | 不採用 | hist_same_course_win_rate / hist_same_course_dist_win_rate が既存。細分化は NaN 率をさらに上げる（弱点の原因の悪化） |
| 当日馬場バイアス（same_day_front_bias） | 本仕様の範囲外 | レポート R-4。リーク設計が繊細（同日 race_num 前方のみ集計）なので独立した仕様書で扱う |

---

## 5. 実験プロトコル（1変更1実験）

```
実験1: v39_course（4列追加）     ← ベース: v38_best_time
   ↓ evaluator 合否判定
実験2: v39_waku（1列追加）       ← ベース: 実験1合格なら v39_course、不合格なら v38_best_time
   ↓ evaluator 合否判定
実験3: v39_pace（1列追加）       ← ベース: その時点の採用ベースライン
```

- **バンドル判断の根拠**: 実験1の4列は「小回り×先行」という単一仮説の構成要素であり、
  かつ #3, #4 は #2 に依存するため同一実験とする。実験2・実験3は独立仮説なので分離する
- 実験1が「合格だが僅差（+0.1pp 未満）」の場合、evaluator の判断で
  feature importance を確認し寄与ゼロの列を落とした再実験を1回まで許可する
- 各実験は 5シード×3フォールドのフルアンサンブル（15モデル）で評価する。
  シード・フォールド・label_gain 等の学習設定は**一切変更しない**（train_config.json の
  features_version のみ変更）

---

## 6. 評価基準

### 合否判定（各実験共通、テスト 2025-01-01 以降・4,775レース想定）

| 指標 | 合格 | 不合格 |
|------|------|--------|
| Top-1 | **> 30.16%**（v38_best_time 超え） | <= 30.16% |
| NDCG@3 | >= 0.5323（v38 比 -0.003 以内） | < 0.5323 |
| Spearman | >= 0.4991（v38 比 -0.005 以内） | < 0.4991 |

3条件すべて満たして合格。Top-1 が改善しても NDCG@3 / Spearman が許容幅を超えて
悪化する場合は不合格（ランキング全体の質を犠牲にしない）。

### リーク停止閾値

**Top-1 > 40% または Spearman > 0.6 → 即座に停止し evaluator へ報告**（合格ではなく危険信号）。

### 副次評価（合否には使わない。方向性の確認のみ）

合格した実験について `analyze_errors.py` を実行し（`error_analysis_v39_*.json`）、
v38 の弱点条件の Top-1 変化を記録する：

| 条件 | v38 基準値 | 期待方向 |
|------|-----------|---------|
| course_code=3（福島） | 22.44% | 改善 |
| course_code=10（小倉） | 23.24% | 改善 |
| 17頭以上 | 25.65% | 改善（特に実験2・3） |
| 短距離〜1400m | 27.52% | 改善（特に実験2） |
| 5-8頭 / 稍重（強い側） | 40.43% / 33.21% | **悪化していないこと** |

弱点条件が改善していなくても全体指標が合格なら採用する（条件別 n は小さく、
条件別の数値をゲートにすると多重検定的な後付け選択になるため）。

---

## 7. implementer への引き渡し事項

### 実装対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `pure_rank/src/create_features.py` | ① モジュール定数 `COURSE_GEOMETRY`（セクション3のテーブル）と `MIN_WAKU_SAMPLES = 50` を追加 ② 実験1: 静的列2 + 交互作用1 + 時系列1（依存順序: course_is_small → hist_track_size_win_rate / front_pref_x_small） ③ 実験2: `hist_waku_course_bias_ts`（J-4 と同型の日次 cumsum + shift(1)） ④ 実験3: `front_pref_x_pace`（自馬除外 + 学習期間定数センタリング、定数はログと manifest に記録） |
| `pure_rank/config/train_config.json` | 各実験の生成前に `features_version` を `v39_course` → `v39_waku` → `v39_pace` と変更（他の設定は変更禁止） |

`common.py` の FORBIDDEN_COLS は変更不要（track_code は除外のまま。派生列は別名なので影響なし）。

### 手順（実験1の例。実験2・3も同様）

1. バックアップ確認: `features_v38_best_time.parquet` が存在すること、
   `pure_rank/models/` の 15 モデルを `pure_rank/archive/models_v38_best_time/`（または既存の
   バックアップ規約に従う場所）へ退避
2. `train_config.json` の features_version を `v39_course` に変更
3. `create_features.py` に実験1の4列を実装 → 実行 → `features_v39_course.parquet` +
   `manifest_v39_course.json` 生成
4. **生成時検証**（学習前。すべて race_date <= 2024-12-31 の行のみで実施）:
   - track_code の value_counts をログ出力（想定外コードの件数確認）
   - 新列の NaN 率・基本統計をログ出力
   - **相関ゲート**: 新列 × 既存131列の Pearson |r| >= 0.7 が1つでもあれば学習せず planner へ報告
     （hist_track_size_win_rate のみ例外規定あり: セクション4実験1参照）
   - 新列と同一レースの is_win の「当該レース内」相関が異常に高くないこと
     （時系列列のリーク簡易チェック。目安 |r| < 0.15）
   - 市場情報 grep チェック（セクション0のコマンド）
5. 学習: `python pure_rank/src/train.py --ensemble`（15モデル）
6. 評価: `python pure_rank/src/evaluate.py` → セクション6の合否判定を evaluator に依頼
7. 合格時のみ: `python pure_rank/src/analyze_errors.py` で `error_analysis_v39_course.json` を
   生成し、弱点条件の変化を記録
8. **不合格時のロールバック**: features_version を直前の採用バージョンに戻し、
   退避したモデルを `pure_rank/models/` に復元。実験結果（指標・相関・importance）を
   ログとして残してから次実験へ

### 実装上の注意

- 既存パターンの流用元: 静的列 → `_build_current_features` の wakuban_surface、
  hist_track_size_win_rate → `_build_hist_features` の hist_same_course_win_rate（197行付近）、
  hist_waku_course_bias_ts → Step J-4 `hist_jockey_course_win_rate`（603〜627行付近）
- merge 後の行順序: create_features.py 末尾の
  `sort_values(["race_date", "race_id", "horse_num"])` が group 配列の前提。merge で行順が
  崩れても最終ソートで回復するが、**行数が merge で増えていないこと**（キー重複による
  ファンアウトがないこと）を assert すること
- course_is_small は 0/1 の数値列として扱い、categorical には追加しない
  （train_config.json の categorical リストは変更しない）
- 実験間で features_version 以外の設定を変えない。複数列の同時追加は実験1のバンドルのみ

### 期待改善の合計（参考）

実験1〜3合計で Top-1 +0.1〜0.5pp（レポート D-1 の見積り +0.3〜0.6pp のうち、
確実性の高い範囲）。市場ベンチマーク（1番人気 ≈31%）超えに向けた主戦場。
