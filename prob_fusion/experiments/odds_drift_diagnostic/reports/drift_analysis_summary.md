# 時系列オッズドリフト 探索的分析レポート

**注意**: この分析で使用した時系列オッズは2025-2026年分のみ(JV-Link保持期間1年の制約)。この期間は evaluation/splits.py の fold3 TEST期間と完全に重複しており、fold3 TESTは既にL1/L2の正式合否判定に使用済みの神聖な期間である。したがって本結果は正式な検証ではなく、あくまで仮説を潰す/残すための探索的分析である。正式なα再フィット・ゲート判定は今後の週次自動取得(RaceAI_WeeklyOddsTS)で新規に溜まるプロスペクティブなデータでのみ行うこと。

- n_races (merged) = 4775
- n_horses (merged) = 66020

## Drift descriptive stats

              drift  drift_late
count  66020.000000     66020.0
mean       0.273056         0.0
std        1.615489         0.0
min       -6.030685         0.0
1%        -3.355669         0.0
5%        -2.370271         0.0
25%       -0.823767         0.0
50%        0.268694         0.0
75%        1.351485         0.0
95%        2.941429         0.0
99%        4.147574         0.0
max        6.612175         0.0


## Bin summary (drift tercile within q_final decile)

       drift_tercile      n  win_rate  mean_q_final
0  down(popularized)  22012  0.073505      0.072845
1               flat  22003  0.072945      0.072203
2  up(unpopularized)  22005  0.070848      0.071932


chi2=1.286, p=0.52577, dof=2
