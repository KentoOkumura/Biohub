# exp021_division_teacher_audit 結果

## 仮説

exp020で固定した母娘9 µm・娘間14 µmの組候補について、既知GEFF edgeから学習に使える誤組の件数と正例母との重なりを測った。学習・推論・公式scoreの比較は対象外。

## 実行証拠

[Kaggle private CPU Notebook](https://www.kaggle.com/code/kentookumura/exp021-division-teacher-audit-diagnostic)のversion 1とversion 2が199動画・19,701 windowを完走した。version 2は同じkernel id_no `134983138`で母のGT出力次数ごとの内訳を追加した。GPUとinternetは無効。version 2の診断処理は495.43秒で、Notebook全体の確定実行時間ではない。モデル学習、公式CV、LB、submissionは行っていない。

exp015 cache identity SHAは`440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`、GT内容bundle SHAは`05459d7c5b398690849590d1032c8883061bd3e26c9df617e10c6110c9dbf2d4`でexp020と一致した。version 2の199行の表SHAは`683725138ab575f4442e12eb9fbc13448663959a7ac2294e1715665698ba9834`、summary SHAは`5efbe85936643be7676a3014bf384b811e4d727f57730f27922631d719fed1ba`で、Kaggle manifestと取得した実ファイルが一致した。取得した小さな生成物はignored `artifacts/diagnostic_v1/`と`artifacts/diagnostic_v2/`へ版ごとに保存した。数値の正は[metrics.json](metrics.json)とKaggleのsummary.jsonとする。

| GT条件に基づく分類 | 全体 | 44b6 | 6bba |
| --- | ---: | ---: | ---: |
| 幾何候補の組 | 8,807,617 | 4,434,037 | 4,373,580 |
| 既知分裂の正例 | 100 | 20 | 80 |
| 既知分裂母で両娘候補がGT対応した誤組 | 4 | 2 | 2 |
| 別母への注釈edgeがある娘を含む誤組 | 3,933 | 1,893 | 2,040 |
| 上2種類の重複を除いた誤組 | 3,935 | 1,893 | 2,042 |
| 別母edgeによる誤組のうちGT出力2本の母 | 3 | 3 | 0 |
| 同、GT出力1本の母 | 3,811 | 1,882 | 1,929 |
| 同、GT出力0本の母 | 119 | 8 | 111 |
| 正例100件のうち、同じ母に別母edge由来の誤組があるもの | 1 | 1 | 0 |
| 正例100件のうち、同じ母に厳格な誤組があるもの | 2 | 1 | 1 |
| GT出力1本の母で、既知娘を含む未確定な組 | 89,703 | 28,777 | 60,926 |

GT出力2本の母に3件、1本の母に3,811件、0本の母に119件で計3,933件。厳格な誤組4件との重複は2件で、重複除去後は3,935件。別母edgeによる誤組は2,065の母候補に分布した。

## 解釈

既知2娘を持つ母と異なるGT対応組、および別母への注釈edgeを持つ娘を含む組は、GT edgeが正しく、7 µmの一対一対応が正しく、1娘に複数の母がない条件の下で、その**組そのもの**を誤りと判定できる。この定義なら正例100件、誤組3,935件、残りは未知とする教師maskを作れる。ただし3,935件は「通常継続を確定した母」ではない。単一GT出力edgeの母に既知娘を含む89,703組も第2娘が未注釈の可能性を残し、生物学的な非分裂負例にはしない。

誤組3,935件の大半はGT出力1本の母に付く。同じ母の正しい2娘組と誤組を比べられる正例母は、厳格な誤組で2/100件、別母edge由来では1/100件に限られた。したがって、この教師だけでは各母の候補2娘組を順位付けする学習を十分に支えない。異なる母間で正例と誤組を分類する学習も、注釈とGT対応で選ばれた組への偏りを別途検証する必要がある。

主催者の[データ仕様](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data)はGTが疎いと明記する。[評価仕様](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)では単一edgeがあるGT母の予測forkがFPになり得るが、局所前後時刻を含むgraph評価であり、本実験の個別組を全て確定FPと呼べない。exp020の既知分裂回収100/151件（66.23%）と同じ固定候補であり、教師の件数が増えてもcoverage上限や公式combined scoreは改善していない。

## ユーザー判断

2026-09-19にユーザーはこの診断実験の完了を判断した。元のdivision_tripletsの学習は、追加の教師を探してから設計する方針を選択した。正例と同じ母で比較できる誤組が最大2/100件に限られるため、現時点で損失や復号を確定せず、元候補は未着手のままとする。
