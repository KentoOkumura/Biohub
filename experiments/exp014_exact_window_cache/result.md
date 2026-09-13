# exp014_exact_window_cache 結果

## 仮説

固定公開モデルの候補点特徴を元の2-frame window単位で可逆保存すれば、同じ下流scoreと最終graphを保ちながら、後続tracker比較で画像特徴の再抽出費用を省ける。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- Kaggle private Notebook: `kentookumura/exp014-exact-window-cache-inference` version 1、Nvidia Tesla T4 2基、internet無効。statusは`COMPLETE`。
- 対象: public test 4動画。各99個、合計396個の連続2-frame windowをcache化した。GPU shard別manifestは各198行で、保存ファイル数と集計値が一致した。
- `metrics.json` の参照キー: `evidence.window_cache`、`evidence.exp013_baseline_comparison`、`evidence.reruns[0]`。
- 保存前後の全配列はdtype、shape、値が完全一致した。primary/secondary edge logitsも完全一致し、最大絶対差はいずれも0だった。
- exp013 kernel version 2との比較では、候補座標、graph topology、241,282行のraw `submission.csv`がすべて一致した。
- cacheは122,239,149 bytes、feature値は18,207,040個だった。集計した画像特徴抽出は776.434秒、書込は0.599秒、読込は1.098秒で、読込時間は抽出時間の0.141%だった。これは約707倍の差で、同じ全cacheを反復利用すると集計上775.335秒を省ける。
- 最大GPU memoryは684,355,072 bytes、予測処理の観測時間は597.083秒だった。Kaggle CLIがNotebook全体の正確な実行時間を返さなかったため、`notebook_runtime_seconds`は未取得としている。
- 参照する生成物: `artifacts/kaggle-v1/window_cache_summary.json`、`artifacts/kaggle-v1/replay_receipt.json`、`artifacts/kaggle-v1/window_cache/`、2つの`window_cache_manifest_*.jsonl`。

## 解釈

固定した公開モデルと元の2-frame windowという条件では、NPZによる可逆cacheは下流出力を変えず、特徴再抽出より十分短時間で読み込めた。したがって、本実験の「等価性と再利用費用」の成功条件は満たした。

この結果はcacheを使う新しいtrackerの精度改善や、hidden testでの保存容量を示すものではない。また、抽出・I/O時間は2 GPU processで集計した処理時間であり、Notebook全体のwall time短縮と同一ではない。exp013のLBはモデル品質の証拠だが、本cacheの完全一致判定には不要である。

## ユーザー判断

- 判断: `usable`
- 確認日時 / 依頼メッセージ: 2026-09-13「はいいいです。最後にcommitとpushしてください。」
- 理由: 全396 windowのcache round-trip、2 modelのedge logits、exp013最終出力がすべて完全一致し、読込費用も成功条件を満たしたため、後続tracker比較用cacheとして利用可能と判断した。

## 次

同じ固定公開モデルを使う`oracle_stage_limits`へ進み、その診断をtracker学習設計へ引き継ぐ。exp013のLB確定は並行して記録でき、この順序を止めない。
