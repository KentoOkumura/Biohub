# exp030_frame_self_attention_diagnostics 結果

## 仮説

フレーム内Self-Attentionを加えるモデルを旧tracker checkpointと互換に導入でき、固定cacheの最大級windowでもKaggle T4でforwardとbackwardを実行できるか調べる。接続精度の仮説はこの診断では検証しない。

## 実行証拠

- Kaggle Notebook: [exp030 self attention diagnostic version 1](https://www.kaggle.com/code/kentookumura/exp030-self-attention-diagnostic)。`KernelWorkerStatus.COMPLETE`。private、Tesla T4、internet無効。
- 構造化された数値と生成物SHAの正: [`metrics.json`](metrics.json)の`frame_self_attention_diagnostic`。回収したJSONは`artifacts/kaggle_diagnostic_v1/frame_self_attention_diagnostic.json`で、SHA256は`metrics.json`の`report_sha256`と一致する。
- コマンドと入力検証: [`SESSION_NOTES.md`](SESSION_NOTES.md)。公開model sourceとcheckpointのSHA、exp015 cache summaryと19,701 windowのidentityを照合した。Notebookで使ったconfig、model source、Jupytext sourceはローカルの正とbyte一致。
- 旧checkpointを現行モードへ厳密に読み込み、公開実装とのlogits差は0。Model A/Bの共有Self Encoder初期stateも一致。mask・空集合・cell順序、A/Bの保存復元、有限なlogitsと勾配の確認が通った。

同じ固定cacheから選んだ代表windowは205×202 cell、最大windowは1,056×1,038 cell。各値は3回の中央値。メモリは計測前の確保量からの最大増分で、3モデルを同時にGPUへ置いた条件で測った。

| window | 構成 | 前向き計算 | 前向き＋勾配計算 | GPUメモリ最大増分 |
| --- | --- | ---: | ---: | ---: |
| 代表 | 現行 | 11.0 ms | 67.1 ms | 21.0 MiB |
| 代表 | Model A | 6.2 ms | 40.7 ms | 20.3 MiB |
| 代表 | Model B | 13.2 ms | 85.6 ms | 21.2 MiB |
| 最大 | 現行 | 65.9 ms | 255.8 ms | 106.8 MiB |
| 最大 | Model A | 60.8 ms | 229.4 ms | 103.2 MiB |
| 最大 | Model B | 69.3 ms | 278.1 ms | 107.8 MiB |

Notebook本体の計時は494.37秒で、そのうちcache identity走査は99.83秒。Kaggle quota表示の使用量差はアカウント全体で0.33時間であり、このNotebook単体のGPU割当消費時間とは断定しない。Kaggle割当全体の実行時間は別途未取得。

## 解釈

Model Bは旧モデルとの互換性・機能guardを満たし、最大windowでもOOMを起こさなかった。現行に対する前向き＋勾配計算の増加は代表windowで約28%、最大windowで約9%。最大windowのメモリ増分差は約1 MiBだった。この短い測定はbatch size 1、合成した勾配用の値、optimizer更新なしという条件であり、3エポック×2foldの学習時間や199動画のgraph復元時間を保証しない。

Model AはCross-Attentionを実行しないため、この診断では現行より短時間だった。精度上の優劣は測っていない。公式combined score、接続・分裂成分、CV、LB、submissionは未計測。公開初期モデルの学習来歴から、後続の胚holdout評価も独立したCVとは扱わない。

## ユーザー判断

- 実験の完了: 2026-09-20にユーザーが完了と判断した。診断の互換性・最大window実行可能性の受け入れ条件は満たした。
- Model A/Bの採否: 接続精度を測っていないため、この診断では判断しない。

## 次

後続の3構成学習、隣接2フレームの両胚別比較、公式graph評価は`exp031_frame_self_attention_spatial`で扱う。実行時のGPU残量は改めて確認する。Kaggle submissionは別途明示依頼まで行わない。
