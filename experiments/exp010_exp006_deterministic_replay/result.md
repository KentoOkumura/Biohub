# exp010_exp006_deterministic_replay 結果

## 仮説

augmentationの乱数をepochとitemへ安定に対応付け、PyTorch/CUDAの決定論的設定を強制すれば、固定したKaggle T4 x2環境で同じ2fold学習とheld-out評価を再実行したときにmodel tensor、予測内容、CVを一致させられる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 構造化された実行証拠: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- authoritative run 1のtrainはKaggle kernel `kentookumura/exp010-exp006-deterministic-replay-train` version 3で完了した。2foldのfull trainingは36,627.947秒、Notebook計測は36,712.660秒だった。
- 同じcheckpointを使うheld-out inferenceはkernel `kentookumura/exp010-exp006-deterministic-replay-inference` version 1で完了した。22,557.637秒で199動画を欠落なく推論し、保存graphからの公式metric再計算も一致した。
- run 1の胚holdout CVは全体0.4981248955、44b6が0.5995244749、6bbaが0.4798695306だった。親exp006の全体0.5533789422との差は-0.0552540467。
- checkpoint tensor、候補、OOF prediction、per-sample metric、公式summaryのcanonical content SHAは[`metrics.json`](metrics.json)の`evidence.artifacts`と`evidence.reruns`へ記録した。
- 2026-09-13 16:14 JSTのGPU残時間は6.16時間で、train run 2だけでも実測約10.20時間を要するため、quota gateにより追加pushを行っていない。
- 2回目のfull train/inferenceがないため、run間一致は未検証であり、再現性を担保済みとは扱わない。

## 解釈

決定論的設定を有効にした1回のfull train/inferenceが完走し、比較に必要なcanonical SHAを取得できることまでは確認した。ただしrun 1のscoreは親exp006より0.0552540467低く、この実験はLB改善を示していない。

現在の学習方針は公開検出器を固定し、下流trackerだけを学習するものである。exp010が再実行する旧exp006の検出器・tracker同時学習は現行路線の学習再現性を直接保証しない。したがって、約16.47 GPU時間と見込まれる追加のtrain/inferenceを使って完全一致を確認する価値は限定的であり、run 1の証拠を保持して停止することを推奨する。今後の学習再現性は、固定公開検出器を使う最初のtracker学習で同じ決定論化と再実行比較を適用する方が現行評価へ直接つながる。

## ユーザー判断

- 判断: 途中停止。実験statusは、このリポジトリ内で候補として使わないことを表す`discarded`。
- 確認日時 / 依頼メッセージ: 2026-09-13 / 「それではこれで進めてください」
- 理由: run 1の証拠は得られたが、旧exp006の検出器・tracker同時学習は現行の固定公開検出器路線を直接検証せず、run 2には約16.47 GPU時間が必要である。完全な再現性は未確認として明記し、追加GPUを現行路線へ回す。

## 次

run 1の証拠を履歴として保持し、追加実行は行わない。次は固定公開検出器の`exact_window_cache`、段階別診断、tracker学習の順に進め、tracker学習の再現性はその路線内で確認する。
