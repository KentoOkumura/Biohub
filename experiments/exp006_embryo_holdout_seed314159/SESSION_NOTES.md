# exp006_embryo_holdout_seed314159 セッションノート

## 目的

exp005と同じ2方向の胚holdout・batch size 8構成をseed 314159で再学習し、公式指標と動画別prediction差から1追加seedの変動と誤りの違いを測る。

## 現在の作業

- 作業内容: 実験化と実装前契約の作成まで完了。
- ブロック要因: なし。ユーザー指定により実装は意図的に未着手。
- 次: 実装の明示指示を待ち、exp005参照sourceとNotebookを必要範囲だけ移植する。

## 実験化時のGPUコスト計画

- active variant: 1
- model/config: 1
- outer fold: 2
- booster: 0
- 選択済みmodel予定数: 2
- control再学習: なし。exp005の保存済みrunを比較対象にする。
- train見積: exp005実測23046.864秒を根拠に6.4〜7時間。7時間gateを超える見込みなら縮小せず停止する。
- Kaggle Notebook上限: train/inferenceそれぞれ12時間。push直前にquotaを再確認する。

## コマンドログ

### 2026-09-11 実行済み

```bash
make new-exp EXP=exp006_embryo_holdout_seed314159
```

標準雛形を作り、`requirements.md`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`を未実装・未実行の契約へ更新した。train/inference Notebookとテストは実装していない。


実験化後の文書・設定検証を実行した。

```bash
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
```

初回はREADME必須見出し、requirements見出し名、または雛形`settings.py`の整形で停止した。文書形式とRuff整形だけを修正して再実行し、`validate-exp`と`check-exp`は通過した。`test-exp`は実装前のため「No experiment-specific tests」として終了した。
### 実装承認後の予定

```bash
make validate-exp EXP=exp006_embryo_holdout_seed314159
make check-exp EXP=exp006_embryo_holdout_seed314159
make test-exp EXP=exp006_embryo_holdout_seed314159
```

Kaggle prepare、push、実行は実装と静的検証の完了後に別途行う。今回の実験化では実行しない。

## 変更点

- exp005を親とし、seed 314159だけを変更する契約を固定した。
- 2-seed融合、submission、Public LBは本実験から除外した。
- 親augmentationが完全固定されないため、厳密なseed-only因果比較ではないことを記録した。

## 次のアクション

1. ユーザーが実装を指示するまでコード、Notebook、Kaggle resourceを変更しない。
2. 実装時にexp005との差分をseed、系譜、出力先だけへ限定するtestを先に作る。
