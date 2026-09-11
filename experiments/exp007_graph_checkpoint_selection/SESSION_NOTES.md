# exp007_graph_checkpoint_selection セッションノート

## 目的

exp005と同じ学習runで全epoch checkpointを保存し、内部選択動画に対する現行proxyと固定公式graph指標が選ぶcheckpointを外側2胚で比較する。

## 現在の作業

- 作業内容: backlogからの実験化と実装前契約の作成まで完了。
- ブロック要因: なし。ユーザー指定により実装は意図的に未着手。
- 次: 実装の明示指示を待ち、全epoch保存、2 selector、unique checkpoint推論を実装する。

## 実験化時のGPUコスト計画

- active training variant: 1
- model/config: 1
- outer fold: 2
- booster: 0
- 保存checkpoint予定数: 6
- selector: 2
- control再学習: なし。同一runのcheckpointへ両selectorを適用する。
- train見積: 7〜9時間、hard gate 11.5時間。
- inference見積: exp005の最大2倍を上限とし、smokeの保守係数込みで11.5時間を超える場合は停止する。

## コマンドログ

### 2026-09-11 実行済み

```bash
make new-exp EXP=exp007_graph_checkpoint_selection
```

標準雛形を作り、`graph_checkpoint`の契約と`HYP-20260910-14`の系譜を移行した。train/inference Notebookとテストは実装していない。


実験化後の文書・設定検証を実行した。

```bash
make validate-exp EXP=exp007_graph_checkpoint_selection
make check-exp EXP=exp007_graph_checkpoint_selection
make test-exp EXP=exp007_graph_checkpoint_selection
```

初回はREADME必須見出し、requirements見出し名、または雛形`settings.py`の整形で停止した。文書形式とRuff整形だけを修正して再実行し、`validate-exp`と`check-exp`は通過した。`test-exp`は実装前のため「No experiment-specific tests」として終了した。
### 実装承認後の予定

```bash
make validate-exp EXP=exp007_graph_checkpoint_selection
make check-exp EXP=exp007_graph_checkpoint_selection
make test-exp EXP=exp007_graph_checkpoint_selection
```

Kaggle prepare、push、実行は実装と静的検証の完了後に別途行う。今回の実験化では実行しない。

## 変更点

- 親をexp005へ更新し、batch size 8の胚holdoutを比較基準にした。
- 各foldのepoch index 0、1、2を保存し、同じ内部選択動画で2 selectorを比較する契約を固定した。
- outerラベルによる選択、閾値調整、追加selector探索を禁止した。

## 次のアクション

1. ユーザーが実装を指示するまでコード、Notebook、Kaggle resourceを変更しない。
2. 実装時にsplit独立性と公式評価器SHAを最初にtestで固定する。
