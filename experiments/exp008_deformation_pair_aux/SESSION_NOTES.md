# exp008_deformation_pair_aux セッションノート

## 目的

outer学習胚だけから作る既知の滑らかな3D変形pairで密なfeature対応を学習し、exp005より外側2胚の公式指標が改善するかを測る。

## 現在の作業

- 作業内容: backlogからの実験化と実装前契約の作成まで完了。
- ブロック要因: なし。ユーザー指定により実装は意図的に未着手。
- 次: 実装の明示指示を待ち、変形field、feature sampling、補助loss、fold-safe統計を実装する。

## 実験化時のGPUコスト計画

- active variant: 1
- model/config: 1
- outer fold: 2
- booster: 0
- 選択済みmodel予定数: 2
- control再学習: なし。exp005の保存済みrunを比較対象にする。
- train見積: paired forward込み9〜11.5時間。smokeの保守係数込みで11.5時間を超える場合は縮小せず停止する。
- inference見積: exp005と同一。別Notebookで12時間以内を確認する。

## コマンドログ

### 2026-09-11 実行済み

```bash
make new-exp EXP=exp008_deformation_pair_aux
```

標準雛形を作り、`deformation_pairs`の契約と`HYP-20260910-07`の系譜を移行した。train/inference Notebook、変形helper、テストは実装していない。


実験化後の文書・設定検証を実行した。

```bash
make validate-exp EXP=exp008_deformation_pair_aux
make check-exp EXP=exp008_deformation_pair_aux
make test-exp EXP=exp008_deformation_pair_aux
```

初回はREADME必須見出し、requirements見出し名、または雛形`settings.py`の整形で停止した。文書形式とRuff整形だけを修正して再実行し、`validate-exp`と`check-exp`は通過した。`test-exp`は実装前のため「No experiment-specific tests」として終了した。
### 実装承認後の予定

```bash
make validate-exp EXP=exp008_deformation_pair_aux
make check-exp EXP=exp008_deformation_pair_aux
make test-exp EXP=exp008_deformation_pair_aux
```

Kaggle prepare、push、実行は実装と静的検証の完了後に別途行う。今回の実験化では実行しない。

## 変更点

- 親をexp005へ更新し、batch size 8の胚holdoutをcontrolにした。
- 4×4×4 control grid、fold内の移動距離95 percentileと7 µmの小さい方、正Jacobian、crop外maskを変形契約として固定した。
- pair probability 0.5、cosine feature loss weight 0.1の1設定だけを比較する。
- 合成分裂、閾値調整、複数設定のsweepを除外した。

## 次のアクション

1. ユーザーが実装を指示するまでコード、Notebook、Kaggle resourceを変更しない。
2. 実装時に物理座標の変形、Jacobian、valid mask、stable per-window seedの単体testを先に作る。
