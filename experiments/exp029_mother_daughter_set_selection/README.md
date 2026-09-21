# exp029_mother_daughter_set_selection

## 概要

母ごとに空・1娘・2娘の候補集合を直接採点し、GEFFの既知接続に整合する集合の確率で学習する。娘の重複は集合全体の整数最適化で防ぐ。固定候補・画像特徴・胚別評価はexp028と揃える。

候補集合の増加による計算費用、既知娘の候補漏れ、疎い注釈での対応なし校正が主なリスク。Kaggleで2-frame診断を行い、事前条件が成立した場合だけ全graphへ進む。

## 正の記録

- [要件と実装方法](requirements.md)
- [設定と系譜](config.yaml)
- [実行記録](SESSION_NOTES.md)
- [数値と実行証拠](metrics.json)
- [結果と判断](result.md)

## 実行入口

- 学習: `exp029_mother_daughter_set_selection_train.ipynb`
- 全graph推論: `exp029_mother_daughter_set_selection_inference.ipynb`（2-frame条件成立後のみ）

- 分裂の保存済みモデル診断: [diagnose_divisions.py](diagnose_divisions.py)（入力準備・再現コマンドは[実行記録](SESSION_NOTES.md)）
