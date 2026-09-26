# exp048_x138_primary_past_feature_attention

## 概要

仮説: 直前フレームの候補8件の画像・位置特徴を、親・娘の組ごとに公開primary trackerのpair MLP内部へ渡す。primary全体と追加attentionを共同学習し、同じ条件でprimaryのみ再学習した対照と胚別に比較する。

- 変更点: `[t-1,t]` のsource特徴を候補IDで整列し、実時間の親を基準に8近傍を選ぶ。forwardとreverseの両方で組別attentionを使う。
- リスク: capture再生成と2条件×2foldのGPU費用、GEFFの部分注釈、公開画像モデルの学習来歴。pair診断は全graphの公式scoreを測らない。
- 次: ユーザー判断により実験は完了・不採用。今回の設計で全graph推論は行わず、残る多時点の別候補は現在の学習方針と個別の根拠に従って検討する。

## 正の記録

[要件と実装方法](requirements.md) · [設定と系譜](config.yaml) · [数値と実験status](metrics.json) · [結果の解釈](result.md) · [実行ログ](SESSION_NOTES.md)

## 実行入口

学習用に実装したNotebook: [exp048_x138_primary_past_feature_attention_train.ipynb](exp048_x138_primary_past_feature_attention_train.ipynb)。実行はユーザーの後続依頼により承認済み。推論Notebookは作成していない。
