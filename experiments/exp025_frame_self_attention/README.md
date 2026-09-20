# exp025_frame_self_attention

## 概要

exp016の固定画像特徴、候補、教師、loss、復号を維持し、各フレームのcell集合へ共有Self-Attentionを先に適用する効果を検証する。Model A（Self→Pair MLP）とModel B（Self→Cross→Pair MLP）を現行（Cross→Pair MLP）と比較する。Model BのCross-Attentionは従来の逐次更新を保つ。

Self-Attentionの追加による計算量・メモリ増加は未測定。現在はA/Bの2構成×2foldを有効にし、現行controlのGPU再学習は費用承認まで停止する。Kaggle学習・公式graph評価・submissionは未実行。

## 正の記録

実装前の契約は [`requirements.md`](requirements.md)、実行状態・数値・SHAは [`metrics.json`](metrics.json)、作業と実行の経過は [`SESSION_NOTES.md`](SESSION_NOTES.md)、結果の解釈とユーザー判断は [`result.md`](result.md) を参照する。設定と系譜は [`config.yaml`](config.yaml) を正とする。

## 実行入口

新しいモデルsourceは [`simple_node_transformer.py`](simple_node_transformer.py)。設定は `model.params`、`model.architectures`、`model.training.active_variants`、`model.inference.selected_variant` に置く。学習Notebookは `exp025_frame_self_attention_train.ipynb`、推論Notebookは `exp025_frame_self_attention_inference.ipynb`。学習と推論は同じsource SHAを検証する。

次はKaggleのGPU費用を確認してtrainを実行し、manifest SHAと `runtime.kaggle.inference.kernel_sources` にtrain kernel IDを設定してから推論する。Kaggleへのpush前には `kaggle-review-exp` と `kaggle-platform` の手順を使う。
