# exp023_synthetic_detector_teacher_audit

## 概要

- 仮説要約: 固定公開検出器が合成動画から出す候補と画像特徴に、完全合成系譜の教師を対応付けられるか診断する。
- 変更点要約: exp022のGT中心を検出中心に置き換え、先頭2時系列の確認後、同じ条件で先頭32時系列へ広げて候補と教師を数える。
- リスク: 合成画像と実画像のtexture・contrast差、Zarr quantile属性がない前処理差、固定閾値で分裂候補が減ること。
- 次: 合成教師を用いる学習の教師mask、loss、復号、実データでの評価条件を設計する。

## 正の記録

- 契約と受け入れ条件: [requirements.md](requirements.md)
- 設定と系譜: [config.yaml](config.yaml)
- 数値と実行証拠: [metrics.json](metrics.json)
- 結果と限界: [result.md](result.md)
- 実行ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 診断Notebook: [exp023_synthetic_detector_teacher_audit_diagnostic.ipynb](exp023_synthetic_detector_teacher_audit_diagnostic.ipynb)
