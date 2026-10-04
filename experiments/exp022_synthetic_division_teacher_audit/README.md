# exp022_synthetic_division_teacher_audit

## 概要

- 仮説要約: 公開合成系譜の先頭32時系列なら、完全な生成系譜から分裂組・誤組・通常継続の教師量を測れる。
- 変更点要約: exp021の疎い実データGTを完全な合成node・edge・division配列へ替える。
- リスク: 合成画像の実画像への転移、固定検出器による候補回収、学習効果はこの診断から分からない。
- 次: 固定公開検出器の候補・特徴と合成教師の対応診断は[exp023](../exp023_synthetic_detector_teacher_audit/result.md)で実施済み。そこで得た教師量と転移・学習の未検証事項を後続設計へ引き継ぐ。

## 正の記録

- 契約と範囲: [requirements.md](requirements.md)
- 設定と系譜: [config.yaml](config.yaml)
- 数値と実行証拠: [metrics.json](metrics.json)
- 結果と判断: [result.md](result.md)
- 実行ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 診断Notebook: [CPU診断Notebook](exp022_synthetic_division_teacher_audit_diagnostic.ipynb)
