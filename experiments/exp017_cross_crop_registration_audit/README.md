# exp017_cross_crop_registration_audit

## 概要

- 仮説要約: 同一胚のcropに共通視野があれば、複数時刻の画像登録が一貫し、共通するGEFF注釈も同じ変換で整合する。
- 変更点要約: 2Dの候補探索、D4変換を含む絞り込み、3D確認、GEFF注釈確認をCPUだけで実行するaudit Notebookを追加した。
- リスク: 重ならず隣接するcrop、任意の時間ずれ、非剛体変形はこの監査では復元できない。負の結果は非連結の証明にならない。
- 結果要約: Kaggle CPUで全199 crop、同一胚内10,613対を調べたが、2Dと3Dの両方を通る共通視野候補は0対だった。同じ非空の停止frame scheduleを持つ24対も0対だった。
- 判断: 実験を完了とし、crop間の軌跡を直接接続せずsample単位のtracker学習を維持する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- CPU監査 Notebook: `exp017_cross_crop_registration_audit_audit.ipynb`
- 正の編集対象: `exp017_cross_crop_registration_audit_audit.py`
- Kaggle準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)のコマンド記録に従う。
- Notebook実行: Kaggle CPU kernelを正式実行とする。ローカルでは合成データを使う実験固有テストだけを実行する。
- 正式実行: [Kaggle Notebook version 2](https://www.kaggle.com/code/kentookumura/exp017-cross-crop-registration-audit-audit)。

## 表記

画像間の位置合わせは image registration、時系列の点と辺はGEFF注釈と記す。独自の包括的名称は付けず、実際に行う比較を記述する。
