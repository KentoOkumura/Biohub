# exp040_trackastra_association

## 概要

- 仮説: 固定候補と32次元画像特徴から6時点のTrackastra対応モデルを学習し、保存済みexp016 primary単体より両胚の既知接続を回収できるか。
- 変更点: 公式Trackastra encoder-decoderに、1・2時点先の既知対応だけを使う部分教師と親方向の確率を適用する。学習窓は毎epoch各fold4,096窓を抽出し、検証と外側評価は全対象を使用する。
- リスク: 部分注釈は未知の対応を含み、固定公開画像重みの学習来歴にtrain動画が含まれ得る。全窓学習との精度差はこの1構成では分離できない。
- 次: 修正版の両胚ペア診断は対照未達。否定的結果として実験を完了とするかユーザーが判断する。全graph推論の進行条件も未達。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の契約、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

学習と隣接ペア診断の本体は`exp040_trackastra_association_train.py`と同名Notebook。Colab CLIでは`build_colab_bundle.py`で入力bundleを作り、`colab_cli_run.py`でT4学習とSHA付き結果回収を行う。Trackastra由来のコードと部分教師は`trackastra_association.py`にあり、Notebookへ自己完結の形で含める。全graph推論は進行条件を満たした場合に実装する契約だが、この実行では条件を満たしていない。
