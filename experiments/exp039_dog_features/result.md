# exp039_dog_features 結果

## 仮説

元解像度の2組のDoG応答を固定点の特徴へ加えると、同構造のゼロ入力対照より両胚の接続と公式combined scoreが改善し、分裂成分を悪化させない。

## 記録の参照先

- 設定と系譜: [config.yaml](config.yaml)
- 数値、実験status、生成物SHA: [metrics.json](metrics.json)
- 実行コマンドと時系列: [SESSION_NOTES.md](SESSION_NOTES.md)
- 事前に固定した比較と進行条件: [requirements.md](requirements.md)

## 実行証拠

[Kaggle CPU特徴生成kernel](https://www.kaggle.com/code/kentookumura/exp039-dog-features-features) v1は逐次処理の12時間費用gateで停止した。ユーザーの選択を受け、DoG値を保った2 workerのframe並列化を行ったv2は、予測9.01時間でgateを通過し、19,900 frame・199動画・19,701 windowの特徴を6.23時間で生成した。v1で保存した20 frameの内容SHAはv2と全件一致し、両胚各1 frameの逐次再計算とも一致した。特徴manifestの内部SHAは`8bae3496adfce605c07dfef2dfe7837f56429b093f544a4a2a32ff4f851ee202`。

[Kaggle T4学習kernel](https://www.kaggle.com/code/kentookumura/exp039-dog-features-train) v1は`COMPLETE`。64 windowの事前benchmarkによる4モデル全体の校正後予測は18,318秒で12時間以内だった。Notebook実測は8,817.33秒、うち学習・評価区間は7,841.21秒。DoGありと同じ66入力構造のゼロ対照を各2foldで学習し、4 checkpoint、fold別正規化統計、群別・動画別診断を保存した。manifest、正規化統計、GT window filter auditと4 checkpointの実ファイルSHAは記録値と一致した。model manifest SHAは`90e919fa33b59cc4bafa1e98e4cdee4ad785cfe41721b67ff6854a211be3b959`。生成物はKaggle出力とローカルの`artifacts/train_v1/`に保存した。

| 外側評価胚 | 既知接続の回収（ゼロ対照 → DoG） | mask内正解pair（ゼロ対照 → DoG） | 分裂母の回収 | 判定可能な誤接続 | 母が異なる娘の組 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6bba | 100,230 → 100,234 / 103,393 | 42,343,742 → 42,343,734 / 42,351,260 | 48 → 48 / 108 | 21 → 21 / 1,056,829負例pair | 9 → 9 / 128,633,665判定可能な組 |
| 44b6 | 17,959 → 17,961 / 18,949 | 13,719,477 → 13,719,479 / 13,721,808 | 6 → 6 / 22 | 14 → 16 / 78,785負例pair | 6 → 8 / 15,644,861判定可能な組 |

両胚の評価はそれぞれ12,392 window・128動画、6,315 window・71動画。両条件で支持件数、候補、教師、maskが一致した。mask内正解率は6bbaでゼロ対照0.9998224846、DoGあり0.9998222957、44b6で0.9998301244、0.9998302702。分裂母回収率はそれぞれ48/108、6/22で同値だった。DoG応答2成分、強度、画像境界、動画別の内訳は保存済みの`training_summary.json`と`metrics.json`にある。

## 解釈

DoGありは既知接続を6bbaで4本、44b6で2本多く回収した。しかし6bbaではmask内正解pairが8件減り、44b6では注釈に反する接続が2本、異なる母を持つ娘の組が2組増えた。事前に定めた両胚すべての進行条件は未達。条件に従って全graph推論を行っておらず、公式combined score、adjusted edge Jaccard、division Jaccard、CV、Public LBは未測定。Kaggle submissionも行っていない。

今回の結果が示すのは、固定候補、元解像度の2組のDoG、66入力の点特徴、同構造のゼロ対照、公開tracker初期値、3 epoch、外側2胚という条件では、接続回収の小さな増加だけでは事前の進行条件を満たせなかったこと。公開画像encoderと初期trackerの学習来歴にtrain 199動画が含まれるため、胚分割の結果を独立した交差検証とは呼ばない。部分注釈下のmask内正解率も真の誤接続率ではない。

## ユーザー判断

- 判断: 不採用として実験を終了。
- 確認日時 / 依頼メッセージ: 2026-09-23「不採用として実験を終了してください。commitとpushしてください」。
- 理由: 6bbaではmask内正解pairが8件減り、44b6では注釈から確定できる誤接続が2件、母が異なる娘の組が2組増えたため、事前の進行条件を満たさなかった。

## 次

全graph推論と提出は行わず、この実験を不採用で終了する。上位仮説にはHOG特徴の未検証候補が残るため、上位仮説自体は継続する。
