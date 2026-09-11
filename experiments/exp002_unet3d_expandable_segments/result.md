# exp002_unet3d_expandable_segments 結果

## 仮説

CUDA利用前に`PYTORCH_ALLOC_CONF=expandable_segments:True`を設定すると、exp001 version 3で失敗したbatch size 16の同一smokeをOOMなく完了し、runtime gate通過後にsplit 0の3 epochsを11時間以内で完走できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: `exp001_temporal_unet3d_baseline` version 3のallocator設定なし・batch size 16 smoke。最初のbackwardでCUDA OOMになった。
- Kaggle Notebook: [exp002 train](https://www.kaggle.com/code/kentookumura/exp002-unet3d-expandable-segments-train) version 2、private、T4 x2、internet無効。version 1はallocator変更後のsmokeを通過したが、11時間gateでfull trainingを停止した。
- `metrics.json`の参照キー: `status=usable`、`public_lb=0.453`、`diagnostic_validation.smoke`、`diagnostic_validation.full_training`、`evidence.kaggle`、`evidence.artifacts`。主評価用CVとPrivate LBは未取得。
- version 2 smoke: 2 iterationsを47.241秒でOOMなく完了した。GPU 0/1のpeak allocatedは12.327/7.866 GiB、peak reservedは14.016/10.963 GiB。3 epochsの予測は36,375.657秒（10.104時間）で、43,200秒（12時間）のgateを通過した。
- full training: 180 train samples、19 validation samples、batch size 16で3 epochsを完走した。学習処理全体は28,488.802秒（約7時間54分49秒）、Notebook全体は28,821.869秒（約8時間00分22秒）だった。

| epoch | validation edge loss | validation edge accuracy | validation node recall | selection score |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0011 | 0.9992 | 0.9023 | 0.9016 |
| 1 | 0.0008 | 0.9995 | 0.8952 | 0.9016 |
| 2 | 0.0005 | 0.9997 | 0.8953 | 0.9016 |

- best checkpointはepoch 0として保存され、checkpoint、model config、model manifest、training summary、training logを回収した。SHAは`metrics.json`の`evidence.artifacts`を正とする。
- Kaggle inference Notebook version 1は、実行時に列挙した4件の公開test datasetを788.633秒で処理した。15,260 node rowsと11,320 edge rows、合計26,580行を生成し、repositoryの提出前検証は重複ID 0、missing 0、infinite 0でPASSした。
- code submission ref `56153451`は2026-09-11 08:38:45 JSTに作成され、固定監視の開始から206分後に`COMPLETE`となった。`metrics.json`のPublic LBは`0.453`、Private LBは未表示である。提出履歴と公開test出力のファイル証拠は[`SUBMISSIONS.md`](../../SUBMISSIONS.md)の`v001`を参照する。

## 解釈

`expandable_segments:True`を設定すると、exp001と同じbatch size 16のsmoke OOMを解消し、version 2では同じ学習条件の3 epochsをKaggleの12時間上限内で完走できた。version 1の11時間gate判定はその時点の契約どおりだが、version 2の実測は約7.91時間であり、2 iterationsからの保守的予測は実際のfull trainingより長かった。

診断用holdoutのselection scoreは3 epochsとも0.9016で、best checkpointは最初に同値へ到達したepoch 0である。edge accuracyは上昇した一方でnode recallはepoch 0から低下しており、epoch追加による診断指標の改善は確認できない。このholdoutはsample単位で、同じ胚由来sampleが学習側とvalidation側へ入る可能性があり、公式評価指標とも異なるため、Public LB `0.453`と直接比較して一般化性能を判断できない。

一方、hidden testでのcode submissionが完走してPublic LBを取得できたため、random initializationから3 epochs学習した3D U-Netの提出可能な比較基準は確立できた。回収した`submission.csv`とそのSHAは公開test実行時の証拠であり、Kaggleがcode submissionとして再実行したhidden testの出力ファイル自体はローカル取得していない。モデルの採用可否を精度面で判断するには、胚を分けた主評価用CVとPublic LBの整合確認が残る。

## ユーザー判断

- 判断: `usable`
- 確認日時 / 依頼メッセージ: 2026-09-11 / ユーザー「はい、いいです。最後にgit commitとpushもしてください。」
- 理由: allocator変更でOOMを解消し、3 epochs学習、hidden test code submission、Public LB `0.453`まで再現可能な初期baselineとして確立できた。精度の一般化判断は、胚を分けた主評価用CVへ引き継ぐ。

## 次

exp002を初期baselineとして固定し、胚を分けた主評価用CVとの整合を後続実験で確認する。
