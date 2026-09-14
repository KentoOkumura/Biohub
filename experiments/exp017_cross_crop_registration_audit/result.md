# exp017_cross_crop_registration_audit 結果

## 仮説

同一胚のcropが共通視野を持つなら、複数時刻の画像を同じ平行移動とD4変換で合わせた正規化相互相関が負例対照より高く、3D画像と共通するGEFF注釈も同じ変換で整合する。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験status、kernel情報、Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- Kaggle CPU kernel: `metrics.json.evidence.kaggle`。version 2、GPU無効、internet無効、計測した監査処理は1,303.05秒。
- 全体: 199 crop、胚 `44b6` 71件、`6bba` 128件。同一胚内10,613対を5時刻の2D最大値投影でscreeningし、150対を8種類のD4変換で再評価した。異なる胚100対をnull対照にした。
- 連続する同一frame: `44b6` は0件。`6bba` は114 cropに834 runあり、長さ2が772、長さ3が11、長さ4が51だった。非空の反復scheduleが完全一致するcrop対は24対だった。同じ画像内容の署名を異なるcropが共有した例は0件だった。
- 2D登録: null対照の相関99%点は0.3851、固定下限との組合せで有効閾値を0.45とした。複数時刻の相関、異時刻対照との差、shiftの安定性をすべて満たす対は0件だった。
- 反復schedule完全一致24対: 2D相関のmedianは0.2559、最大は0.4167で、全対が0.45未満だった。shift MADのmedianは縮小画像で7.45 pixel、最小でも2.24 pixelで、固定上限2 pixelを満たさなかった。同時刻相関と17 frameずらした相関の差のmedianは-0.0079だった。
- 3D登録: score上位30対と反復schedule完全一致24対の重複を除く48対を確認した。3D相関の最大は0.2266で固定下限0.35を下回り、2Dと3Dを通過した対は0件だった。反復schedule完全一致24対の3D相関medianは0.0420だった。
- GEFF: 48対中5対に3 µm以内の空間一致が1～3 nodeあり、1対では対応させたedgeが2本あった。しかし該当対を含む全48対が画像登録を失敗している。多くのnodeから任意の変換を選んだ後の少数一致なので、領域をまたぐ同一軌跡の証拠には数えない。
- 生成物: `artifacts/kaggle_v2/artifacts/audit_v1/` のCSV、JSON、PNG。正のmanifest SHAは `metrics.json.evidence.artifacts.artifact_manifest_sha`。

## 解釈

公開trainデータでは、「同じframe番号、D4回転・反転、平行移動、画像面積20%以上の共通領域」という条件で、crop間の共通視野を復元できなかった。停止frameの反復scheduleは `6bba` 内の時刻進行が共通するcrop群を示す手掛かりにはなるが、scheduleが完全一致する24対にも画素の重なりは確認できなかった。

したがって、現時点のデータだけを根拠にcropの軌跡を一つへ結合して学習することは支持しない。誤ったcrop間接続を教師へ加える危険が大きい。

この結果は、cropが空間的に無関係であることの証明ではない。重複しない隣接領域、任意の時間offset、D4以外の回転、非剛体変形、20%未満の小さい重複は今回の監査範囲外である。特に画素が重ならない隣接cropは、画像だけから相対配置を決められないため、主催者のcrop origin、absolute time、transform、永続IDなどのmetadataが必要になる。

## ユーザー判断

- 判断: 完了。crop間の軌跡を直接接続せず、sample単位のtracker学習を維持する。
- 確認日時 / 依頼メッセージ: 2026-09-14、ユーザー回答「それでいいです。git commitとpushしてください。」
- 理由: 199 cropのCPU監査で、停止frame scheduleが完全一致する24対を含め、複数時刻の2D登録と3D確認を通過するcrop対が0件だった。公開metadataだけでcrop間接続を教師へ加える根拠がない。

## 次

現行のsample単位のtracker学習を維持する。crop統合を再検討する条件は、主催者から大域座標・絶対時刻・crop変換が提供されるか、時間offsetや小重複を含む追加監査で3D画像と注釈の両方に明確な対応が見つかることである。
