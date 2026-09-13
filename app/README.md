# アプリ

実験確認用に任意で使う Streamlit アプリです。

## 実験ダッシュボード

```bash
task app
```

実験スコア表、`experiment_summary.md`、提出履歴を表示します。

## OOF 分析

```bash
task oof-app
```

`task`が利用できない環境では、同名の`make app` / `make oof-app`を使います。

`project.yml.paths.experiments_dir`で指定した実験ディレクトリの`*/artifacts/`からOOF CSVファイルを探し、簡易テーブル、数値要約、分布ビュー、欠損値要約を提供します。旧実験との読み取り互換のため、旧形式の`*/features/`と各実験ディレクトリ直下も検索しますが、新しい生成物の保存先には使用しません。

## 細胞追跡 EDA ビューアー

大きな元画像は Kaggle 上で直接読みます。
[private Kaggle Notebook](https://www.kaggle.com/code/kentookumura/biohub-tracking-eda-viewer)
を開き、**Edit → Run All** で操作してください。保存済みの静的ページは初期表示の確認用です。
GPU・Internet は不要です。

### 操作

1. サンプルを選び、時刻スライダーまたは再生ボタンで時間を移動する。
2. XY・XZ・YZ の投影または3D表示を選ぶ。Z断面の範囲を1枚にすると断面表示になる。
3. 正解の系譜を選んで軌跡をたどる。正解 ID を選ぶと、その時刻と系譜へ移動する。
4. 「前/次の未対応時刻」で、選択した系譜の未対応の正解を確認する。
5. 対応距離・検出 score の下限を変えて再計算する。CSV 保存で全時刻の対応表と設定 JSON を保存する。

緑は対応する検出がある正解、赤は未対応の正解、青は対応した検出、灰色は未対応の検出です。
紫の実線は正解の軌跡と分裂、黄色の破線は同時刻の検出・正解の対応です。
点にマウスを重ねると ID と物理座標を確認できます。投影の拡大・移動と3D回転は Plotly の操作を使います。

### 入力と解釈

- 検出: exp014 の `window_cache/<dataset>/*.npz`。32次元特徴は読み込まず、ID・座標・score・maskを読む。
- サンプル一覧には検出キャッシュと正解 GEFF の両方があるものだけを表示する。現在の exp014 は
  公開 test 4件だけを処理したため、train 199件のうち同名の4件だけを選択できる。
- 正解: 競技データの `train/<dataset>.geff`。画像は同名の `.zarr` の `0` 配列。
- `kernel_sources` に `kentookumura/exp014-exact-window-cache-inference`、
  `competition_sources` に `biohub-cell-tracking-during-development` を接続済み。
- Zarr は `pilkwang/biohub-tracking-support-pack-50ep-v1` の wheel からオフライン導入する。
- GEFF の voxel 座標を z=1.625、y=x=0.40625 µm/voxel で物理座標へ変換する。
  exp014 は保存済みの物理座標をそのまま使う。
- 同時刻の3次元距離が指定上限以内となる1対1の割当を求める。対応数を最大化し、その中で距離合計を最小化する。
  既定の上限は7 µm。これは EDA 用の対応付けで、公式 tracking score は計算しない。
- 対応計算は表示用のZ範囲・系譜選択より前に行う。画面の件数は全視野・全Zの件数。
- 正解は疎なので、未対応の検出を誤検出と断定しない。候補 ID を時間方向の軌跡 ID とみなさない。
- 正解の「系譜」は GEFF の接続成分で、分裂後の両方の娘細胞を含む。
- 同時刻を含む隣接キャッシュは重複して数えない。ID・座標が一致しない場合はエラーにする。
  score はその時刻を含む最初の window を使い、異なる window で score が変わる件数を読み込み情報へ記録する。
- キャッシュのない時刻を未検出扱いにせず、対応表に「キャッシュなし」と記録する。
- 画像は選んだ1時刻・Z範囲だけ読む。Notebook 版の画像投影キャッシュは最大6件。
- 公開 test と同名 train の比較は EDA 用であり、汎化性能の検証には使わない。

対応表には `gt_id`、`detection_id`、`distance_um`、`nearest_distance_um`、
`candidates_in_radius`、`detection_score`、分裂親の印と対応状態を保存する。
`nearest_distance_um` は1対1割当による距離と異なる場合がある。

### コードと再生成

読み込み・対応付けは [tracking_data.py](tracking_data.py)、描画は [tracking_plots.py](tracking_plots.py)、
Notebook の操作部は [tracking_notebook.py](tracking_notebook.py)、既定値は [tracking_viewer.yaml](tracking_viewer.yaml)。
Notebook はこれらのソースを埋め込んだ自己完結形式で、リポジトリの clone は必要ない。
Plotly は [Kaggle 向け renderer](https://plotly.com/python/renderers/#kaggle-and-azure) を使い、
ブラウザーが描画ライブラリを読み込む。Kaggle 計算環境の Internet 設定は OFF のままで使える。
[生成 Notebook](kaggle/tracking_viewer.ipynb) と [metadata](kaggle/kernel-metadata.json) は次で更新する。

```bash
task build-tracking-viewer
task check-tracking-viewer
task test-tracking-viewer
```

### ローカル版

```bash
task tracking-app
```

`task` がなければ、すべて同名の `make` ターゲットを使う。必要な追加依存は起動時に `uv` が解決する。
検出キャッシュと train のパスをサイドバーから指定する。既定の train パスは `project.yml.data.train_dir`。
正解・元画像がローカルにない場合は、明示したうえで検出点だけ表示する。

### 検証

合成データで、競合する1対1割当・距離上限・空フレーム・mask・重複 window・座標不一致・
物理座標変換・分裂・画像の軸とZ範囲・Notebook の操作・ローカル版の時間移動を検証する。
Kaggle 実行では `viewer_input_check.json` と `viewer_initial_matches.csv` を出力する。
画面上の操作には、Notebook の対話セッションが必要。

2026-09-13 の確認: Kaggle v4 は CPU・Internet OFF で正常終了し、実画像の初期投影と
100時刻の対応計算、検証 JSON / CSV の生成まで到達した。対象テスト8件、合成画像を使う
実 Jupyter kernel での初期表示・時刻変更、ローカル HTTP 応答を確認済み。
ブラウザー上での実表示・マウス操作は未確認。
関連する既存テストは60件成功し、「実験ディレクトリが空」というテンプレート専用テスト1件は
既存実験があるため失敗した。全体リンク検査も既存 exp001 の `requirements.md` にある
相対リンク4件で失敗したが、このアプリの文書リンクは正常。これらの既存問題は今回変更していない。

2026-09-13 の表示修正: Kaggle でアイコンフォントが読み込まれず四角に×で表示された再生操作と
CSV保存を、日本語の文字ボタンへ変更した。画面上にも「検出キャッシュあり4件 / 正解train 199件」
と選択範囲を表示する。
