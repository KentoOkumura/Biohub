---
title: "Biohub Cell Tracking: 0.946 LB Notebook 解説"
date: "2026-09-13"
types:
- survey
hypotheses:
- HYP-20260910-12
experiments:
- exp011
topics:
- public-notebooks
- baseline
- architecture
status: final
summary: 採用した公開0.946 Notebookについて、入力、2つのTemporalUNet3D、D4 TTA、候補点検出、Node Transformer、ILP、軌跡修復、出力検証をコードに沿って解説する。
---

# Biohub Cell Tracking: 0.946 LB Notebook 解説

作成日: 2026-09-13

## 結論

採用した [Biohub Cell Tracking: 0.946 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-946-lb) は、連続する2枚の3D画像から細胞候補点を検出し、候補点間の時間方向の接続を予測し、整数線形計画法（integer linear programming、ILP）と複数段の軌跡修復を経て `submission.csv` を作る推論Notebookである。Notebook内で学習は行わない。

構成の中心は次の4点である。

1. seedの異なる2つの `TemporalUNet3D + SimpleNodeTransformer` checkpointを使う。
2. xy平面の8通りのD4変換を、検出logitだけでなくprimary modelのassociation用UNet特徴にも適用する。
3. primary modelの時間順・逆順予測を確率空間で統合し、primaryが迷う接続だけsecondary modelで補助する。
4. ILPの後にも、移動の整合性、欠落frame、細胞分裂、短い軌跡を対象とする規則とDeepCenterの出力を使ってgraphを修正する。

Notebook名の0.946は、この全体を通したPublic LBの作者報告であり、detector単体、`TemporalUNet3D`単体、0.946で追加されたtest-time augmentation（TTA）単体のスコアではない。また、exp011ではこのスコアを再実行して独立確認していない。

## 対象と証拠範囲

- 対応する上位仮説: `HYP-20260910-12`
- 対象実験: [`exp011_public_detector_selection`](../../experiments/exp011_public_detector_selection/)
- Notebookの識別: kernel id `133199516`、取得したcontent SHA `ae8e01a262211045161984e469e8be23e3386bab9140fe12df503dc6a1e010e6`
- 主な証拠: 2026-09-12にKaggleから取得したNotebook source、Notebookが展開する推論source、3つのcheckpointとartifact manifest
- 機械可読な選定内容: [`public_detector_selection.json`](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)
- この文書からは判断できないこと: 0.946の独立再現性、detector単体精度、各後処理の単独寄与、hidden test全件の実行時間、将来再学習するtrackerの精度

Notebook本体のcode licenseは、取得したmetadataとsourceから確認できなかった。この文書は処理内容を要約し、Notebook独自コードを転載しない。

## 全体の処理

```mermaid
flowchart TD
    A["test/*.zarr<br/>3D time-series"] --> B["連続2 frameを読み込み<br/>z/y/x stride = 1/4/4"]
    B --> C["movie共通quantileで正規化<br/>0.001 / 0.999"]
    C --> P["Primary<br/>TemporalUNet3D"]
    C --> S["Secondary<br/>TemporalUNet3D"]
    P --> PT["8-view D4 TTA<br/>検出logit + association特徴"]
    S --> ST["8-view D4 TTA<br/>検出logit"]
    S --> SF["元画像viewの<br/>32-channel特徴"]
    PT --> D["2 modelの検出logitを校正・統合<br/>候補点保持率guard"]
    ST --> D
    D --> N["3D max-pool局所最大<br/>細胞候補点"]
    N --> PA["Primary SimpleNodeTransformer<br/>時間順 + 逆順"]
    PT --> PA
    N --> SA["Secondary SimpleNodeTransformer<br/>primary低margin時のみ補助"]
    SF --> SA
    PA --> E["edge確率を統合<br/>候補edgeを抽出"]
    SA --> E
    E --> I["ILPでlineage graphを選択"]
    I --> R["移動・gap・division修復<br/>DeepCenterによる追加候補確認"]
    R --> O["短い軌跡のfilter・座標平滑化"]
    O --> G["GEFF graph"]
    G --> CSV["submission.csv<br/>run_stats.csv"]
```

この図で重要なのは、細胞の位置を出す検出と、前後frameの同じ細胞を結ぶassociationが別の段階であることだ。後半のgraph修復は、その両方の誤りを一部補うが、元の画像モデルを再学習しているわけではない。

## 1. 実行環境とartifactの固定

NotebookはKaggleのinputからsupport artifactを探し、次のものを `/kaggle/working/tracking_repo` に展開する。

- 推論repositoryとオフライン依存package
- primary `TemporalUNet3D + SimpleNodeTransformer` checkpoint
- seedの異なるsecondary checkpoint
- 修復候補の確認に使う `DeepCenterUNet3D` checkpoint

展開後、推論sourceのPythonファイル一覧と各SHA、source全体のmanifest SHA、3 checkpointのSHAを照合してから処理を続ける。推論scriptはその後に文字列置換でpatchされるため、想定したsourceから少しでもずれるとpatch位置の検査で停止する。これは変更に弱い一方、異なるsourceへ誤ってpatchを当てたまま推論することを防いでいる。

公開Notebookは既定で作者のmirror datasetを参照する。exp011で採用した再利用契約では、同じcheckpoint SHAを持ち、licenseを確認できたPilkwangの元datasetを正とした。

| 用途 | model | checkpoint SHAの先頭12文字 |
| --- | --- | --- |
| primary検出・特徴・association | `TemporalUNet3D + SimpleNodeTransformer` | `12f6881ee362` |
| secondary検出・association | `TemporalUNet3D + SimpleNodeTransformer` | `9bac2fa0dadc` |
| gap・division追加候補の確認 | `DeepCenterUNet3D` | `8040999a92f6` |

完全なSHA、dataset version、artifact manifest SHAは選定manifestを正とする。

## 2. 入力と正規化

1つのmovieについて、window size 2、stride 1で連続frameを処理する。z方向は間引かず、yとxは4 voxelごとに読み込む。入力shapeは概念的には `batch × 2 × 1 × z × y/4 × x/4` である。

強度は各windowで計算し直さず、Zarr metadataの `image_statistics.quantiles` に記録されたmovie共通の0.001 quantileと0.999 quantileを使う。入力値を $x$ 、2つのquantileを $q_{0.001}$ 、 $q_{0.999}$ とすると、画像モデルへ渡す値は次のとおりである。

```math
x_{\mathrm{norm}}
=
\max\left(
0,
\frac{x-q_{0.001}}{q_{0.999}-q_{0.001}+10^{-6}}
\right)
```

windowごとのmin-max正規化ではないので、隣接window間で強度scaleが変わりにくい。

## 3. TemporalUNet3Dと細胞候補点

primaryとsecondaryは同じ構造を持つ独立した2組のmodelで、parameterは共有しない。各組は `TemporalUNet3D`、1-channelのdetection head、`SimpleNodeTransformer` からなる。

```mermaid
flowchart LR
    X["2 frames<br/>B × 2 × 1 × Z × Y/4 × X/4"] --> U["TemporalUNet3D<br/>32 → 64 → 128 → 64 → 32"]
    U --> F["frame別feature map<br/>B × 2 × 32 × Z × Y/4 × X/4"]
    F --> H["1 × 1 × 1 detection head"]
    H --> L["center logit<br/>B × 2 × 1 × Z × Y/4 × X/4"]
    L --> C["局所最大から候補点"]
    F --> I["候補座標で32-channel特徴を取得"]
    C --> I
    I --> T["SimpleNodeTransformer<br/>frame間の全候補pairを採点"]
    T --> E["edge logits<br/>B × N_t × N_{t+1}"]
```

`TemporalUNet3D` のchannel幅は `[32, 64, 128]` で、入力済みのdownsample grid上でencoderとdecoderを構成する。構造は次のとおりである。

| 段階 | channel | 空間処理 | 時間方向の処理 |
| --- | ---: | --- | --- |
| encoder stage 1 | 32 | `Conv3d → BatchNorm3d → ReLU` を2回 | 省略 |
| encoder stage 2 | 64 | 2倍max pooling後、同じconv block | 同じ空間位置にある2 frame間の4-head self-attention |
| encoder stage 3 | 128 | さらに2倍max pooling後、同じconv block | 同じ空間位置にある2 frame間の4-head self-attention |
| decoder stage 1 | 64 | trilinear upsample、64-channel skipと連結、conv block | なし |
| decoder stage 2 | 32 | trilinear upsample、32-channel skipと連結、conv block | なし |
| feature head | 32 | 1×1×1 convolution | frame別の32-channel特徴を出力 |

時間attentionでは、各空間位置を独立に扱い、その位置にある長さ2のframe列だけへself-attentionを適用する。これは後述する候補点集合間の `SimpleNodeTransformer` とは別のattentionである。前者は同じvoxel位置の画像特徴を2 frame間で混ぜ、後者は異なる座標を含む全候補点間の接続を採点する。

class定義から数えた1 branchのparameter数は、`TemporalUNet3D` が1,496,320、detection headが33、`SimpleNodeTransformer` が580,353で、合計2,076,706である。primaryとsecondaryはこの構造を1組ずつ持つ。DeepCenterは別modelなので、この数には含めない。

32-channel出力には次の2つの使い道がある。

- 1×1×1 convolutionのdetection headで、各voxelが細胞中心であることを表す1-channel logitを作る。
- 候補点座標で32-channel feature mapを整数indexingし、association特徴として使う。

detection logitへsigmoidを適用した確率が0.965を超え、かつ約5 µmの3D max-pooling範囲内で局所最大となるvoxelを細胞候補点にする。物理距離5 µmは、各軸のvoxel sizeに合わせて奇数のpooling kernelへ変換される。

## 4. 8-view D4 TTA

TTAは、入力を変換してそれぞれ推論し、出力を元の向きへ戻して平均する処理である。このNotebookはz軸を変換せず、xy平面の正方形が持つ8通りのD4対称変換を使う。

| view | 処理 |
| ---: | --- |
| 1 | 元画像 |
| 2–4 | x反転、y反転、xy両方の反転 |
| 5–6 | 90度、270度回転 |
| 7–8 | 転置、反転を伴う転置 |

変換を $T_i$ 、detection logitを $L$ とすると、平均logitは次のように書ける。

```math
\bar{L}(x)
=
\frac{1}{8}
\sum_{i=1}^{8}
T_i^{-1}\!\left(L\!\left(T_i(x)\right)\right)
```

0.946で追加された中心的な差分は、primary modelの32-channel feature map $F$ にも同じ処理を行う点である。

```math
\bar{F}(x)
=
\frac{1}{8}
\sum_{i=1}^{8}
T_i^{-1}\!\left(F\!\left(T_i(x)\right)\right)
```

```mermaid
flowchart LR
    X["2-frame input"] --> V["8 D4 views"]
    V --> L["detection logits"]
    V --> F["primary 32-channel features"]
    L --> LI["inverse transform"]
    F --> FI["inverse transform"]
    LI --> LA["8-view average"]
    FI --> FA["8-view average"]
    LA --> C["candidate coordinates"]
    FA --> A["primary association features"]
    C --> A
```

secondary modelもdetection logitには8-view平均を使うが、associationへ渡す32-channel特徴は元画像のviewだけから取得する。したがって「0.946では2 modelの全特徴をTTA平均する」という理解は正しくない。

## 5. 2 modelの検出結果の統合

secondary detection logitは、平均と標準偏差をprimaryへ合わせてから統合される。標準偏差の比は0.5から2.0へclipする。最終的なdetection logitはprimary 0.20、secondary 0.80の重みである。

ただし、統合後の候補数がprimary単独の候補数の90%未満になるframeでは、候補点を失いすぎたと判断してprimary単独のdetection logitへ戻す。この判定はframeごとに行い、診断用JSON Linesへ候補数と選択結果を記録する。

このguardは候補点の数だけを見ており、候補点の正しさや座標一致を確認するものではない。

## 6. SimpleNodeTransformerによるassociation

### 6.1 入力するnode特徴

`SimpleNodeTransformer` は画像全voxelをtokenにするのではなく、検出済みの細胞候補点だけをtokenとして扱う。frame $t$ の候補数を $N_t$ 、frame $t+1$ の候補数を $N_{t+1}$ とする。各候補点には次の情報を連結する。

- 候補点座標で取り出した32-channel UNet特徴
- window内の相対時刻とz/y/xをsin/cosで符号化した32-dimensional位置embedding

位置embeddingは、時刻、z、y、xの各軸を対応するwindowまたは画像shapeで正規化する。軸 $a$ の正規化座標を $u_a$ とすると、周波数 $1,2,4,8$ のsinとcosを並べた8 dimensionsを作る。

```math
\mathrm{PE}_a(u_a)
=
\left[
\sin(2^k\pi u_a),
\cos(2^k\pi u_a)
\right]_{k=0}^{3}
\in \mathbb{R}^{8}
```

4軸を連結すると32 dimensionsとなる。時刻にはmovie全体の絶対frame番号ではなく、2-frame window内の0と1を使う。node $i$ の入力は次の64 dimensionsである。

```math
v_i
=
\left[
f_i^{\mathrm{UNet}}\in\mathbb{R}^{32};
\mathrm{PE}(t_i,z_i,y_i,x_i)\in\mathbb{R}^{32}
\right]
\in\mathbb{R}^{64}
```

候補座標でのUNet特徴取得は整数indexingであり、周囲のvoxelをtrilinear補間する処理ではない。したがってassociationは候補座標そのものを微分可能に補正せず、検出済み座標の画像特徴と位置embeddingから接続を予測する。

### 6.2 Transformer本体の構造

```mermaid
flowchart LR
    S0["source nodes at t<br/>B × N_t × 64"] --> SP["Linear 64 → 128<br/>LayerNorm"]
    T0["target nodes at t+1<br/>B × N_{t+1} × 64"] --> TP["同じLinear 64 → 128<br/>LayerNorm"]
    SP --> B1["Cross-attention block × 4<br/>source ← target<br/>target ← updated source"]
    TP --> B1
    B1 --> SN["source LayerNorm<br/>B × N_t × 128"]
    B1 --> TN["target LayerNorm<br/>B × N_{t+1} × 128"]
    SN --> P["全source-target pairを作る<br/>128 + 128 + 3 = 259"]
    TN --> P
    R["相対 z/y/x<br/>(source − target) / 100"] --> P
    P --> M["pair MLP<br/>259 → 128 → 64 → 1"]
    M --> O["edge logits<br/>B × N_t × N_{t+1}"]
```

| 項目 | 実装値 |
| --- | ---: |
| node入力 | 64 dimensions |
| hidden dimension | 128 |
| cross-attention heads | 4 |
| 1 headあたりのdimension | 32 |
| cross-attention blocks | 4 |
| block内MLP | 128 → 256 → 128 |
| activation | GELU |
| dropout | 0.3 |
| pair scorer入力 | 259 dimensions |
| pair scorer | 259 → 128 → 64 → 1 |
| source方向のpair chunk | 32 nodes |

各attention headは、query候補と反対frameの全key候補との内積から重み行列を作る。head $h$ の射影行列を $W_Q^{(h)}$ 、 $W_K^{(h)}$ 、 $W_V^{(h)}$ とすると、sourceからtargetを見るheadは次の計算に対応する。

```math
\begin{aligned}
A_h
&=
\mathrm{softmax}\!\left(
\frac{(QW_Q^{(h)})(KW_K^{(h)})^\top}{\sqrt{32}}
\right),\\
\mathrm{head}_h(Q,K)
&=
A_h(KW_V^{(h)})
\end{aligned}
```

$A_h$ のshapeは $N_t\times N_{t+1}$ である。4 headsの出力を連結して128 dimensionsへ戻す。このattentionはedgeを直接確定せず、各候補の表現へ反対frameの候補集合全体の文脈を加える。その後のpair MLPが個々のedge logitを出す。

各blockはpre-LayerNorm型で、cross-attentionとMLPの両方にresidual connectionを持つ。source表現を $q$ 、target表現を $k$ とすると、source側の更新は概念的に次の形である。MHAでは4 headsに分けるため、各headのquery、key、valueは32 dimensionsである。

```math
\begin{aligned}
\hat{q}
&=
q+\mathrm{MHA}\!\left(
\mathrm{LN}_1(q),
\mathrm{LN}_1(k),
\mathrm{LN}_1(k)
\right),\\
q'
&=
\hat{q}+\mathrm{MLP}\!\left(\mathrm{LN}_2(\hat{q})\right)
\end{aligned}
```

同じblock内で、次にtarget側をquery、更新済みsource側 $q'$ をkey/valueとしてtarget表現を更新する。

```math
\begin{aligned}
\hat{k}
&=
k+\mathrm{MHA}\!\left(
\mathrm{LN}_1(k),
\mathrm{LN}_1(q'),
\mathrm{LN}_1(q')
\right),\\
k'
&=
\hat{k}+\mathrm{MLP}\!\left(\mathrm{LN}_2(\hat{k})\right)
\end{aligned}
```

この2方向は別々のTransformerを使うのではなく、同じblockのattention・MLP parameterを共有する。また、sourceとtargetを同時更新するのではなく、target側はそのblockで更新済みのsource表現を参照する。この処理をparameterの異なる4 blocksで繰り返し、最後に両候補集合へLayerNormを適用する。

padding用maskを受け取る実装だが、この公開Notebookの動画別推論では候補をpaddingせず、maskはすべて真である。また `model.eval()` で実行するため、構造上のdropout 0.3は推論時には無効になる。

ここでいう「2方向のcross-attention」は、後述するprimary modelの「時間順・逆順推論」とは異なる。block内部では1回のforward中に両候補集合の表現を更新する。時間順・逆順推論は、modelへのsourceとtargetの入力自体を入れ替えてedge logitをもう一度計算し、その2回の出力確率を統合する処理である。

### 6.3 全候補pairの採点

4 blocks後のsource表現 $q_i$ とtarget表現 $k_j$ に、original-resolution voxel座標の差を100で割った3 dimensionsを加える。これは物理単位のµmではない。

```math
r_{ij}
=
\frac{c_i-c_j}{100}
\in\mathbb{R}^{3},
\qquad
g_{ij}
=
[q_i;k_j;r_{ij}]
\in\mathbb{R}^{259}
```

各 $g_{ij}$ を共有のpair MLPへ通し、すべての組に1つずつedge logit $s_{ij}$ を出す。pair MLPは `Linear(259,128) → GELU → Dropout(0.3) → Linear(128,64) → GELU → Linear(64,1)` である。Transformer内には距離による候補削減や局所近傍maskがなく、計算量は基本的に $N_tN_{t+1}$ に比例する。

source候補を32件ずつ処理する `pair_chunk_size=32` は、一度に作る `B × chunk × N_{t+1} × 259` tensorを小さくしてpeak memoryを抑えるための実装である。全pairを最終的に採点する点は変わらず、計算量を近傍探索のように減らすものではない。

最終出力shapeは `B × N_t × N_{t+1}` である。forward・reverse・secondaryの統合後、source軸へsoftmaxを適用するため、各target候補について「どのsource候補をparentとするか」の相対確率になる。明示的な「接続なし」tokenはTransformer出力に含まれず、0.48の候補edge閾値と後段のILPにあるappearance・disappearance costが未接続を扱う。

### 6.4 primaryの時間順・逆順統合

primary modelは $t \rightarrow t+1$ と $t+1 \rightarrow t$ の両方向でedgeを予測する。逆方向logitの平均とscaleを時間順logitへ合わせた後、確率へ変換する。実装は単純な50対50の調和平均ではなく、逆方向の寄与 $w=0.15$ を使う重み付き調和平均である。

```math
\tilde{p}_{ij}
=
\left(
\frac{1-w}{p^{\mathrm{forward}}_{ij}}
+
\frac{w}{p^{\mathrm{reverse}}_{ij}}
\right)^{-1},
\qquad
w=0.15
```

その後、候補間で再正規化し、log確率をprimaryの元logit scaleへ合わせる。どちらか一方向だけが高いedgeを抑えつつ、時間順予測を主に残すための処理である。

### 6.5 secondaryのlow-margin consensus

secondary modelのedge logitも、平均とscaleをprimaryへ合わせる。ただし常に0.15を混ぜるのではない。各targetに対してprimaryの上位2 parent候補の確率差をmarginとし、次の条件でだけsecondaryを使う。

- primary marginが0.35未満である。
- primaryとsecondaryが同じparentを1位にしている。

marginが0.35へ近づくほどsecondaryの重みは0へ下がり、primaryが最も迷っているときに最大0.15となる。2 modelが不一致ならsecondaryの重みは0である。

統合後はsource軸へsoftmaxを適用し、確率0.48を超えるedgeをILP前の候補に残す。

## 7. ILPによるgraph選択

候補点をnode、接続候補をedgeとするgraphを作り、`tracksdata` のILP solverへ渡す。採用設定は次のとおりである。

| 項目 | 値 |
| --- | ---: |
| edge probability weight | -1.0 |
| appearance weight | 0.0 |
| disappearance weight | 2.0 |
| division weight | 1.2 |

ILPは各edgeを独立に閾値処理するだけではなく、trackの出現、消失、分裂を含むgraph全体の整合性を考慮して接続を選ぶ。出力はmovieごとのGEFF graphとして一度保存される。

GPUが2台見える場合はmovie一覧を交互に2 shardへ分けて別processで推論し、全movieが重複なく揃ったことを確認してからGEFFを統合する。1 GPUでは単一processで実行する。

## 8. ILP後の軌跡修復

NotebookはILP出力をそのまま提出しない。次の順序でnodeとedgeを修正する。

```mermaid
flowchart TD
    A["ILP後のGEFF graph"] --> B["次frame以外のedgeと<br/>14 µm超edgeを除外"]
    B --> C["移動予測 + learned edge確率で<br/>Hungarian assignment"]
    C --> D["1 nodeへの複数parentを解消"]
    D --> E["1-frame gap close<br/>既存node再利用または1 node追加"]
    E --> F["2-frame gap recovery<br/>2 node追加"]
    F --> G["safe division追加<br/>幾何条件 + DeepCenter"]
    G --> H["孤立nodeを除外"]
    H --> I["短いcomponentを除外<br/>高信頼なら一部救済"]
    I --> J["linear trackの座標を<br/>局所直線fitで平滑化"]
    J --> K["最終node / edge"]
```

### 移動に基づく再接続

各frame間で、直前の移動ベクトルから次位置を予測し、予測位置との差、元の移動距離、学習済みedge確率をcostにしたHungarian assignmentを行う。6 µmのtight passと10 µmのrelaxed passを順に使う。候補が得られた場合、コード上は元のedge集合をこのassignment結果で置き換える。

### 1-frame gap close

frame $t$ のtrack終端とframe $t+2$ のtrack始端を対応付け、間のframe $t+1$ にある孤立候補点を再利用する。近い孤立候補点がなければ中点にnodeを追加し、元画像の局所強度重心で位置を補正する。

基準は1 stepあたり5.0 µmで、周囲の細胞間隔に応じて小さく補正される。合計移動が8.5 µm以上で新しいnodeを追加する場合は、DeepCenter heatmapの最大値が0.25以上であることを要求する。

設定上の `GAP_CLOSE_MAX_GAP` は2だが、この関数は実装内で有効値を1へ制限する。2 frame分が欠ける場合は、次の別処理が担当する。

### 2-frame gap recovery

frame $t$ とframe $t+3$ を接続し、間に2 nodeを追加する。全移動10.2 µm以下、1 step平均4.4 µm以下に加え、前後の既存edgeから得られる移動方向との整合性を要求する。追加数にはframe単位と全体のcapがある。

### safe divisionの追加

すでに1 childを持つparentに対して、次frameの未接続候補を2番目のchildとして追加する。主な条件は次のとおりである。

- parentから追加childまで9.0 µm以下
- 既存childと追加childの距離が14.0 µm以下
- 既存childまでの距離が10.0 µm以下
- 追加childが既存childから見た相互最近傍候補である
- その次のframeで2系統が2.25 µm以上離れる方向へ進む
- 2 childのparent距離の非対称度が0.6以下
- 追加child位置のDeepCenter scoreが0.25以上

frameごとの追加数と全体の追加数にもcapを設ける。DeepCenterは全細胞を検出し直すためには使わず、gap closeまたはdivisionで追加しようとする一部の候補を確認するgateとしてだけ使う。

### 短い軌跡と座標平滑化

node数6未満のconnected componentは原則として除外するが、divisionを含むcomponentは残す。除外量が多いときは、node数4以上、平均edge確率0.88以上、平均edge距離3.0 µm以下のcomponentを、上限内で戻す。

最後に、各nodeから一意に接続できる前後方向へ最大2 nodeずつたどり、3点以上を得られた場合は3次元座標へ直線をfitする。元座標20%、fit座標80%で平滑化し、graph topologyはこの段階では変えない。

## 9. 出力と実行時検査

最終的な `submission.csv` は、node行とedge行を同じtableへ格納する。

| row type | 主な値 |
| --- | --- |
| `node` | `dataset`、`node_id`、`t`、`z`、`y`、`x` |
| `edge` | `dataset`、`source_id`、`target_id` |

Notebookは次を検査してから完了する。

- testの全movieが1回ずつ出力されている。
- `id` が0から連続している。
- nodeとedgeの両方が存在する。
- 座標と時刻が負でない。
- edgeは必ず次frameへ接続する。
- 各nodeの入次数は1以下、出次数は2以下である。
- 候補点保持率guardの診断が全movieを覆う。

`run_stats.csv` にはmovieごとのnode数、edge数、各修復の候補数・採用数・除外数、推論時間などを記録する。加えて、checkpoint SHA、support source SHA、submission SHA、topology、候補点保持率をJSONへ保存する。

## 10. 0.946で変わった範囲

Notebook本文が示す変更履歴では、0.941から0.946の差はprimaryのassociation特徴へ8-view D4平均を追加したことに限定される。

| 処理 | 0.941 | 0.946 |
| --- | --- | --- |
| detection logitの8-view TTA | あり | あり |
| primary association特徴 | 元画像viewのみ | 8-viewを元の向きへ戻して平均 |
| D4 view数 | 8 | 8 |
| bidirectional association weight | 0.15 | 0.15 |
| safe division parent距離 | 9.0 µm | 9.0 µm |
| DeepCenter division threshold | 0.25 | 0.25 |
| gap close基準 | 5.0 µm/step | 5.0 µm/step |

したがって、0.946を「TTA view数を増やした構成」や「後処理を全面変更した構成」と捉えるのは誤りである。既存の8 viewから得ていたUNet特徴を、primary associationまで利用するようにした変更である。

## 11. exp011での採用範囲

このNotebook自体は推論だけを行う。一方、exp011が将来の下流学習へ渡した契約は次のとおりであり、Notebook内で実行済みの学習内容ではない。

- primary checkpointの `SimpleNodeTransformer` を初期値にする。
- primary `TemporalUNet3D` とdetection headを固定する。
- secondary branch全体とDeepCenterを固定する。
- 候補点生成、特徴抽出、secondaryの補助、ILP、graph修復を最初の比較では固定する。
- 変更対象をprimary trackerの重みに限定する。

これにより、detectorや後処理の変更とtracker再学習の効果を最初の比較で混在させない。

## 注意点

- 0.946はNotebook題名、本文、実行receiptに基づく作者報告で、exp011によるPublic LBの再現値ではない。
- receiptはleaderboard feedbackを設定へ使用したと記録しているため、独立validationで選ばれた固定構成とは扱わない。
- primary checkpointのbest epoch、seed、split、独立validationの有無はartifact manifestから確認できない。
- secondary association特徴には8-view平均が適用されない。
- 動的source patchは固定したsupport sourceに強く依存する。sourceを更新する場合はpatch成功だけでなく処理意味の再監査が必要である。
- Notebook独自patchをリポジトリへコピーする前にcode licenseを確認する必要がある。
- full inference、hidden test実行時間、候補点数、特徴cache容量は未計測である。

## 関連ファイル

- 選定理由とartifact来歴: [`biohub-public-detector-selection_20260912.md`](biohub-public-detector-selection_20260912.md)
- 実験結果とユーザー判断: [`result.md`](../../experiments/exp011_public_detector_selection/result.md)
- 選定manifest: [`public_detector_selection.json`](../../experiments/exp011_public_detector_selection/assets/public_detector_selection.json)
- Notebook: [Biohub Cell Tracking: 0.946 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-946-lb)

## 次に確認すること

original Pilkwang datasetをKaggle inputへ付けた小規模runで、3 checkpoint SHA、import、候補点数、primary TTA特徴shape、secondary元view特徴shape、1 movieの実行時間、出力schemaを確認する。これは説明文書の作成では実行していない。
