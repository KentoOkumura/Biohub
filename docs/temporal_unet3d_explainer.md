# TemporalUNet3D の3D画像処理と時間情報

この文書は、Biohub – Cell Tracking During Development の公式 baseline で使われている `TemporalUNet3D` について、3D画像の入力から細胞中心の検出、後段のtrackingまでを図解します。

参照した公式実装:

- [`TemporalUNet3D`](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/models/temporal_unet.py)
- [学習処理](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/train_unet_transformer.py)
- [画像とvoxel sizeの読み込み](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/io.py)

## 1. 入力と3D座標

元画像は時間を含む `(T, Z, Y, X)` の配列です。各時点に、Z枚の `Y × X` 断面画像からなる3D volumeがあります。

```mermaid
flowchart LR
    D["OME-Zarr<br/>(T, Z, Y, X)"] --> F0["時点 t<br/>3D volume (Z, Y, X)"]
    D --> F1["時点 t+1<br/>3D volume (Z, Y, X)"]
    F0 --> W["連続2 frameのwindow"]
    F1 --> W
    W --> I["モデル入力<br/>(B, T=2, C=1, Z, Y, X)"]
```

- `B`: batch
- `T`: 時間。公式学習処理の既定window sizeは2
- `C`: 画像channel。既定は輝度1 channel
- `Z`: 奥行き
- `Y`: 高さ
- `X`: 幅

`x`、`y`、`z` は3本の入力channelではありません。1個の輝度値を持つvoxelを並べる3本の空間軸です。

公式baselineの既定voxel sizeとdownsampleは次の関係です。

```mermaid
flowchart LR
    A["元の既定voxel size<br/>Z: 1.625 µm<br/>Y: 0.40625 µm<br/>X: 0.40625 µm"] --> B["空間downsample<br/>(Z, Y, X) = (1, 4, 4)"]
    B --> C["処理時のvoxel size<br/>Z・Y・Xとも約1.625 µm"]
```

Z方向は維持し、Y・X方向を4分の1にします。これにより、既定scaleでは3軸の物理的なvoxel sizeがほぼ等しくなります。

## 2. TemporalUNet3D 内部の処理

時間軸をZ軸へ結合して4次元畳み込みを行う構成ではありません。`Conv3d` は各frameの `(Z, Y, X)` を処理し、encoderの一部で時間方向のself-attentionを適用します。

```mermaid
flowchart LR
    I["入力<br/>(B, T, 1, Z, Y, X)"] --> R["frame単位へreshape<br/>(B×T, 1, Z, Y, X)"]
    R --> E1["Conv3d 3×3×3<br/>32 channels"]
    E1 --> P1["MaxPool3d(2)"]
    P1 --> E2["Conv3d 3×3×3<br/>64 channels"]
    E2 --> A2["voxelごとの<br/>temporal self-attention"]
    A2 --> P2["MaxPool3d(2)"]
    P2 --> E3["Conv3d 3×3×3<br/>128 channels"]
    E3 --> A3["voxelごとの<br/>temporal self-attention"]
    A3 --> U2["trilinear upsample<br/>+ skip connection<br/>64 channels"]
    U2 --> U1["trilinear upsample<br/>+ skip connection<br/>32 channels"]

    E2 -. "skip" .-> U2
    E1 -. "skip" .-> U1
```

既定設定では、計算量が大きいfull-resolutionの最初のencoder stageではtemporal self-attentionを省略します。downsample後の64 channelsと128 channelsのstageで、同じ空間位置に対応する複数frameの特徴を時間方向に統合します。

decoderからは、各frame・各voxelに32 channelsの特徴を持つfieldが得られます。そこから二つの用途へ分岐します。

```mermaid
flowchart LR
    U["TemporalUNet3D出力<br/>(B, T, 32, Z, Y, X)"] --> F["voxel特徴field<br/>後段のnode特徴に使用"]
    U --> H["1×1×1 Conv3d<br/>detection head"]
    H --> L["細胞中心logit<br/>(B, T, 1, Z, Y, X)"]
```

学習時は、正解の細胞中心に対応するvoxelをpositive、それ以外をnegativeとして、重み付きBinary Cross Entropyでdetection headを学習します。

## 3. 細胞中心の検出からtrackingまで

3D U-Netが直接予測するのはtracking edgeではありません。最初に3Dの細胞中心mapから候補nodeを抽出します。

```mermaid
flowchart LR
    L["細胞中心logit<br/>(Z, Y, X)"] --> N["3D local maximum suppression<br/>既定の物理範囲: 5 µm"]
    N --> C["候補node<br/>(z, y, x)"]
```

候補位置では、TemporalUNet3Dのvoxel特徴を取得します。それとは別に、明示的な `(t, z, y, x)` 座標からsinusoidal positional embeddingを作ります。

```mermaid
flowchart LR
    C["候補node (z, y, x)"] --> FI["候補位置のvoxel特徴<br/>F[z, y, x]"]
    P["明示的な座標<br/>(t, z, y, x)"] --> PE["sinusoidal<br/>positional embedding"]
    FI --> CAT["voxel特徴と位置特徴を結合"]
    PE --> CAT
    CAT --> NT["SimpleNodeTransformer<br/>t と t+1 の候補pairをscore"]
    NT --> E["edge logit<br/>同一細胞・分裂候補"]
```

したがって各座標の役割は次のように分かれます。

- 3D U-Net内: `Z`、`Y`、`X` を画像の空間軸として処理
- detection loss: 正解中心のvoxel位置として使用
- 3D局所最大値抽出: 細胞中心候補 `(z, y, x)` を生成
- node特徴: 候補位置のvoxel特徴を取得
- tracking: `(t, z, y, x)` のpositional embeddingをnode transformerへ入力

## 4. 物体検出時から時間方向を使う理由

各frameを独立に検出してからtrackingする構成も可能です。TemporalUNet3Dを使う主な理由は、後段のtrackerが、検出段階で存在しない候補nodeを原則として接続できないためです。

```mermaid
flowchart TB
    subgraph Independent["各frameを独立に3D U-Netで検出"]
        direction LR
        I0["時点 t<br/>細胞が暗い"] --> U0["tだけで検出"] --> M0["候補nodeを検出できない"]
        M0 --> T0["後段tracker<br/>t → t+1 のedgeを作れない"]
    end

    subgraph Temporal["TemporalUNet3Dで検出"]
        direction LR
        I1["時点 t と t+1<br/>2 frameを入力"] --> U1["3D Conv<br/>+ temporal attention"] --> M1["tにも候補nodeを残せる可能性"]
        M1 --> T1["後段tracker<br/>t → t+1 のedgeを作れる"]
    end
```

時間情報には、次の効果が期待されます。

- あるframeだけ暗い、またはぼけた細胞を隣接frameの特徴で補う
- 1 frameだけ現れるノイズによる輝点を抑える
- 密集した細胞の中心を時間的な変化から区別しやすくする
- 分裂付近の1個から2個への変化をvoxel特徴へ反映する
- node transformerへ渡す特徴を、隣接frameとの対応付けに使いやすくする

これは必ず独立検出より優れることを意味しません。単frameで十分なnode recallが得られる場合や、frame間の移動が大きい場合は、独立した3D U-Netの方が単純で計算量も少なくなります。効果を判断するには、検出閾値、3D local maximum suppression、node transformer、学習・validation splitを固定し、node recall、edge Jaccard、division Jaccardと実行時間を比較する必要があります。
