---
title: Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計
date: '2026-09-20'
types: [survey]
hypotheses: [HYP-20260920-02]
experiments: [exp016]
topics: [architecture, tracking]
status: final
summary: 現行の入出力・lossを維持し、共有Encoderによるフレーム内Self-AttentionとCross-Attentionを切り替える設計。Model Bの逐次更新はユーザー確認済み。実装・学習は未実施。
---

# Biohub SimpleNodeTransformerへのフレーム内Self-Attention追加設計

作成日: 2026-09-20

## 結論と依頼範囲

`SimpleNodeTransformer`の入力投影と既存Cross-Attentionの間に、同一フレームのcell集合を処理する共有`nn.TransformerEncoder`を追加する案を推奨する。tとt+1には同じ重みを別々に適用し、Self-Attentionの段階では両フレームのtokenを混ぜない。各cellの出力を残し、既存のPair MLP（多層パーセプトロン）へ渡す。

今回は設計のみ。モデルsource、実験config、Notebook、lossは変更していない。実験採番・バックログ登録・学習・Kaggle実行・submissionも行っていない。この文書の`final`はコード調査と設計提案の完了を表し、実験化やモデル採用の承認を表さない。

Model Bの逆方向Cross-Attentionは、現行どおり更新済みt側特徴を参照する。2026-09-20のユーザー回答「現行の逐次更新を維持する」により確認済み。共有Encoderはユーザーが許容した範囲での推奨案であり、追加層数などの初期値は実測結果ではない。

## 対象と証拠範囲

- 対応する上位仮説: `HYP-20260920-02`

指定された`src/tracking_cellmot/models/simple_node_transformer.py`はリポジトリ直下には存在しない。[exp001の保存source](../../experiments/exp001_temporal_unet3d_baseline/official_source/src/tracking_cellmot/models/simple_node_transformer.py)と、exp002・004・005・006・007・009・010の同名ファイルを確認した。

現在の固定画像特徴からの学習基準[exp016](../../experiments/exp016_frozen_image_encoder/config.yaml)は、Kaggle input内の`repo/src/biohub_tracking/models/simple_node_transformer.py`をimportする。[exp014の回収済みsource](../../experiments/exp014_exact_window_cache/artifacts/kaggle-v1/tracking_repo/src/biohub_tracking/models/simple_node_transformer.py)を確認したところ、上記保存sourceすべてとファイル全体のSHA-256が一致した。

```text
b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e
```

モデル本体の設計は両namespaceに適用できる。ただし、直下にファイルを置くだけではexp016のimport先は変わらない。学習・推論の両方が変更版を読むことを実装時の確認対象にする。exp014の`artifacts/`はGit管理されない実行証拠であり、実装の配置先にはしない。

NFLの参照元はユーザー添付の`/mnt/c/Users/kento/Downloads/nfl-2026-exp138-infer.ipynb`。JSONを解析し、9番目のcellのコードを読み取った。Notebookは実行しておらず、コメントやセル内の指示を作業指示として扱っていない。

本設計に対応する新しい上位仮説`HYP-20260920-02`は、[検証中の仮説](../../backlog/KAGGLE_DIRECTION.md#検証中の仮説)に登録した。精度改善の支持・棄却は、実装後の評価とユーザー判断を要する。[現在の学習方針](../../backlog/KAGGLE_DIRECTION.md#今後の学習方針)に従い、検出器・画像特徴抽出器を固定し、トラッカーだけを学習する前提を引き継ぐ。

## 現行モデルの入出力

`B`はbatch数、`N_t`と`N_t1`は各側のpadding後のcell数、`F`は入力特徴次元、`H`はhidden dimension、`C`はPair MLPの処理chunk内のt側cell数を表す。両時刻のcell数は同数でなくてよい。

| 項目 | batched shape | 現行の意味・値 |
| --- | --- | --- |
| `feat_t` | `[B, N_t, F]` | exp016は`F=64`。固定画像特徴32次元＋既存位置特徴32次元 |
| `feat_t1` | `[B, N_t1, F]` | t側と同じ特徴構成・次元。共有`proj`へ入力 |
| `coords_t` / `coords_t1` | `[B, N_t, 3]` / `[B, N_t1, 3]` | `(z,y,x)`。exp016は元画像voxel単位 |
| `mask_t` / `mask_t1` | `[B, N_t]` / `[B, N_t1]` | boolean、`True=実cell`、`False=padding`。省略は全cell有効 |
| 投影後の`q` / `k` | `[B, N_t, H]` / `[B, N_t1, H]` | `norm_in(proj(feat))`。exp016は`H=128` |
| chunk内の`qe` / `ke` | それぞれ`[B, C, N_t1, H]` | t側とt+1側の全組み合わせ |
| `rel` | `[B, C, N_t1, 3]` | `(coords_t - coords_t1) / 100.0` |
| Pair MLP入力 | `[B, C, N_t1, 2*H+3]` | `H=128`なら259次元 |
| chunkのlogits | `[B, C, N_t1]` | MLP末尾の1次元を`squeeze(-1)` |
| 最終edge logits | `[B, N_t, N_t1]` | chunkをt側cell軸で連結した活性化前のscore |
| 教師`target` | `[B, N_t, N_t1]` | 正解の隣接時刻接続が1、それ以外が0。教師maskは別途計算 |

unbatched入力では特徴`[N,F]`・座標`[N,3]`にbatch軸を補い、出力は`[N_t,N_t1]`に戻す。現行forwardはmaskを自動unsqueezeしないため、maskを渡す場合は`[1,N]`が必要。追加実装では`[N]`も正規化して受け付け、既存の`[1,N]`を維持する。

constructorの`feat_dim=33`は既定値であり、実際の学習入力次元ではない。[特徴組み立て](../../experiments/exp016_frozen_image_encoder/frozen_tracker.py)の`build_window_example`と[学習Notebook source](../../experiments/exp016_frozen_image_encoder/exp016_frozen_image_encoder_train.py)は64次元を明示している。

位置特徴は既存コードの正規化した`(t,z,y,x)`のsin/cos埋め込みを維持する。exp016のモデル用座標はcacheのgrid座標に`[1,4,4]`を掛けた値。教師の候補対応に使うµm単位の座標と混同せず、relative coordinateの符号・順序・`/100.0`を変えない。

## 現行のCross-Attentionとloss

[モデルsource](../../experiments/exp001_temporal_unet3d_baseline/official_source/src/tracking_cellmot/models/simple_node_transformer.py)の`CrossAttentionBlock`は、LayerNormをAttentionの前に適用し、Attention出力の残差加算と、GELUを使うMLPの残差加算を行う。各block内で同じ重みを両方向に使う。

```python
q = norm_in(proj(feat_t))
k = norm_in(proj(feat_t1))
for block in blocks:
    q = block(q, k, kv_mask=mask_t1)
    k = block(k, q, kv_mask=mask_t)  # 同じblockで更新したqを使う
q, k = norm_out(q), norm_out(k)
```

exp016の`blocks`は4層、headsは4、中間MLP幅はhiddenの2倍、dropoutは0.3。学習時はCross-Attentionとpair chunkにactivation checkpointing（中間活性を再計算して保存メモリを減らす処理）がある。

現行には明示的なSelf-Attentionはないが、複数回のCross-Attentionにより他方のフレームを介して同一側の情報が伝わり得る。今回の差は、時刻間融合より前に同一フレーム内の直接のAttentionを追加することである。

[exp016の`batch_legacy_focal_bce`](../../experiments/exp016_frozen_image_encoder/frozen_tracker.py)は、各batchの有効cell数で`logits[b,:n_t,:n_t1]`と教師を切り出す。その後、2次元logitsへ`softmax(dim=0)`を適用し、二値交差エントロピー（BCE）にfocal weightを掛ける。`dim=0`は親候補であるt側cell軸で、各t+1側cellについて親候補を正規化する。

教師maskは「正例接続のある行、または正例接続のある列」。paddingを表すnode maskとは意味が異なる。未知の注釈の扱いも含め、教師maskとlossを今回変更しない。モデル内にsigmoidやsoftmaxを追加せず、`BCEWithLogitsLoss`への変更も行わない。

現行collateは有効cellを先頭へ詰め、末尾をpaddingする。lossの個数によるsliceはこの前提に依存する。Attentionは途中にFalseのあるmaskも扱える設計にするが、学習データの先頭詰め契約は維持する。

## NFL Notebookとの対応

添付Notebookの`TemporalTransformerBranch`は、入力投影、系列順の位置埋め込み、`TransformerEncoder`、LayerNormを順に適用する。`self_branch`と`target_branch`は別インスタンスで、別々に文脈化した結果を結合している。

実際のbranch入力は`[B*N_players,T,F]`であり、Attentionは時間軸`T`にかかる。選手間の処理は後段の`spatial_branch`にある。今回参考にするのは、各入力群を独立に文脈化してから融合する構成である。

| NFLでの処理 | 今回のCell Tracking設計 |
| --- | --- |
| `self_branch` / `target_branch`を別々に実行 | t側 / t+1側のcell集合にEncoderを別々に実行 |
| branchごとに異なる重み | 初回は同じ特徴形式のため重みを共有 |
| 時間軸`T`をAttention | 各フレームのcell軸`N_t` / `N_t1`をAttention |
| 時間順のpositional encoding | 既存の座標由来の位置特徴を利用。cell番号の埋め込みは追加しない |
| 時間方向の集約後に軌道予測 | 集約せず各cellの表現を保持し、全cell pairを採点 |

`Temporal`という語はこの設計ではNFLとの対応を表す設定名に限る。処理説明・docstringでは「フレーム内のcell間Self-Attention」と明記する。フレーム全体の平均poolingや、対応不明な同じindex同士の結合は行わない。

## Model A・Bと現行モデルの切り替え

| 比較対象 | `use_temporal_self_attention` | `use_cross_attention` | 処理 |
| --- | --- | --- | --- |
| 現行モデル | `false` | `true` | 既存投影 → Cross-Attention → Pair MLP |
| Model A | `true` | `false` | 既存投影 → フレームごとのSelf-Attention → Pair MLP |
| Model B | `true` | `true` | 既存投影 → フレームごとのSelf-Attention → Cross-Attention → Pair MLP |

両方`false`も投影＋Pair MLPとして定義可能だが、依頼された3モデルの主比較には含めない。設定誤認を防ぐため、解決後の両flagを記録する。

既存の数値設定は`model.params`にあるため、トップレベルの`architecture`を新設せず、同じ場所へ追加する。以下はModel B用の推奨例で、実験configにはまだ反映していない。

```yaml
model:
  name: SimpleNodeTransformer
  params:
    feature_dim: 64
    hidden_dim: 128
    n_heads: 4
    n_blocks: 4                  # 既存Cross-Attentionの層数。意味を変えない
    mlp_ratio: 2.0              # 既存constructor既定値をconfigに明示
    dropout: 0.3
    pair_chunk_size: 32
    use_temporal_self_attention: true
    use_cross_attention: true
    n_self_blocks: 2             # 追加Self-Attentionの層数。初回の提案値
```

hidden dimension、heads、dropout、MLP比率は既存の値をSelf-Attentionでも共有する。層数だけを`n_self_blocks`で分離し、`n_blocks`をSelf-Attentionの層数へ読み替えない。Self-Attention専用の追加入力投影は不要。

constructorは既存引数の順序を変えず、末尾へkeyword-onlyの新規引数を追加する。新規flag省略時は`false/true`、`n_self_blocks`の既定値は2とする。既存configの`mlp_ratio`省略時も2.0を維持する。`hidden_dim`が`n_heads`で割り切れること、Self-Attention有効時の`n_self_blocks >= 1`、Cross-Attention有効時の`n_blocks >= 1`を検証する。

## Encoderとforwardの設計

追加module名は`self_encoder`とし、t/t+1で同一インスタンスを使う。`nn.TransformerEncoderLayer`を`batch_first=True`、`activation="gelu"`、`norm_first=True`で構成し、中間MLP幅を`int(hidden_dim * mlp_ratio)`とする。`norm_first=True`は既存Cross-Attentionの正規化位置へ合わせる選択である。[PyTorch 2.11のAPI仕様](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoderLayer.html)で引数とshapeを確認した。

これを`nn.TransformerEncoder`で積み、追加の末尾normは置かず、既存`norm_out`をPair MLP直前に使う。denseなpadding tensorの扱いを一定にするため、初回は`enable_nested_tensor=False`とする。causal maskは渡さない。cellの並び順は時間順ではないため、各有効cellが同じフレームの全有効cellを参照する。

以下は説明用擬似コードで、実装済みコードではない。mask正規化・空入力処理・activation checkpointingは後述の契約で包む。

```python
# 1. 現行と同じ共有投影。[B, N, F]から[B, N, H]へ変換。
q = self.norm_in(self.proj(feat_t))
k = self.norm_in(self.proj(feat_t1))

# 2. 同じEncoderを別々に呼び、各フレーム内のcell同士を文脈化。
#    tとt+1のtokenは連結しない。
if self.use_temporal_self_attention:
    q = encode_valid_frame(q, mask_t, self.self_encoder)
    k = encode_valid_frame(k, mask_t1, self.self_encoder)

# 3. Model Bと現行モデルでのみ時刻間Attentionを実行。
#    blockは正規化したqをQuery、他方をKey/Valueとして扱う。
#    ユーザー確認済み：逆方向は更新済みqを使う。
if self.use_cross_attention:
    for block in self.blocks:
        q = block(q, k, kv_mask=mask_t1)
        k = block(k, q, kv_mask=mask_t)

# 4. cell数・順序を保ったまま既存の出力正規化へ渡す。
q = self.norm_out(q)
k = self.norm_out(k)

# 5. 既存のt側chunk処理とPair MLPを使う。
#    [q_i, k_j, (coords_t_i - coords_t1_j) / 100.0]は2*H+3次元。
#    出力は活性化前の[B, N_t, N_t1]、unbatchedなら[N_t, N_t1]。
logits = existing_chunked_pair_scoring(q, k, coords_t, coords_t1)
```

Model Aでは時刻間の特徴融合を各pairのMLPで初めて行う。Model Bでは各blockがt→t+1、t+1→更新済みtの2方向を処理する。同時更新への変更や、Self/Crossを層ごとに交互配置する変更は含めない。

## paddingと空入力の契約

1. 外部interfaceは`True=実cell`を維持し、PyTorchの`src_key_padding_mask`には`~mask`を渡す。PyTorchのboolean key padding maskは`True=無視する位置`である。[MultiheadAttention API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html)
2. 各Self-Attention layerへ同じmaskを渡し、paddingがKey/Valueとして実cellの結果へ影響しないようにする。Query側paddingの出力はkey padding maskだけではゼロにならないため、Encoder入力のpaddingを無害な値へ置き、出力も`masked_fill`でゼロに戻す。単なる乗算ではNaNを除去できないため使わない。
3. Cross-Attentionは現在と同じ相手側maskを使う。追加経路では必要な箇所でQuery側paddingをゼロに戻す。padding Queryの計算結果を次のKey/Valueや有効pairへ混入させない。既存モードの正常入力に対する有効logitsの演算とstate keyは維持する。
4. mask省略は全cell有効。unbatchedの`[N]`maskを`[1,N]`へ変換し、既存の`[1,N]`も受け付ける。shape不一致やbooleanでないmaskは明確なエラーにする。
5. 全paddingのbatch要素はAttentionへ渡す前に除外し、対応する表現をゼロで戻す。Cross-Attentionは両側に実cellがあるbatch要素だけ処理する。全Keyをmaskしたsoftmaxの未定義なケースを作らず、NFL Notebookのようにpadding tokenを1つ有効扱いする方法は使わない。
6. `N_t=0`または`N_t1=0`ならAttentionとpair構築を省き、正しい空shapeのlogitsを返す。現行の空chunk連結エラーもこの境界で防ぐ。有効pairがない要素はlossを0とし、空lossのbackwardで失敗しないことを確認する。通常のexp016 cacheは空候補を拒否するため、学習データの採用方針を変えるものではない。
7. padding行・列のlogitsは対応scoreとして利用しない。現行と同じloss側sliceを維持し、一律に`-inf`を詰めてから全列softmaxする処理は追加しない。戻り値をtupleにするなど、loss interfaceを変えない。

既存の正常な先頭詰め入力について互換性を検証し、従来未対応だった1次元mask・全padding・空入力は追加の境界対応として区別する。

## checkpoint互換性と初期化

`proj`、`norm_in`、`blocks`、`norm_out`、`pair_mlp`の名前・次元・登録順を維持する。Self-Attention無効時は追加の学習parameterを作らず、`self_encoder=None`とする。既存constructorと既存state dictの`strict=True`読み込みを維持し、同一環境・evalモードの正常入力で旧実装のlogitsと一致することを確認する。

Model Aでも互換性を優先して`blocks`は登録したままforwardを省く。既存checkpoint抽出処理が`blocks`の存在を検査するためである。未使用blockには勾配がなく、現行AdamWの更新対象にならない。報告時は保存される全parameter数と、forwardに使われるparameter数を区別する。

| 読み込み用途 | 方針 |
| --- | --- |
| 旧checkpoint → 現行モード | `strict=True`。旧configは既定flagで現行モードとして読む |
| 旧checkpoint → A/Bの学習初期値 | 既存moduleをロードし、`self_encoder.*`だけを新規初期化。欠損keyをその集合へ限定して検査 |
| A/B checkpoint → 同じ構成で推論・再開 | 保存構成を再生成して`strict=True`。新規Encoderを含む欠損を許さない |
| A/B checkpoint → 旧構成、構成metadata欠落 | 明示エラー。新規Encoderを黙って捨てたり、未学習Encoderを推論へ補ったりしない |

単に`strict=False`へ一括変更して不整合を無視しない。旧checkpointからA/Bを初期化しても、新規Self-Attentionが恒等写像になる保証はなく、旧モデルと同じ予測にはならない。A/Bは学習と評価を必要とする。Cross-AttentionをSelf-Attentionへコピーして初期化する別案は初回比較へ混ぜない。

新規Encoderは既存moduleの構築後に作り、新規moduleだけを初期化する。`TransformerEncoder`は層を同じ初期値から複製するため、各新規層のAttention投影・Linearをseed管理下で独立に初期化する。既存moduleまで`model.apply(...)`で再初期化しない。[TransformerEncoderの初期化に関する公式説明](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoder.html)

checkpointとmodel manifestへ、解決済み`model.params`、共有Encoderであること、Cross-Attentionの逐次更新方針、実装sourceのSHA、初期checkpointのSHAを保存する。学習と推論のモデル生成処理を揃える。新規stateに`self_encoder.*`があるのに構成metadataがない場合は、旧config用fallbackを適用しない。

## 実装時の変更箇所

| 対象 | 必要な変更 |
| --- | --- |
| `SimpleNodeTransformer` | 新規flagと共有Encoder、Self/Cross分岐、mask正規化・空入力処理、データフローのコメント |
| `CrossAttentionBlock` | Attention・MLP・state名を維持。空batchの選別やpadding Query処理は可能な限り呼び出し側で包む |
| 実装承認後の実験の`config.yaml` | `model.params`へflag・Self層数・MLP比率を追加。A/B/現行の解決済み設定を保存 |
| 学習・推論のモデル生成箇所 | 追加引数、変更版moduleのimport、旧重みの限定的な学習初期化、新規重みの厳密な復元 |
| checkpoint抽出・model manifest | 既存prefix対応を維持し、新規構成の一致を検査・記録 |
| テスト | 3構成、padding、順序、勾配、旧checkpoint、保存・復元の契約を検証 |

参考になる接続箇所は、[exp016学習source](../../experiments/exp016_frozen_image_encoder/exp016_frozen_image_encoder_train.py)の`new_initialized_tracker`、[推論source](../../experiments/exp016_frozen_image_encoder/graph_inference.py)の`_load_tracker`、[共通処理](../../experiments/exp016_frozen_image_encoder/frozen_tracker.py)の`tracker_logits`とcheckpoint抽出関数である。

実装時の第一案は、変更モデルをリポジトリ管理の`src/tracking_cellmot/models/simple_node_transformer.py`へ置き、実装承認後の実験の学習・推論から明示importすること。必要最小限のpackage入口も用意する。固定された公開画像モデルは既存`biohub_tracking`から引き続き読む。過去実験の`official_source`や取得artifactを直接書き換えて、保存済みsource SHAとの対応を失わせない。

## 比較方法と計算費用

初回は3構成で同じ固定検出候補、画像特徴cache、位置特徴、座標、教師、loss、分割、optimizer、epochs、checkpoint選択指標、復号を使う。Self-Attentionだけを学習対象に限定せず、従来のprimary tracker全体と追加Encoderを更新する。固定画像モデル・secondary trackerは現行方針どおり固定する。

exp016と同じ公開primary checkpointから共通moduleを初期化する案を推奨する。A/Bの新規Encoderは同じ初期stateを使う。現行構成も同じ条件で再学習する対照を用意し、保存済みexp016は参照値として示す。これは公開重みからの変更効果の比較であり、scratchからのarchitecture比較とは区別する。

中心となる比較はModel B対現行である。既存Cross-Attentionを保ったままSelf-Attentionを追加する差を調べられる。Model A対Bは同じSelf-Attentionの後でCross-Attentionを使う効果、Model A対現行は構成を置き換える全体効果を調べる。Bの改善が得られてもparameter数と計算量が増えるため、それだけでSelf-Attention固有の優位性を断定しない。必要なら後続でparameter数・計算量を揃えた比較を別途設計する。

既存の公式scoreと接続・分裂成分を両胚別に示し、接続正例の再現率、既知分裂の回収、loss、有効件数、学習・推論所要時間、peak GPU memoryも記録する。閾値や復号をA/Bの結果を見て同時に変えない。公開画像モデルの学習来歴を引き継ぐため、既存と同様に条件付きの比較とし、独立した交差検証（CV）とは呼ばない。

Self-AttentionのAttention部分の計算量は、各層で`N_t*N_t + N_t1*N_t1`に比例して増える。既存Cross-Attentionは各blockで2方向の`N_t*N_t1`を扱う。Pair MLPのchunkingは追加Self-Attentionのメモリを抑えるものではない。

既存のCross-Attention・Pair MLPのactivation checkpointingを維持する。追加Encoderも学習時に再計算を使えるようにし、dropoutの乱数状態を保持する。cell集合をchunkごとに独立Encodeすると参照できるcellが変わるため、Pair MLPのchunk処理をSelf-Attentionへそのまま流用しない。

Kaggleで代表的なcell数と最大級のcell数のwindowを使い、追加2層で学習・推論費用を測る。週30 GPU時間・課金なし・提出推論12時間以内の既存制約を使い、3構成すべてのフル実行が収まるとは保証しない。予算超過時に無断でcell数を削る・画像モデルを変更するなどして比較条件を変えない。

## 実装後の受け入れ条件

| 検証 | 受け入れ条件 |
| --- | --- |
| 現行モードの回帰 | 同じ旧stateが`strict=True`で読め、同じ環境のeval出力が旧実装と一致。正常なbatched/unbatched、mask有無を確認 |
| 3構成のshape | `N_t != N_t1`を含め、出力が`[B,N_t,N_t1]`または`[N_t,N_t1]` |
| padding不変性 | 実cellが同じならpadding数・padding位置の有限特徴値・座標値を変えても、有効pair logitsが数値許容差内で一致 |
| 全padding・空集合 | batch内の一部／全体がpadding、片側0件、両側0件でshapeが正しく、有効出力・loss・勾配にNaNが出ない |
| cell順序 | 特徴・座標・maskを一緒に並べ替えると、eval出力の対応する行／列だけが同じように並べ替わる。学習lossは先頭詰めへ戻して比較 |
| フレームの分離 | Self-Attention直後のt側出力がt+1側入力の変更に依存しない。逆側も同様。共有重みの別呼び出しを確認 |
| 実際のcell間参照 | Self-Attention有効時に別の有効cellへ情報・勾配が伝わり、token単位MLPの代用になっていないことを確認 |
| Cross-Attentionの順序 | Bの逆方向が更新済みt側をKey/Valueへ渡す。AではCross-Attentionを実行しない |
| lossと勾配 | 既存lossを変更せず1 step backwardでき、A/Bで`self_encoder`と`pair_mlp`に有限の勾配が流れる |
| 保存・復元 | 3構成それぞれで保存・再生成後のeval出力が一致。構成不一致・予期しない欠損keyはエラー |
| chunkと再計算 | dropoutを制御した検証でchunk有無・activation checkpointing有無の有効出力と勾配が一致 |
| 実行経路 | 学習と推論のimport先・構成・source SHAが一致し、Kaggleのオフライン実行で読める |

実装後は対象実験の`check-exp`と`test-exp`、共通moduleに対応する限定テストを実行する。最初のフル実行と公式評価はKaggleで行う。

設計作成時に行ったのはsource・設定・loss・Notebookコードの静的確認とsource SHA照合であり、上表のモデル実行テスト、メモリ測定、精度比較は未実施。Self-Attentionがtracking精度を改善するかは、これから検証する問いである。
