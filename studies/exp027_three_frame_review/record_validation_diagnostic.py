"""Record the verified internal-threshold diagnostic without changing train evidence."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/exp027_multi_frame_tracker"
ART = EXP / "artifacts/kaggle_validation_diagnostic_v1/validation_diagnostic"


def main() -> None:
    readout_path = Path(__file__).with_name("validation_readout.json")
    readout = json.loads(readout_path.read_text())
    summary = json.loads((ART / "diagnostic_summary.json").read_text())
    assert readout["integrity"] == "PASS" and summary["reference_reproduced"]
    metrics_path = EXP / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    original_train = metrics["train_stage"]
    original_context = metrics["context_diagnostic"]
    metrics["validation_diagnostic"] = {
        **summary,
        "outer_evaluation": readout["outer_results"],
        "outer_comparisons": readout["outer_comparisons"],
    }
    metrics.setdefault("evidence", {})["validation_diagnostic"] = {
        "kernel_id": "kentookumura/exp027-multi-frame-tracker-validation-diagnostic",
        "kernel_version": 1,
        "source_kernel_version": 4,
        "summary_sha256": readout["internal_summary_sha256"],
        "selected_thresholds_sha256": readout["selected_thresholds_sha256"],
        "prediction_content_sha256": summary["prediction_content_sha256"],
        "outer_summary_sha256": readout["outer_summary_sha256"],
        "readout_sha256": hashlib.sha256(readout_path.read_bytes()).hexdigest(),
        "notebook_runtime_seconds": summary["notebook_runtime_seconds"],
        "resource": "Tesla T4",
        "internet_enabled": False,
        "integrity_verified": True,
        "source_sha256": hashlib.sha256(
            (EXP / "exp027_multi_frame_tracker_validation_diagnostic.py").read_bytes()
        ).hexdigest(),
    }
    assert (
        metrics["train_stage"] == original_train
        and metrics["context_diagnostic"] == original_context
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    internal_lines = []
    labels = {
        "two_frame_saved": "保存2時点",
        "three_frame_full": "3時点・過去あり",
        "three_frame_masked": "同じ3時点重み・過去なし",
    }
    for fold, entry in readout["internal_selected_thresholds"].items():
        for mode in ["two_frame_saved", "three_frame_full", "three_frame_masked"]:
            data = entry["modes"][mode]
            a, b = data["fixed"], data["selected"]
            internal_lines.append(
                f"| {entry['internal_embryo']} → {entry['outer_embryo']} | {labels[mode]} | {data['selected_threshold']:.6f} | {a['true_positive_count']} → {b['true_positive_count']} / {b['positive_edge_count']} | {a['false_positive_active_pair_count']} → {b['false_positive_active_pair_count']} |"
            )
    outer_lines = []
    for embryo in ["44b6", "6bba"]:
        for mode in ["two_frame_saved", "three_frame_full", "three_frame_masked"]:
            a, b = (readout["outer_results"][embryo][mode][k] for k in ["fixed", "selected"])
            outer_lines.append(
                f"| {embryo} | {labels[mode]} | {a['true_positive_count']} → {b['true_positive_count']} / {b['positive_edge_count']} | {100 * b['positive_edge_recall']:.2f}% | {a['false_positive_active_pair_count']} → {b['false_positive_active_pair_count']} |"
            )
    intervals = []
    for embryo in ["44b6", "6bba"]:
        c = readout["outer_comparisons"][embryo]["selected_full_minus_selected_two"]
        ci = c["video_bootstrap"]["recall_delta_interval"]
        intervals.append(
            f"{embryo}: {100 * c['recall_delta']:+.2f}ポイント、95%区間 {100 * ci[0]:+.2f}〜{100 * ci[1]:+.2f}ポイント"
        )
    section = (
        """## 内部validationで閾値を決めた追加診断

[Kaggle validation diagnostic version 1](https://www.kaggle.com/code/kentookumura/exp027-multi-frame-tracker-validation-diagnostic)は追加学習0回で正常終了した。学習側の内部validation全19動画・1,849窓を3条件で推論し、元の2時点・3時点指標を再現した。Notebook実行時間は360.15秒（6.00分）。4重みの前後SHA、全窓の予測ファイル／内容SHA、内部分割、実行sourceの一致を確認した。

### 実行前に固定した閾値の決め方

foldごとに、内部validationの2時点モデルで確率0.5を超えるactive pair内教師負例数を上限にした。各条件の内部教師負例確率から、この上限を超えない最小閾値を選んだ。閾値を決めるために正例recallや外側評価の結果は使っていない。2時点モデルにも同じ規則を適用した。閾値とその選択規則は保存済みの確率から再計算して一致した。

| 内部胚 → 外側胚 | 条件 | 内部で選んだ閾値 | 既知接続回収 0.5 → 選択閾値 | 教師負例予測 0.5 → 選択閾値 |
| --- | --- | ---: | ---: | ---: |
"""
        + "\n".join(internal_lines)
        + """

内部validationでは、教師負例予測数を揃えると3時点の回収数が両学習胚で2時点を上回った。44b6側の元のrecall低下は、固定0.5では確率尺度の違いも比較に含めていたことを示す。

### 内部で確定した閾値を外側へ固定適用

同じ外側128窓の保存済みlogitsを再利用した。外側の教師や予測を見て閾値を選び直していない。

| 外側胚 | 条件 | 既知接続回収 0.5 → 選択閾値 | 選択閾値のrecall | 教師負例予測 0.5 → 選択閾値 |
| --- | --- | ---: | ---: | ---: |
"""
        + "\n".join(outer_lines)
        + """

6bbaで3時点の未回収が5件減り、2時点との差は10件から5件に縮まった。教師負例予測は21→23件で、2時点の23件と同数だった。44b6では回収が1件増え、2時点より3件多く、教師負例予測は1件少なかった。したがって固定0.5の影響は確認できたが、内部で閾値を調整しても両胚での改善にはならなかった。

正解親の1位率は閾値変更では変わらず、両胚で全3条件が同じ総数だった。分裂親は44b6が0件、6bbaが1件で、選択閾値でも全条件の回収は0件。動画を単位に2,000回再標本化した3時点−2時点のrecall差（両方が内部選択閾値）は、"""
        + "、".join(intervals)
        + """。閾値や重み自体を選び直す不確実性は含まない。

### 確率分布から分かったこと

内部44b6では、既知親の平均確率が2時点0.7936→3時点0.7736に下がり、正解親1位率は96.22%→96.13%だった。内部6bbaでも平均確率は0.8526→0.8482と少し下がる一方、1位率は98.04%→98.05%だった。確率の平均だけでは閾値近傍の分布を表せない。実際に6bba側では、平均確率が低くても元の0.5で3時点の回収数が多かった。

既知親が一つある子に限ると、最大確率の平均は正答率より低い。例えば内部44b6の3時点モデルは最大確率平均0.7861、正答率0.9613だった。この注釈された子の集合では、出力確率をそのまま正答頻度として読むと低めに出る。未知の子まで含む全接続の確率校正を確認した結果ではない。

結論として、固定閾値の影響と、外側胚へ移したときの残る差を分けられた。前後情報を入力しただけで対応関係の識別が改善するとは確認できていない。過去情報の混合方法、時刻・位置の表現、教師、1epochの学習条件は原因候補として残る。この診断から、追加の閾値探索や全graph評価を自動で始めない。

### 証拠と数値精度

構造化した数値は実験の`metrics.json`の`validation_diagnostic`、検証・再集計は`studies/exp027_three_frame_review/validation_readout.json`、窓別確率と接続別CSVは実験の`artifacts/kaggle_validation_diagnostic_v1/validation_diagnostic/`を参照する。元のtrainと過去入力除去の証拠は保持した。

内部のfloat32平均2箇所にCPU環境差による5.96e-8の差があった。件数、選択閾値、分位点、内容SHAは厳密一致した。外側は保存logitsのCPU softmax再計算と保存確率の差が最大1.74e-6だったため、正例には保存済み確率を使った。教師負例の閾値までの最小距離は2.01e-4で、元の0.5における件数も再現した。bitwise同一のsoftmax計算とは主張しない。

公開画像重みの来歴、部分注釈、各外側胚64窓、1epochの制限は継続する。内部validationはcheckpoint選択にも使った集合であり、独立した校正用集合ではない。全graph・公式score・Kaggle submissionは未実施。実験の完了・採否は未判断。
"""
    )
    survey = ROOT / "docs/surveys/biohub-exp027-three-frame-analysis_20260920.md"
    text = survey.read_text()
    text = text.replace(
        "summary: 保存重みの過去入力比較をKaggleで実行。前回のrecall差は正解親1位を保った接続の確率0.5通過で説明でき、過去追加による親1位件数の純増は両胚とも0だった。",
        "summary: 内部validationで教師負例数を揃えて閾値を固定すると外側6bbaの3時点の不足は10件から5件へ縮小した。固定0.5の影響はあるが、両胚での改善にはならなかった。",
    )
    old = "次に評価する場合は、学習側の内部validationで確率の分布とrecall・教師負例予測数の関係を確認し、決めた扱いを外側評価へ適用する。"
    text = text.replace(
        old,
        "学習側の内部validationで確率分布と閾値を確認し、決めた閾値を外側評価へ適用した。結果は以下の追加診断に示す。",
    )
    text = text.split("\n## 内部validationで閾値を決めた追加診断")[0]
    survey.write_text(text + "\n" + section)
    result = EXP / "result.md"
    text = result.read_text().replace(
        "GTが3時点とも存在する有効窓", "中央ペアの両時点にGTが存在する有効窓"
    )
    marker = "## 実行履歴と判断の限界"
    block = (
        """## 内部validationで固定した閾値の診断

[Kaggle validation diagnostic version 1](https://www.kaggle.com/code/kentookumura/exp027-multi-frame-tracker-validation-diagnostic)は全1,849窓の内部validationを追加学習0回で評価し、元の指標を再現した。実行時間360.15秒。内部の2時点0.5における教師負例予測数を上限として各条件の閾値を決め、保存済み外側予測へ固定適用した。

| 外側胚 | 条件 | 既知接続回収 0.5 → 選択閾値 | 選択閾値のrecall | 教師負例予測 0.5 → 選択閾値 |
| --- | --- | ---: | ---: | ---: |
"""
        + "\n".join(outer_lines)
        + """

固定0.5の影響はあり、6bbaの2時点との差は10件から5件へ縮小した。ただし教師負例予測数を同数にできても5件の不足が残り、両胚での改善条件は満たさなかった。親1位率は変わらず、分裂親の回収も0のまま。閾値変更を全graphへ採用した結果ではない。

内部閾値の再計算、4重みの前後SHA、1,849窓の予測SHA、元の外側0.5指標を検証した。数値・証拠は[metrics](metrics.json)の`validation_diagnostic`、[検証集計](../../studies/exp027_three_frame_review/validation_readout.json)、詳細は[考察](../../docs/surveys/biohub-exp027-three-frame-analysis_20260920.md#内部validationで閾値を決めた追加診断)を参照する。微小な数値差と部分注釈・校正集合の制限も考察に記録した。

"""
    )
    if "## 内部validationで固定した閾値の診断" in text:
        start = text.index("## 内部validationで固定した閾値の診断")
        stop = text.index(marker, start)
        text = text[:start] + text[stop:]
    result.write_text(text.replace(marker, block + marker))
    notes = EXP / "SESSION_NOTES.md"
    notes.write_text(
        notes.read_text().split("\n## 2026-09-20: validation diagnostic version 1の終了・回収")[0]
    )
    with notes.open("a") as handle:
        handle.write("""
## 2026-09-20: validation diagnostic version 1の終了・回収

- Kaggle statusはCOMPLETE、元の内部指標はreference_reproduced=true。実行時間360.15秒。19動画・1,849窓、3推論条件、4保存重み、追加学習0回。
- 全1,870ファイル・82,716,591 bytesを回収した。多数の窓別NPZの取得は、このタスクの直列CLIを停止後、同じKaggle APIを12並列・timeout付きで実行した。全ファイルを取得し、1,849窓のfile/content SHAを検証した。実行Notebookのsource SHAもローカルと一致した。
- 内部だけでの閾値選択を再計算して一致。外側での3時点回収は44b6が163/192、6bbaが484/522。教師負例予測は18件・23件。2時点は160/192・489/522、19件・23件で、6bbaの不足が10件から5件へ縮小したが、両胚の改善は成立しない。
- 内部平均値2箇所のfloat32丸め差5.96e-8、外側softmax再計算の最大差1.74e-6を検出した。正例は保存確率を使用し、教師負例の最小閾値距離2.01e-4と元の0.5件数の再現を確認。閾値や件数・内容SHAを許容誤差で緩めていない。
- 終了後GPU quotaは使用15.93h・Kaggle残29.07h、週30h制約の残14.07h、refresh 2026-09-26 00:00 UTC。quota差は他sessionを含む可能性がある。
- train_stageとcontext_diagnosticを保持し、新しい数値・証拠をvalidation_diagnosticへ追加した。詳細な結論はdocs/surveysの既存考察へ統合。全graph、追加学習、公式評価、submissionは行っていない。実験採否・完了は未判断。
""")
    readme = EXP / "README.md"
    text = readme.read_text()
    text = text.replace(
        "内部validationで閾値を固定して外側へ適用する診断も終了し、その結果を考察へ追記した。", ""
    )
    text = text.replace(
        "全graph推論は保留し、",
        "内部validationで閾値を固定して外側へ適用する診断も終了し、その結果を考察へ追記した。全graph推論は保留し、",
    )
    readme.write_text(text)
    print("Recorded validation diagnostic; prior training/context metrics preserved.")


if __name__ == "__main__":
    main()
