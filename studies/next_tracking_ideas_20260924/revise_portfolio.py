"""Reassess the saved ideas using exp043 and the current exp040 evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
OLD = ROOT / 'studies/next_tracking_ideas_20260923/idea_portfolio.json'
P43 = ROOT / 'experiments/exp043_x138_self_trained_head/metrics.json'
P40 = ROOT / 'experiments/exp040_trackastra_association/result.md'
p = json.loads(OLD.read_text())
m = json.loads(P43.read_text())
p['evidence_cutoff'] = '2026-09-24。exp043採用・完了後、exp040は旧損失で本学習済み・損失修正後の再学習は未測定。'
p['task_summary'] = 'exp043の局所座標回帰の結果を受け、既存12案の根拠・比較対象・検証順を再評価する。案の変更を目的化せず、設計確定・学習・backlog更新は行わない。'
p['assumptions'] = [
    '前日の12案を再評価した。新しい独立発想passやsubagent作業を行ったとはしない。',
    'confidence Aは障害の直接証拠、Bは間接根拠、Cは成立条件未実測。精度改善の確率ではない。',
    'exp043のPublic LB 0.950は構成全体の結果。座標head単独のLB寄与、接続・分裂の改善は未測定。',
    'exp043の5分割は動画単位で両胚が学習側にも含まれる。胚外評価ではなく、公開画像モデルにも条件付けた比較。',
    '新たなtrackerの比較は同じexp043入力と公開trackerを対照にする。exp016の旧数値を直接の基準にしない。旧入力の実験を続ける場合はexp016との比較を維持し、転移を別検証とする。',
    '点ごとのOOF予測、5fold評価head、特徴captureは確認したローカル回収物にない。OOF内容SHAは実配列の代用にならない。',
    '新規実験はKaggleのみ・週45GPU時間・課金なし。exp040は既に明示されたColab例外を持つ。quota残量を新たに確認したわけではない。',
]
p['evidence_register']['E043'] = 'experiments/exp043_x138_self_trained_head/{requirements.md,config.yaml,metrics.json,result.md,SESSION_NOTES.md}'
p['review_evidence'] = {
    'exp043_status': m['status'],
    'exp043_public_lb': m['public_lb'],
    'coordinate_validation': m['evidence']['coordinate_validation']['overall'],
    'coordinate_validation_by_embryo': m['evidence']['coordinate_validation']['by_embryo'],
    'source_sha256': {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [OLD, P43, P40]
    },
    'scope': '保存済み集計と実装の再読解。新しい予測生成・点ごとの誤差解析・公式評価はしていない。',
}
p['positive_evidence_ledger'] = [{
    'evidence_id': 'E043',
    'supported': '中心と6近傍との差分による224次元固定特徴から、既知中心への3軸残差を学び、同じ対応点での平均距離を動画分割で下げた。',
    'not_established': '32次元中心特徴だけの十分性、時間方向の誤差低下、identity識別、headのLB寄与、胚外一般化、接続・分裂への効果。',
    'implication': '位置を固定値として扱う前提を見直し、補正位置での特徴再取得と同一構成の対照を評価条件へ追加する。',
}]
p['closure_ledger'].append({
    'evidence_id': 'E040',
    'closed_instantiation': '(6時点固定特徴,直接対応教師,正規化BCEを確率clampで計算,4096窓/epoch,2胚holdout,10epochs)の旧実行は単体進行条件未達。',
    'not_closed': '数値安定化した同じ損失での学習結果、直接対応教師・多時点利用全般。',
    'reason': '本学習後にclamp域で勾配が消える不具合を修正。修正版は未学習。旧実行は親順位も低く、校正だけの問題ではないが、不具合だけで全低下を説明できる証拠もない。',
})
reviews = [
    ('I01', '維持・優先は条件付き', 'exp043はexp029の自己部分集合に負ける障害を検証しておらず、損失案の根拠は変わらない。ただしexp043でも同じ分裂障害が支配的とは限らない。', 'exp043で正解2娘が候補に存在し採点で消える件数を測り、小例での分裂loss配分対照から始める。'),
    ('I02', '維持・入力と診断を変更', '位置誤差が運動参照を汚し得ることへの関心は増すが、中心距離改善から変位誤差改善は導けない。', '補正前後の同一候補で時間差分誤差・参照支持・正解順位を測る。正しい参照と実参照を分ける。'),
    ('I03', '維持・位置候補への拡張を条件付きで追加', '補正前後のどちらが良いかは一様ではない。graph候補に加え同じ検出IDの位置候補を残す案へ接続できるが、動画平均の悪化だけで点ごとの選別可能性は示されない。', '元位置と補正位置の正解上限差があるか、その差をGTなしで選べるか確認。1ID1位置と合法な接続を共同選択し、重複nodeにしない。'),
    ('I04', '維持・測定条件を強める', '座標補正の残差も入力誤差だが、平均距離しかないため実誤差生成へ直ちに使えない。', '学習に使わなかった動画の点ごとの残差、時間・近傍相関、欠損長を保存してから生成と実OOF適応を検討する。'),
    ('I05', '最初の作業へ繰り上げ', '採用構成の入力・採点・復号が旧実験と異なるため、元の対照数値を使うと新手法の効果を誤認する。', 'exp043の同一入力と公開trackerによる接続・分裂の対照を取得し、各段階の破壊・回収を確認する。'),
    ('I06', '維持・特徴抽出位置を先に固定', '局所特徴に位置を補正する情報がある証拠は増えたが、identityを識別する証拠ではない。', '正しい補間特徴を対照とし、隣接/非隣接教師と同じ学習量を比較。観測分岐なしを非分裂確定とは扱わない。'),
    ('I07', '根拠を更新・無条件の先行推奨を撤回', 'exp040は未学習ではなく旧実装で悪化。数値不具合の修正後はまだ未測定。exp043からの帰結とは分ける。', '既存設計の修正版で学習成立・単体条件を再検証する範囲。旧入力のexp016対照を維持し、同時にexp043入力へ替えない。'),
    ('I08', '維持・後続', '座標補正は時間的な組織移動の推定ではないため、独立画像参照の有用性は未検証のまま。', '正しい点参照は有効で、実点参照だけ失敗する場合に独立画像参照を比較。'),
    ('I09', '維持・局在誤差と移動を分ける', 'exp043が推定する中心補正と母から娘への実移動は別の量。両者を同じノイズとして学ばせない。', '補正残差の不確実性と生物学的移動の分布を分け、等回収率で競合を減らすか測る。'),
    ('I10', '維持・実入力への転移条件を更新', '合成教師の有効性は検証されていない。実入力側の座標・特徴条件が変わる。', 'exp043の実残差を観測するまで独立jitterで代用しない。補間特徴を含む同一条件で実OOFへの転移を確認。'),
    ('I11', '維持・低優先', '座標局在の改善は蛍光量保存や娘組の画像量識別を支持しない。', '既知分裂と距離・密度を揃えた競合で追加情報を測る。'),
    ('I12', '優先を上げI05と同時に扱う', '補正は座標、特徴の抽出位置、候補採点、後処理に届き得る。headの平均位置誤差だけでは作用点が分からない。', 'まず補正あり/なしの自然な経路を比較し、必要時だけ座標と特徴を分けた診断を追加。診断上の不整合な入力を採用構成にしない。'),
]
p['reassessment'] = [dict(idea_id=i, decision=d, reason=r, next_test=t) for i, d, r, t in reviews]
by_id = {c['id']: c for c in p['idea_cards']}
for i, d, reason, test in reviews:
    c = by_id[i]
    c['review_20260924'] = dict(decision=d, reason=reason, next_test=test)
    if i != 'I07':
        c['evidence_ids'] = list(dict.fromkeys([*c['evidence_ids'], 'E043']))
        c['full_test'] += ' exp043へ適用する場合は同じ座標・補間特徴・公開tracker・後処理の対照を用意する。旧入力での比較結果を新入力での効果へ直接移さない。'
    if i in {'I01', 'I05', 'I12'}:
        c['cheap_test'] = test + ' ' + c['cheap_test']
by_id['I01']['counterevidence'] += ' exp029固有の障害を直しても、採用済みexp043に同じ障害がなければ利益は小さい。'
by_id['I03']['changed_mechanism'] += ' 発展案として同一検出IDの元位置と補正位置を排他的な状態として保持する。座標状態に整合する特徴・接続得点を使い、同じ点を2細胞として追加しない。'
by_id['I03']['nearest_prior_attempt'] += ' 位置候補への拡張はbacklog/joint_position_edges.mdとposition_mixture.mdにも関連する。現行保留状態を自動解除しない。'
by_id['I03']['counterevidence'] += ' 位置別に文脈依存trackerを評価する費用が増え、個別edgeの位置状態を混ぜると実在しないgraphになる。'
by_id['I03']['cheap_test'] += ' 元/補正位置のoracle利益と、画像・接続だけでの実選別を別測定する。動画単位の3件悪化だけでは候補追加しない。'
by_id['I05']['title'] = 'exp043と同じ入力で対照を作り、位置補正が接続判断へ届く段階を確かめる'
by_id['I05']['nearest_prior_attempt'] += ' exp043は中心距離のみで、対応するtracker指標は未取得。'
by_id['I07']['hypothesis'] = '直接対応教師が役立つかは、数値不具合を修正した同じ設計で再評価する必要がある。未学習だから優先するという根拠は撤回する。'
by_id['I07']['cheap_test'] = '旧実行の監査と数値不具合の修正を確認し、修正版の損失・勾配・小例適合を確認する。修正で精度が戻るとは仮定しない。'
by_id['I07']['compute_estimate'] = '旧実装のColab T4で2fold×10epochsが25,447.14秒（約7時間4分）。修正版の再学習時間は未測定。前回の9.77時間は過去のKaggle費用予測。'
by_id['I07']['counterevidence'] = '旧実行は両胚の接続回収・親順位・分裂回収で対照未達。数値不具合は確定したが、それだけが原因とは未確定。'
by_id['I07']['full_test'] = '既存exp040の修正版を同じ入力・分割・窓抽出で比較。両胚の単体条件が成立した場合だけ全graphへ進む。exp043入力への移植は別比較。'
by_id['I07']['kill_criterion'] = '修正版でも単体進行条件未達、予算不成立の場合。外側閾値探索で救済せず、多時点全般を棄却しない。'
by_id['I07']['confidence'] = 'C'
p['portfolio'] = [
    {'idea_id': 'I05', 'slot': 'compute_enabler', 'why': '最初の作業。新入力と同じ公開tracker対照を作り、後続の学習対象を残る誤りに合わせる。'},
    {'idea_id': 'I01', 'slot': 'safe', 'why': '本命の学習案は維持。ただしexp043に候補内の2娘選択障害が残る場合に優先し、小例・損失配分対照から進む。'},
    {'idea_id': 'I02', 'slot': 'orthogonal', 'why': '他細胞参照は維持。補正後の変位・支持の誤差を先に確認し、中心距離の改善から有効性を推定しない。'},
    {'idea_id': 'I03', 'slot': 'exploration', 'why': '複数graph保持に位置候補の共同選択という条件付き発展を追加。oracle利益とGTなし選別が必要。'},
    {'idea_id': 'I04', 'slot': 'exploration', 'why': '誤履歴学習は維持。新しい補正残差の時間相関を測定してから学習分布へ反映する。'},
]
p['rejected'] += [
    {'idea': 'Public LBの0.006改善を座標headの寄与と断定する', 'failed_gate': '構成全体が変わり、同一x138のheadなし対照がない。', 'reopen_condition': '同じ入力・候補・公開tracker・後処理による対照で寄与を分ける。'},
    {'idea': 'exp043の最終headを20学習動画へ適用してOOFと呼ぶ', 'failed_gate': '最終headは20動画すべての教師を学習済み。', 'reopen_condition': '同じ5foldの学習外予測を再生成するか、別の評価動画と露出条件を明示する。'},
    {'idea': '平均中心距離が改善したので履歴速度の誤差も減ったと仮定する', 'failed_gate': '時間的bias・分散・相関は未測定。', 'reopen_condition': '同一系列の補正前後で変位・加速度等の誤差を直接測る。'},
    {'idea': '元位置と補正位置を別nodeとして候補へ追加する', 'failed_gate': '同一細胞の重複予測を増やす。', 'reopen_condition': '同一検出IDに排他的な位置状態を割り当て、接続も同じ状態と整合させる。'},
]
(OUT / 'idea_portfolio.json').write_text(json.dumps(p, ensure_ascii=False, indent=2) + '\n')
print('Saved 12 reassessed ideas; exp043 evidence copied from metrics; original portfolio retained.')
