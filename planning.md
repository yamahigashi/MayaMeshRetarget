# Implementation Plan

## 1. 改善の目的と概要

本プランでは、`create_optimized_correspondence_points` 関数における候補絞り込みロジックを強化し、より信頼性の高い対応点 (Correspondence Point) を生成するための実装方針をまとめます。

主な改善点は以下のとおりです:

- 距離・レイ品質など単純な指標に加えて、**法線 (Normal)** や **ラプラシアン (Laplacian)**、**ウェイトベクタの類似度** など多次元的な指標を導入したスコア評価を行う。
- スコア計算をモジュール化し、保守・拡張を容易にする。
- 既存のレイキャストやマッピングとの整合性を崩さずに段階的にアップデートできる設計とする。
- 計算コストが増える分については、事前計算の導入やバッチ処理の最適化などでカバーする。

---

## 2. 改修範囲と設計方針

### 2.1 改修対象モジュール
- `create_optimized_correspondence_points`
  - Raycast結果を受け取って、最終的に `CorrespondencePoint` を生成しているコアロジック。
  - ここに多次元的な評価指標（法線やウェイト、ラプラシアンなど）を組み込み、スコアリングを強化する。

- `RegistrationOptions` (もしくは類似の設定クラス)
  - 新たに `scoring_components` のようなリストや設定を追加し、ユーザが各スコア計算の有効・無効や重み付けを制御できるようにする。

- `ScoringComponent` (新規作成)
  - 各種スコア計算をカプセル化したクラスまたはインターフェース群を導入する。

### 2.2 スコアリングの全体像

1. レイキャストで得られたヒット情報（空間距離やレイ品質など）をもとに「仮対応候補」を生成。
2. **法線・ウェイト・ラプラシアン**など事前に計算可能な情報を頂点インデックスごとにキャッシュしておく。
3. `create_optimized_correspondence_points` 内で各候補のスコアを算出する際に、キャッシュした値を参照してスコアを合算。
4. 合算したスコアを降順ソートして上位N件だけ残す (デフォルトは `max_points_per_target` など)。

### 2.3 実装のポイント

1. **頂点法線 / ラプラシアンの事前計算**  
   - Maya API 等を用いて、各メッシュの頂点ごとの法線を取得し `source_normals[vtx_idx]`, `target_normals[vtx_idx]` のような形で格納。
   - ラプラシアンは近傍頂点からの差分ベクトルを計算し、同様にキャッシュしておく。
   - これらは「メッシュ読み込み時の初期化フェーズ」あるいは `MeshRegistration` コンストラクタ実行後の任意の時点で計算し、保管しておく。

2. **ウェイトベクトルの取得・正規化**  
   - 既に `find_correspondence_using_skeleton` などでスキンウェイトを取り扱っているため、同様に全頂点のウェイトベクトル (influence数分) を取得する。
   - Joint名のマッチングが取れているものだけを対象とした部分ベクトルを作るのも一手。  
   - コサイン類似度などを計算しやすいように正規化（L2ノルム）も事前に行っておく。

3. **スコアコンポーネントの導入**  
   - 例として以下のコンポーネントを作る:
     - `DistanceScoring` (空間距離ベース)
     - `RayQualityScoring` (レイ情報)
     - `NormalScoring` (法線ベース)
     - `WeightVectorScoring` (ウェイトベクタ類似度)
     - `LaplacianScoring` (ラプラシアンベース)
   - インターフェース (抽象クラス) `IScoringComponent` を定義しておき、`compute_score(context) -> float` の形で実装。

4. **`create_optimized_correspondence_points` のフロー**  
   - `raycast_result_array` をループし、候補点がある場合は以下の流れで評価:
     1. ソース頂点情報（法線、ウェイト、ラプラシアンなど）取得
     2. ターゲット側については三角形＋バリセン補間 or 最近傍頂点の情報から(法線、ウェイト、ラプラシアン) を取得
     3. スコアコンポーネントを順番に呼び出して合算
     4. スコアがしきい値以上なら候補に加える
   - 最後にスコア降順ソート、上位 N 件を `CorrespondencePoint` に変換して返す。

5. **パフォーマンス最適化**  
   - 法線/ラプラシアン/ウェイトなどの取得はループ内で計算せず、**必ず事前計算済みの配列から取り出す**。
   - レイキャスト回数はすでにバッチ処理されているが、スコア計算自体も可能な限りNumPyベースのベクトル演算化を検討する。
   - スレッド数やバッチサイズ (`batch_size`) はオプションで調整できるようにする。

---

## 3. 実装ステップ ✓

以下は具体的な作業手順です。

1. ✓ **(準備) 頂点法線・ラプラシアン・ウェイトベクトルの事前計算**  
   - ✓ `MeshObject` に `normals`, `laplacians`, `weights` プロパティを持たせるか、または別途キャッシュマップを作成する。
   - ✓ Maya API (または NumPyで独自計算) を用いて全頂点分を取得。  
   - ✓ ウェイトベクトルはスキン影響数分の配列を取り、正規化するかは運用次第。

2. ✓ **(新規) `scoring_components.py` or `scoring.py` の作成**  
   - ✓ `class IScoringComponent` のインターフェースを定義
   - ✓ `class DistanceScoring`, `class NormalScoring`, `class WeightScoring`, などを実装
     - ✓ `compute_score(self, context) -> float`
     - ✓ contextは辞書形式で `{'src_pos':..., 'tar_pos':..., 'normal_src':..., 'normal_tar':..., ...}` などを渡す

3. ✓ **(修正) `RegistrationOptions` にスコアリング設定を追加**  
   - ✓ `scoring_components: list[IScoringComponent] = field(default_factory=list)` のようなメンバを追加
   - ✓ 既存の `distance_weight` / `ray_weight` などは `scoring_components` の初期化パラメータとして移行するか、後方互換のために暫定的に残しておく。

4. ✓ **(修正) `create_optimized_correspondence_points`**  
   - ✓ スコア計算部分を関数やループ内で以下のように変更:
     1. ✓ `context` の作成 (ソース/ターゲットの頂点情報 + レイ情報)
     2. ✓ `total_score = sum(comp.compute_score(context) for comp in scoring_components)`
     3. ✓ しきい値や上位N件のフィルタリングを実施
   - ✓ 既存の `distance_weight` や `ray_weight` を使った処理は、一旦 `DistanceScoring`, `RayQualityScoring` としてカプセル化する。

5. **テスト実行とパフォーマンス評価** (残タスク)
   - 小規模メッシュで挙動確認 (法線が反映されているか、スコアが妥当に計算されているか)  
   - 大規模メッシュ・高サンプル数でパフォーマンス測定し、バッチサイズやスレッド数を調整する。
   - 結果を可視化 (`visualize_correspondences`) して妥当性を検証。

6. ✓ **ドキュメント整備**  
   - ✓ 新たに導入した `scoring_components` の使い方 (例: ユーザが `options.scoring_components` に各コンポーネントをappendして設定する方法) を Readme / docstring などに明記。

### 実装内容まとめ ✓

1. 新たに `scoring_components.py` モジュールを作成し、スコア計算ロジックをカプセル化
   - ✓ 基底クラス `IScoringComponent` とその実装
   - ✓ 距離・レイ品質の既存スコアリングをコンポーネント化
   - ✓ 新規に法線・ウェイト・ラプラシアンスコアリングを実装

2. `MeshObject` クラスに拡張機能を追加
   - ✓ 頂点法線・ラプラシアン・ウェイトベクトルのキャッシング機能
   - ✓ 事前計算機能による最適化

3. `RegistrationOptions` クラスの拡張
   - ✓ スコアリングコンポーネント設定の追加
   - ✓ 後方互換性確保

4. `create_optimized_correspondence_points` 関数の強化
   - ✓ コンテキスト生成による情報集約
   - ✓ コンポーネントベースのスコアリング機構
   - ✓ 既存機能との互換性維持

5. `MeshRegistration` クラスを拡張
   - ✓ mesh data の事前計算機能
