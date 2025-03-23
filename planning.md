# MayaMeshRetarget Codebase Improvements

## 1. コードスタイルと静的解析の問題点

Ruff と pyright による静的解析の結果、以下の問題点が見つかりました：

### 型アノテーションの問題

1. **関数引数の型アノテーション不足**:
   - 全体で約216件の関数引数に型アノテーションが欠如しています（ANN001）
   - 特に util.py や cluster.py に多く見られます

2. **戻り値の型アノテーション不足**:
   - 約45件の公開関数（ANN201）と約21件の非公開関数（ANN202）に戻り値の型アノテーションがありません
   - 関数定義は古いコメント形式（`# type: (...) -> ...`）を使用しており、最新のPEP 484形式（`def func(...) -> return_type:`）に更新が必要です

3. **型の不一致**:
   - numpy.ndarray と Maya の MPoint 型の間の不一致（inpaint.py）
   - RetargetableObject と MeshObject 型の不一致（test_performance.py）
   - numpy.around の decimals パラメータに float が指定されている問題（cluster.py）
   - NDArray と list[int] の不一致（utils.py の戻り値型）
   - dia_array と dia_matrix の不一致（inpaint.py）

### コード構造の問題

1. **複雑すぎる関数**:
   - refine_clusters_by_topology（C901、PLR0915）が複雑すぎるため、リファクタリングが必要です
   - ステートメント数過多の関数（PLR0915）が複数存在します

2. **命名規則の問題**:
   - 関数内での大文字変数の使用（N806、44件）
   - 無効な関数名（N802、30件）：PEP8ではスネークケースが推奨されています
   - OpenMaya を om としてインポートするのは N813 違反ですが、Mayaコードでは一般的なので無視設定にすべきです

3. **ドキュメントの問題**:
   - 未ドキュメントのパブリックメソッド（D102、19件）
   - 未ドキュメントの __init__ メソッド（D107、7件）
   - ドキュメント文字列の構造の問題（D205、5件）

4. **未使用の引数**:
   - 未使用の関数引数（ARG001、8件）
   - 未使用のメソッド引数（ARG002、4件）

5. **その他の問題**:
   - ambiguous-unicode-character：コメントやドキュメント内の曖昧なUnicode文字
   - 長すぎる行（E501）
   - 未定義の名前参照（F821）
   - NumPyの属性アクセスエラー（np.warningsなど）

## 2. 改善計画

### A. 型アノテーションの改善

1. **最新の型アノテーション形式への移行**:
   ```python
   # 変更前:
   def function(argument):
       # type: (Type) -> ReturnType
   
   # 変更後:
   def function(argument: Type) -> ReturnType:
   ```

2. **カスタム型の定義**:
   ```python
   from typing import Union, List, TypeVar
   import numpy as np
   from maya.api import OpenMaya as om
   
   # カスタム型の定義
   MeshPath = Union[om.MDagPath, str]
   VertexArray = np.ndarray
   
   # 使用例
   def process_mesh(mesh_path: MeshPath, vertices: VertexArray) -> None:
       ...
   ```

3. **型変換関数の作成**:
   ```python
   def convert_ndarray_to_mpoint(point: np.ndarray) -> om.MPoint:
       """NumPy配列からMayaのMPointに変換する関数"""
       return om.MPoint(point[0], point[1], point[2])
   ```

### B. コード複雑性の改善

1. **パフォーマンス改善**:
   ```python
   # 変更前:
   def expensive_calculation():
       logger.debug(f"Complex result: {calculate_expensive_thing()}")
   
   # 変更後:
   def expensive_calculation():
       if logger.isEnabledFor(logging.DEBUG):
           logger.debug(f"Complex result: {calculate_expensive_thing()}")
   ```

2. **エラー修正**:
   - numpy.around の decimals 引数を整数に変更:
   ```python
   # 変更前
   rounded_weights = np.around(weights, precision)
   
   # 変更後 
   rounded_weights = np.around(weights, int(precision))
   ```

   - float のインデックスアクセスを修正:
   ```python
   # 変更前
   value = some_float[0]  # エラー
   
   # 変更後
   value = float(some_float)  # または適切な変換
   ```

### C. ドキュメントの改善

1. **Google スタイルのドキュメント文字列の統一**:
   ```python
   def function(arg1: Type1, arg2: Type2) -> ReturnType:
       """関数の概要。
       
       Args:
           arg1: 最初の引数の説明
           arg2: 2番目の引数の説明
           
       Returns:
           戻り値の説明
           
       Raises:
           ValueError: エラーが発生する条件
       """
   ```

2. **クラスと重要なメソッドのドキュメント追加**:
   - 特に mesh.py, joint.py のクラスとパブリックメソッド
   - __init__ メソッドの引数説明

### D. 構成と設定の改善

1. **pyproject.toml の更新**:
   - 非推奨設定の修正（fixable など）
   - 適切なイグノア設定の追加（Maya固有の命名規則など）
   ```toml
   [tool.ruff.lint]
   ignore = [
     "D100",  # 公開モジュールのdocstringが必要
     "D104",  # パッケージのdocstringが必要
     "D203",  # クラスdocstringの前に1行空行が必要
     "D213",  # マルチライン docstring の概要が1行目に必要
     "N813",  # CamelCase import as lowercase (Maya標準のom等を許可)
   ]
   ```

### E. エラー修正の優先順位

1. 型エラー：実行時エラーの原因になる可能性のある問題
   - numpy.around の decimals パラメータ型（整数に修正）
   - __getitem__ メソッド呼び出しエラー（配列アクセスの修正）
   - RetargetableObject と MeshObject の型変換

2. 実装の一貫性：コードの可読性と保守性を向上させる改善
   - NumPy NDArray の戻り値型とリスト型の適切な変換
   - dia_array と dia_matrix の互換性の確保

3. スタイルとドキュメント：長期的な保守性向上
   - 命名規則の統一
   - ドキュメント改善

## 3. 具体的な実装タスク

1. **✅ 型定義ファイルの作成（types.py）**
   ```python
   """プロジェクト共通の型定義。"""
   
   from typing import Union, List, Sequence, TypeVar, Any
   import numpy as np
   from numpy.typing import NDArray
   from maya.api import OpenMaya as om
   
   # 共通型エイリアス
   MeshPath = Union[om.MDagPath, str]
   VertexArray = NDArray[np.float64]
   JointWeights = NDArray[np.float64]
   
   # 変換関数
   def to_mpoint(point: np.ndarray) -> om.MPoint:
       """NumPy配列からMayaのMPointに変換する"""
       return om.MPoint(point[0], point[1], point[2])
   
   def to_ndarray(point: om.MPoint) -> np.ndarray:
       """MayaのMPointからNumPy配列に変換する"""
       return np.array([point.x, point.y, point.z])
   ```

2. **✅ cluster.py の改善**
   - around() 関数の引数型修正
   - refine_clusters_by_topology() のリファクタリング

3. **✅ inpaint.py の改善**
   - ndarray と MPoint 間の適切な変換処理
   - SciPy配列型の適切な変換

4. **✅ utils.py の改善**
   - 戻り値型の一貫性の確保
   - float[0] のようなインデックスアクセスの修正

5. **✅ documentation.md ファイルの作成**
   - コード規約の文書化
   - 型アノテーションのガイドライン
   - 命名規則と例

## 4. 実施した改善

1. **✅ types.py モジュールの作成**
   - カスタム型の定義（MeshPath, VertexArray, JointWeights など）
   - Maya と NumPy 間の型変換関数の実装

2. **✅ 主要ファイルの型アノテーション改善**
   - cluster.py: 引数と戻り値の型アノテーション追加、np.around のパラメータ型修正
   - inpaint.py: MPoint と ndarray 間の変換処理の追加
   - util.py: 戻り値型の一貫性の確保、型アノテーションの追加

3. **✅ ドキュメント作成**
   - documentation.md: コーディング規約、型アノテーションのガイドライン、命名規則

## 5. 残りの課題

1. **型の互換性エラー**
   - RetargetableObject と MeshObject 間の型変換（test_performance.py）
   - numpy.bool と builtins.bool の互換性（cluster.py）
   - dia_array と dia_matrix の互換性（inpaint.py）

2. **SciPy 配列のアクセスエラー**
   - coo_array, dia_array の __getitem__ メソッド未定義エラー

3. **NumPy の属性アクセスエラー**
   - np.warnings 属性未定義エラー（logic.py）

4. **戻り値型の不一致**
   - NDArray[intp] が list[int] に割り当て不可能（utils.py）
   - MDagPath リストが str リストに割り当て不可能（logic.py）

5. **float[0] のようなインデックスエラー**
   - utils.py, registration/utils.py の float へのインデックスアクセス

## 6. 今後の運用方針

1. **継続的な静的解析の導入**
   - GitHub Actions を使用した自動チェック
   - pre-commit hooks の設定

2. **型スタブの管理と更新**
   - NumPy、SciPy、Maya用の型スタブの定期的な更新
   - 型の互換性チェックを自動化

3. **コードレビュープロセス**
   - 型チェック合格の必須化
   - ドキュメント更新の確認

4. **グラデュアルリファクタリング**
   - 機能ごとに優先順位を決めて段階的に改善
   - 変更影響範囲を最小限に抑えるための戦略

## 5. 型アノテーション改善例

### cluster.py 修正例:
```python
def cluster_vertices_by_skin_weight(
    mesh_paths: list[MeshPath], 
    precision: int = 3, 
    min_vertices_per_cluster: int = 6
) -> np.ndarray:
    """Cluster vertices based on their weight similarity."""
    # ...
    rounded_weights = np.around(weights, precision)  # precision は int 型に
    # ...
```

### inpaint.py 修正例:
```python
def __compute_cotangent(v1: om.MPoint, v2: om.MPoint, v3: om.MPoint) -> float:
    """Calculate the cotangent of the angle between three points."""
    # ...

# 呼び出し側
cot = __compute_cotangent(
    to_mpoint(points[i]), 
    to_mpoint(points[j]),
    to_mpoint(points[k])
)
```

### utils.py 修正例:
```python
def parse_joint_names(input_str: str) -> tuple[list[int], list[int], list[str]]:
    """Parse joint names from input string."""
    indices1 = np.array([...], dtype=np.int_)
    indices2 = np.array([...], dtype=np.int_)
    
    # 戻り値の型一致
    return indices1.tolist(), indices2.tolist(), joint_names
```
