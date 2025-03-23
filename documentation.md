# MayaMeshRetarget コーディング規約

## 型アノテーション

### 基本方針

- すべての関数の引数と戻り値に型アノテーションを追加する
- レガシーなコメント形式の型ヒント（`# type: (...)` ）ではなく、PEP 484形式（`def func(arg: Type) -> ReturnType:`）を使用する
- カスタム型は `types.py` モジュールに定義する

### 型の定義

```python
from typing import Union, List, Optional, Dict, Tuple
import numpy as np
from maya.api import OpenMaya as om

# 基本的な型エイリアス
MeshPath = Union[om.MDagPath, str]
VertexArray = np.ndarray  # 頂点座標の配列
JointWeights = np.ndarray  # ジョイントウェイトの配列
```

### 型変換関数

OpenMaya型とNumPy配列間の変換は、専用の変換関数を使用する：

```python
def to_mpoint(point: np.ndarray) -> om.MPoint:
    """NumPy配列からMayaのMPointに変換する"""
    return om.MPoint(point[0], point[1], point[2])

def to_ndarray(point: om.MPoint) -> np.ndarray:
    """MayaのMPointからNumPy配列に変換する"""
    return np.array([point.x, point.y, point.z])
```

## コード構造

### 関数の複雑さ

- 複雑な関数（C901, PLR0915違反など）は小さな関数に分割する
- 1つの関数は単一の責任原則に従うべき

### パフォーマンス考慮

ログ出力などの高コスト操作は条件付きで実行する：

```python
# 変更前:
logger.debug(f"Complex result: {calculate_expensive_thing()}")

# 変更後:
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Complex result: {calculate_expensive_thing()}")
```

## 命名規則

- 変数、関数、メソッド：snake_case
- クラス：CamelCase
- 定数：UPPER_SNAKE_CASE
- プライベート関数/変数：_leading_underscore
- 非公開の内部関数：__double_leading_underscore

### インポート規則

- 標準ライブラリのインポート
- サードパーティライブラリのインポート  
- ローカルモジュールのインポート

が標準だが、pyproject.tomlがimport順序の警告を無効化している。

## エラー処理

- 明示的なエラーメッセージで例外を発生させる
- 不必要なtry-except-passパターンを避ける（代わりにcontextlib.suppressを使用）
- 戻り値としてNoneを返す場合は戻り値の型にOptionalを使用する

## ドキュメント

### Googleスタイルのドキストリング

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

### クラスドキュメント

クラスとそのすべてのパブリックメソッドにはドキュメント文字列を含める：

```python
class SomeClass:
    """クラスの説明。
    
    詳細な説明があれば、ここに記述します。
    """
    
    def __init__(self, arg: Type) -> None:
        """初期化メソッド。
        
        Args:
            arg: 引数の説明
        """
```

## Mayaコード規約

- `om` と `oma` としてそれぞれ `OpenMaya` と `OpenMayaAnim` をインポートするのは標準的なプラクティス
- MDagPathオブジェクトの適切な管理（参照カウンティング）
- ビューポート操作が必要な場合は、 `viewport_off` デコレータを使用
- アンドゥ操作のために `one_undo` デコレータを使用