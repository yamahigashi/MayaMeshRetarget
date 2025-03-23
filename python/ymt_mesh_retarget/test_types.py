"""テスト用のスクリプト：型変換ユーティリティ

このスクリプトはtype.pyで定義された型変換ユーティリティをテストします。
"""

import numpy as np
from maya.api import OpenMaya as om

from .types import (
    to_mpoint, 
    to_ndarray,
    to_mvector,
    to_ndarray_from_vector,
    ensure_list
)


def test_mpoint_conversion():
    """MPoint変換のテスト"""
    # NumPy配列からMPointへの変換
    np_array = np.array([1.0, 2.0, 3.0])
    mpoint = to_mpoint(np_array)
    
    print("NumPy → MPoint 変換テスト:")
    print(f"  入力: {np_array}")
    print(f"  出力: {mpoint.x}, {mpoint.y}, {mpoint.z}")
    
    # MPointからNumPy配列への変換
    converted_array = to_ndarray(mpoint)
    print("\nMPoint → NumPy 変換テスト:")
    print(f"  入力: {mpoint.x}, {mpoint.y}, {mpoint.z}")
    print(f"  出力: {converted_array}")
    
    # 往復変換の精度を検証
    is_equal = np.allclose(np_array, converted_array)
    print(f"\n往復変換の正確性: {'成功' if is_equal else '失敗'}")


def test_mvector_conversion():
    """MVector変換のテスト"""
    # NumPy配列からMVectorへの変換
    np_array = np.array([1.0, 2.0, 3.0])
    mvector = to_mvector(np_array)
    
    print("\nNumPy → MVector 変換テスト:")
    print(f"  入力: {np_array}")
    print(f"  出力: {mvector.x}, {mvector.y}, {mvector.z}")
    
    # MVectorからNumPy配列への変換
    converted_array = to_ndarray_from_vector(mvector)
    print("\nMVector → NumPy 変換テスト:")
    print(f"  入力: {mvector.x}, {mvector.y}, {mvector.z}")
    print(f"  出力: {converted_array}")
    
    # 往復変換の精度を検証
    is_equal = np.allclose(np_array, converted_array)
    print(f"\n往復変換の正確性: {'成功' if is_equal else '失敗'}")
    

def test_ensure_list():
    """リスト変換のテスト"""
    # NumPy配列からリストへの変換
    np_array = np.array([1, 2, 3, 4, 5])
    python_list = ensure_list(np_array)
    
    print("\nNumPy配列 → Pythonリスト 変換テスト:")
    print(f"  入力: {np_array}, 型: {type(np_array)}")
    print(f"  出力: {python_list}, 型: {type(python_list)}")
    
    # すでにリストの場合
    already_list = [10, 20, 30]
    result = ensure_list(already_list)
    
    print("\nすでにリストの場合:")
    print(f"  入力: {already_list}, 型: {type(already_list)}")
    print(f"  出力: {result}, 型: {type(result)}")
    print(f"  同一オブジェクト: {already_list is result}")


if __name__ == "__main__":
    test_mpoint_conversion()
    test_mvector_conversion()
    test_ensure_list()
    
    print("\nすべてのテストが完了しました")