import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 設定: 入出力ファイル名
# ---------------------------------------------------------
INPUT_FILE = 'motodata_1_updated.csv'
OUTPUT_FILE = 'motodata_1_processed.csv'

def main():
    """
    更新版データセットの前処理スクリプト
    主な処理:
    1. カテゴリ変数のダミー変数化（One-Hot Encoding）
       - 新しい特徴量（新_主モチーフ、技法分類など）を含む
    2. 複数回答項目（カンマ区切り）の展開処理
    """

    # データの読み込み
    print(f"Reading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        print("元データで重複している列名があります:", duplicated_cols)

    # ---------------------------------------------------------
    # 1. 通常のダミー変数化 (One-Hot Encoding)
    # ---------------------------------------------------------

    columns_to_dummy = [
        '口縁部_新_主モチーフ',
        '胴部_新_主モチーフ',
        '口縁部_技法_分類',
        '胴部_技法_分類'
    ]

    print("--- Processing One-Hot Encoding ---")
    
    # pandasのget_dummiesで変換 (欠損値NaNも一つのカテゴリとして扱う)
    df_processed = pd.get_dummies(df, columns=columns_to_dummy, dummy_na=True)

    # 生成されたbool型の列をint型(0/1)に変換
    bool_cols = df_processed.select_dtypes(include='bool').columns
    df_processed[bool_cols] = df_processed[bool_cols].astype(int)


    # ---------------------------------------------------------
    # 2. 結果の保存と確認
    # ---------------------------------------------------------
    print("\n--- Check generated columns (Example: '口唇部_装飾') ---")
    print([col for col in df_processed.columns if '口唇部_装飾' in col])
    
    print("\n--- Processed DataFrame Info ---")
    df_processed.info()

    # CSV出力
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved preprocessed data to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()