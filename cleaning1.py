import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 設定: 入出力ファイル名
# ---------------------------------------------------------
INPUT_FILE = 'motodata_1.csv'
OUTPUT_FILE = 'motodata_1_pre_processed.csv'

def main():
    """
    卒業論文用データ前処理スクリプト
    主な処理:
    1. 文字列の不要な空白除去
    2. カテゴリ変数のダミー変数化（One-Hot Encoding）
    3. 複数回答項目（カンマ区切り）の展開処理
    """
    
    # データの読み込み
    print(f"Reading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # 1. 文字列データのクリーニング
    # ---------------------------------------------------------
    print("--- Processing string columns (stripping whitespace) ---")
    
    # オブジェクト型（文字列）の列を抽出し、前後の空白を除去
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].str.strip()
        # 必要に応じて全角スペースを除去する場合は以下を有効化
        # df[col] = df[col].str.replace(r'^[ 　]+|[ 　]+$', '', regex=True)

    print(f"Cleaned {len(string_columns)} string columns.")

    # ---------------------------------------------------------
    # 2. 通常のダミー変数化 (One-Hot Encoding)
    # ---------------------------------------------------------
    # ダミー変数化対象のカラムリスト
    columns_to_dummy = [
        '口唇部', '口唇部_断面形', '口唇部_器面調整',
        '口縁部', '口縁部_形状', '口縁部_方向', '口縁部直下', '口縁部_技法_縄文_特徴',
        '口縁部_技法_磨消縄文_縄文', '口縁部_技法_磨消縄文_施文順序', '口縁部_技法_磨消縄文_図地', '口縁部_器面調整',
        '口縁部_主モチーフ', '口縁部_主モチーフ_区画文', '口縁部_主モチーフ_曲線文', '口縁部_主モチーフ_直線文',
        '口縁部_主モチーフ_特殊文', '口縁部_状態_連続・非連続', '口縁部_状態_退化', '口縁部_列数', '口縁部_文様が開放/閉鎖',
        '口縁部_変形', '口縁部_主文様同士が並行/対向', '口縁部_文様方向', '口縁部_主モチーフ内モチーフ', '口縁部_主モチーフ内モチーフ_区画文',
        '口縁部_主モチーフ内モチーフ_曲線文', '口縁部_主モチーフ内モチーフ_直線文', '口縁部_主モチーフ内モチーフ_特殊文',
        '口縁部_間モチーフ', '口縁部_間モチーフ_区画文', '口縁部_間モチーフ_曲線文', '口縁部_直線文', '口縁部_特殊文', '口縁部_内面モチーフ',
        '頸部', '頸部の傾向',
        '胴部', '胴部方向', '胴部_技法_縄文_特徴',
        '胴部_技法_磨消縄文_縄文', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_器面調整',
        '胴部_主モチーフ', '胴部_主モチーフ_区画文', '胴部_主モチーフ_曲線文', '胴部_主モチーフ_直線文',
        '胴部_主モチーフ_特殊文', '胴部_状態_連続・非連続', '胴部_状態_退化', '胴部_列数', '胴部_文様が開放', '胴部_変形',
        '胴部_主文様同士が並行/対向', '胴部_文様方向', '胴部_主モチーフ内モチーフ', '胴部_主モチーフ内モチーフ_区画文',
        '胴部_主モチーフ内モチーフ_曲線文', '胴部_主モチーフ内モチーフ_直線文', '胴部_主モチーフ内モチーフ_特殊文', '胴部_間モチーフ',
        '胴部_間モチーフ_区画文', '胴部_間モチーフ_曲線文', '胴部_間モチーフ_直線文', '胴部_間モチーフ_特殊文', '胴部_内面モチーフ',
        '上端連繋', '下端連繋', '横位連繫線'
    ]

    # pandasのget_dummiesで変換 (欠損値NaNも一つのカテゴリとして扱う)
    df_processed = pd.get_dummies(df, columns=columns_to_dummy, dummy_na=True)

    # 生成されたbool型の列をint型(0/1)に変換
    bool_cols = df_processed.select_dtypes(include='bool').columns
    df_processed[bool_cols] = df_processed[bool_cols].astype(int)

    # ---------------------------------------------------------
    # 3. 複数選択項目の処理 (Multi-label processing)
    # ---------------------------------------------------------
    # カンマ区切りなどで複数の値が入っているカラムのリスト
    multi_select_columns = [
        '口唇部_装飾',
        '口縁部_技法',
        '口縁部_技法_沈線_特徴',
        '口縁部_技法_磨消縄文_沈線',
        '胴部_技法',
        '胴部_技法_沈線_特徴',
        '胴部_技法_磨消縄文_沈線',
        '頸部の状態',
    ]

    print("--- Processing multi-select columns ---")

    for col_name in multi_select_columns:
        # 元データがNaNのレコードを保持するためのフラグ列を作成
        df_processed[f'{col_name}_nan'] = df_processed[col_name].isnull().astype(int)

        # データクリーニング:
        # 1. 欠損値を空文字に変換
        # 2. カンマで分割してリスト化
        # 3. 各要素の空白を除去
        # 4. パイプ(|)区切りで再結合
        cleaned_series = df_processed[col_name].fillna('').str.split(',') \
            .apply(lambda lst: [str(item).strip() for item in lst if str(item).strip()]) \
            .str.join('|')

        # get_dummiesで展開 (sep='|'を指定)
        dummies = cleaned_series.str.get_dummies(sep='|').add_prefix(f'{col_name}_')
        
        # 元のデータフレームに結合
        df_processed = pd.concat([df_processed, dummies], axis=1)

    # 処理済みの元の列を削除
    df_processed = df_processed.drop(columns=multi_select_columns)

    # ---------------------------------------------------------
    # 4. 結果の保存と確認
    # ---------------------------------------------------------
    print("\n--- Processed DataFrame Info ---")
    df_processed.info()

    # CSV出力
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved preprocessed data to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()