import pandas as pd
import sys

# ---------------------------------------------------------
# 設定: ファイル名とパラメータ
# ---------------------------------------------------------
INPUT_ORIGINAL_FILE = 'motodata_1.csv'
INPUT_PROCESSED_FILE = 'motodata_1_pre_processed.csv'
OUTPUT_FILE = 'motodata_1_updated.csv'

# 結合に使用するキー列
MERGE_KEYS = ['名称', '場所']

def create_category_from_dummies(df, dummy_cols, prefix_to_remove):
    """
    ダミー変数の群（One-Hot）から、元の単一カテゴリ変数を復元・生成する関数。
    各行で値が1となっている列の名前を取得し、プレフィックスを除去して返す。
    """
    # データフレーム内に実際に存在する列のみを対象にする
    valid_cols = [col for col in dummy_cols if col in df.columns]
    
    if not valid_cols:
        return pd.Series('', index=df.index)

    # 各行で最大値を持つ列名（つまり1が入っている列）を取得
    new_category = df[valid_cols].idxmax(axis=1)
    
    # 全て0（該当なし）の行は空文字にする
    all_zero_mask = df[valid_cols].sum(axis=1) == 0
    new_category.loc[all_zero_mask] = ''
    
    # 列名から不要な接頭辞を削除
    return new_category.apply(
        lambda x: x.replace(prefix_to_remove, '') if x != '' else x
    )

def categorize_techniques(row, prefix):
    """
    技法に関する列をチェックし、優先順位に基づいて主要な技法ラベルを返す関数。
    優先順位: 磨消縄文 > 幅広沈線 > 細い沈線 > 縄文 > 無文
    """
    # 各技法の有無を取得（存在しない列は0扱い）
    has_masho = row.get(f'{prefix}_磨消縄文', 0) == 1
    has_futo_sen = row.get(f'{prefix}_幅広の沈線', 0) == 1
    has_hoso_sen = row.get(f'{prefix}_細い沈線', 0) == 1
    has_jomon = row.get(f'{prefix}_縄文', 0) == 1
    has_mumon = row.get(f'{prefix}_無文', 0) == 1
    has_nan = row.get(f'{prefix}_nan', 0) == 1

    if has_masho:
        return '磨消縄文'
    elif has_futo_sen:
        return '幅広の沈線'
    elif has_hoso_sen:
        return '細い沈線'
    elif has_jomon:
        return '縄文'
    elif has_mumon:
        return '無文'
    elif has_nan:
        return 'nan'
    else:
        return 'その他'

def main():
    """
    メイン実行処理
    1. 元データと前処理済みデータの読み込み
    2. 文字列のクリーニング（空白除去）
    3. 主モチーフカテゴリの生成
    4. データの結合
    5. 技法の分類処理
    6. 保存
    """
    
    # ---------------------------------------------------------
    # 1. データの読み込み & 文字列クリーニング
    # ---------------------------------------------------------
    try:
        print(f"Reading original data from {INPUT_ORIGINAL_FILE}...")
        df_original = pd.read_csv(INPUT_ORIGINAL_FILE)
        
        # Originalデータの空白除去処理（結合キーの不一致を防ぐため重要）
        string_columns_org = df_original.select_dtypes(include=['object']).columns
        for col in string_columns_org:
            df_original[col] = df_original[col].str.strip()
            # 全角スペース除去が必要な場合は以下を有効化
            # df_original[col] = df_original[col].str.replace(r'^[ 　]+|[ 　]+$', '', regex=True)
            
        print(f"Reading processed data from {INPUT_PROCESSED_FILE}...")
        df_processed = pd.read_csv(INPUT_PROCESSED_FILE)

    except FileNotFoundError as e:
        print(f"Error: File not found ({e.filename}).")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. 特徴量エンジニアリング: 主モチーフカテゴリの生成
    # ---------------------------------------------------------
    print("--- Generating Motif Categories ---")

    # 口縁部主モチーフリスト
    rim_motif_cols = [
        '口縁部_主モチーフ_無文','口縁部_主モチーフ_区画文_方形・窓枠状', '口縁部_主モチーフ_区画文_楕円形', '口縁部_主モチーフ_区画文_波状', 
        '口縁部_主モチーフ_曲線文_スペード文', '口縁部_主モチーフ_曲線文_円文', '口縁部_主モチーフ_曲線文_横に長いJ字文', 
        '口縁部_主モチーフ_曲線文_波状文', '口縁部_主モチーフ_曲線文_渦巻文', '口縁部_主モチーフ_曲線文_連弧文', 
        '口縁部_主モチーフ_直線文_三角文', '口縁部_主モチーフ_直線文_横位線', '口縁部_主モチーフ_直線文_縦位線', 
        '口縁部_主モチーフ_特殊文_刺突文', '口縁部_主モチーフ_特殊文_羽状文', '口縁部_主モチーフ_モチーフ不明:沈線', 
        '口縁部_主モチーフ_モチーフ不明:磨消縄文', '口縁部_主モチーフ_モチーフ不明:その他',
    ]

    # 胴部主モチーフリスト
    body_motif_cols = [
        '胴部_主モチーフ_無文', '胴部_主モチーフ_区画文_方形・窓枠状', '胴部_主モチーフ_区画文_楕円形', '胴部_主モチーフ_区画文_波状', 
        '胴部_主モチーフ_曲線文_スペード文', '胴部_主モチーフ_曲線文_横に長いJ字文', '胴部_主モチーフ_曲線文_波状文', 
        '胴部_主モチーフ_曲線文_渦巻文', '胴部_主モチーフ_曲線文_縦に長いJ字文', '胴部_主モチーフ_曲線文_連弧文', 
        '胴部_主モチーフ_直線文_斜文', '胴部_主モチーフ_直線文_縦位線','胴部_主モチーフ_特殊文_紡錘文', '胴部_主モチーフ_特殊文_その他', 
        '胴部_主モチーフ_モチーフ不明:沈線', '胴部_主モチーフ_モチーフ不明:磨消縄文','胴部_主モチーフ_モチーフ不明:その他'
    ]

    # 関数を適用して新しい列を作成（df_processedを使用）
    # ※ 注意: df_originalに列を追加していく
    df_original['口縁部_新_主モチーフ'] = create_category_from_dummies(
        df_processed, rim_motif_cols, '口縁部_主モチーフ_'
    )
    df_original['胴部_新_主モチーフ'] = create_category_from_dummies(
        df_processed, body_motif_cols, '胴部_主モチーフ_'
    )

    # ---------------------------------------------------------
    # 3. データの結合
    # ---------------------------------------------------------
    print(f"--- Merging DataFrames on {MERGE_KEYS} ---")
    try:
        # df_original（更新済み）と df_processed を結合
        df_merged = pd.merge(df_original, df_processed, on=MERGE_KEYS, how='inner')
        print(f"Merged row count: {len(df_merged)}")
        
    except KeyError:
        print(f"Error: Merge keys {MERGE_KEYS} not found in columns.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. 特徴量エンジニアリング: 技法の分類
    # ---------------------------------------------------------
    print("--- Categorizing Techniques ---")
    
    # apply関数で行ごとに分類処理を実行
    df_merged['口縁部_技法_分類'] = df_merged.apply(
        categorize_techniques, axis=1, prefix='口縁部_技法'
    )
    df_merged['胴部_技法_分類'] = df_merged.apply(
        categorize_techniques, axis=1, prefix='胴部_技法'
    )

    # ---------------------------------------------------------
    # 5. 結果の確認と保存
    # ---------------------------------------------------------
    print("\n--- Processed Data Preview (Head) ---")
    check_cols = ['名称', '場所', '口縁部_新_主モチーフ', '胴部_新_主モチーフ', '口縁部_技法_分類', '胴部_技法_分類']
    print(df_merged[check_cols].head())

    df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully saved updated data to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()