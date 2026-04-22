import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI環境がない場合のエラー回避
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
import gower
import os
import warnings

from sklearn.manifold import TSNE
import hdbscan

# 警告の抑制
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義
# ==============================================================================

INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'
TARGET_SITE = r'Yano'

# 固定シード値（アンカー抽出時と同じシードでクラスター番号を一致させる）
RANDOM_SEED = 0

MULTI_VALUE_FEATURES = [
    '口縁部_技法_沈線_特徴', '口縁部_技法_磨消縄文_沈線', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_沈線_特徴'
]
MULTI_VALUE_SEPARATOR = ', '

# Hokei_Makeshi_or_Habahiro 用の特徴量
FEATURES_HABAHIRO = [
    '口縁部_技法_沈線_特徴', '口縁部_形状', '口縁部_主文様同士が並行/対向', '口縁部_状態_退化', 
    '口縁部_文様が開放/閉鎖', '口縁部_文様方向', '口縁部_変形', '口縁部_直下', '頸部の傾向', '胴部_新_主モチーフ',
    '胴部_技法_沈線_特徴', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文'
]
FEATURES_HOKEI_MAKESHI = [
    '口縁部_技法_磨消縄文_施文順序','口縁部_技法_磨消縄文_図地','口縁部_技法_磨消縄文_沈線',
    '口縁部_技法_磨消縄文_縄文','口縁部_形状','口縁部_主文様同士が並行/対向','口縁部_状態_退化',
    '口縁部_文様が開放/閉鎖','口縁部_文様方向','口縁部_変形','口縁部_直下','頸部の傾向', '胴部_新_主モチーフ',
    '胴部_技法_沈線_特徴', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文'
]
FEATURES_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO))

# ==============================================================================
# 2. データ前処理関数
# ==============================================================================

def preprocess_multivalue_features(df, features_list):
    df_processed = df.copy()
    encoded_cols = []
    
    target_multivalue = [f for f in features_list if f in MULTI_VALUE_FEATURES and f in df.columns]
    categorical_cols = [f for f in features_list if f not in MULTI_VALUE_FEATURES and f in df.columns]

    for feature in target_multivalue:
        temp_unknown = '___UNKNOWN___'
        series = df_processed[feature].fillna(temp_unknown).replace('不明', temp_unknown)
        
        dummies = series.str.get_dummies(sep=MULTI_VALUE_SEPARATOR)
        dummies.columns = [f"{feature}_{col}" for col in dummies.columns]
        
        if f"{feature}_{temp_unknown}" in dummies.columns:
            dummies.drop(columns=[f"{feature}_{temp_unknown}"], inplace=True)
            
        duplicates = [c for c in dummies.columns if c in df_processed.columns]
        if duplicates:
            df_processed.drop(columns=duplicates, inplace=True)
            
        df_processed = pd.concat([df_processed, dummies], axis=1)
        encoded_cols.extend(dummies.columns)
        
    df_processed.drop(columns=target_multivalue, inplace=True, errors='ignore')
    categorical_cols = [c for c in categorical_cols if c not in encoded_cols]
    final_features = categorical_cols + encoded_cols
    
    return df_processed, final_features, categorical_cols, encoded_cols

# ==============================================================================
# 3. メイン処理
# ==============================================================================

def main():
    print("=== t-SNE Clustering (Hokei_Makeshi_or_Habahiro) Started ===")
    
    try:
        df_all = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # --- データのフィルタリング ---
    query_str = "(`口縁部_新_主モチーフ` == '区画文_方形・窓枠状') and (`口縁部_技法_分類` in ['磨消縄文', '幅広の沈線']) and (`口縁部` != 'なし')"
    df_scenario = df_all.query(query_str).copy()
    print(f"Filtered Data: {len(df_scenario)} rows")
    
    if df_scenario.empty:
        print("Error: No data after filtering.")
        return

    # 特徴量の前処理
    df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(df_scenario, FEATURES_UNION)
    df_processed['is_target'] = df_processed[SITE_COLUMN_NAME].str.contains(TARGET_SITE, regex=True, na=False)

    # 出力ディレクトリ作成
    output_dir = "Result_tSNE_Hokei_Makeshi"
    os.makedirs(output_dir, exist_ok=True)

    # --- Gower距離計算 (Uniform) ---
    print("\n--- Calculating Gower Distance ---")
    data_mining = df_processed[final_features].replace('不明', np.nan)
    weights = np.ones(len(data_mining.columns)) / len(data_mining.columns)
    dist_matrix = gower.gower_matrix(data_mining, weight=weights)

    # --- t-SNE 実行 ---
    print(f"--- Running t-SNE (Perplexity=30, Seed={RANDOM_SEED}) ---")
    reducer = TSNE(n_components=2, perplexity=30, metric='precomputed', init='random', random_state=RANDOM_SEED)
    coords = reducer.fit_transform(dist_matrix)

    # --- HDBSCAN クラスタリング ---
    print("--- Running HDBSCAN ---")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
    clusters = clusterer.fit_predict(coords.astype(np.float64))

    # プロット用データフレーム作成
    df_result = df_scenario[['名称', '場所']].copy()
    df_result['X'] = coords[:, 0]
    df_result['Y'] = coords[:, 1]
    df_result['cluster'] = clusters
    df_result['is_target'] = df_processed['is_target'].values

    # === Matplotlib 白黒印刷用 散布図作成 ===
    print("--- Generating Black & White Plot ---")
    
    # 4.1インチ × 4.0インチ (dpi=200で 820×800 px の画像を出力)
    fig, ax = plt.subplots(figsize=(4.1, 4.0))

    # 白黒印刷用のスタイル辞書 (色ではなく形で区別)
    cluster_styles = {
        4:  {'fc': '#000000', 'ec': 'black', 'marker': 'o', 'label': 'CL4 (定型)'},
        2:  {'fc': '#FFFFFF', 'ec': 'black', 'marker': '^', 'label': 'CL2 (非定型)'},
        0:  {'fc': '#777777', 'ec': 'black', 'marker': 's', 'label': 'CL0'},
        1:  {'fc': '#FFFFFF', 'ec': 'black', 'marker': 'D', 'label': 'CL1'},
        3:  {'fc': '#000000', 'ec': 'black', 'marker': '*', 'label': 'CL3'},
        5:  {'fc': '#FFFFFF', 'ec': 'black', 'marker': 'v', 'label': 'CL5'},
        -1: {'fc': '#333333', 'ec': 'none',  'marker': 'x', 'label': 'Noise'}
    }

    unique_clusters = sorted(df_result['cluster'].unique())
    
    # 背景の薄いグリッド線 (視認性を上げるため)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    # 描画順を制御 (ノイズは最背面、CL4・CL2は最前面に描画)
    draw_order = [-1] + [c for c in unique_clusters if c not in [-1, 2, 4]] + [c for c in unique_clusters if c in [2, 4]]

    for c in draw_order:
        if c not in unique_clusters:
            continue
            
        subset = df_result[df_result['cluster'] == c]
        
        # 辞書にない未知のクラスターが出た場合のフォールバック
        st = cluster_styles.get(c, {'fc': '#555555', 'ec': 'black', 'marker': 'P', 'label': f'CL{c}'})
        
        if c == -1:
            # ノイズ: バツ印、小さめ、半透明にして背面に沈める
            ax.scatter(subset['X'], subset['Y'], label=st['label'], marker=st['marker'], 
                       color=st['fc'], s=30, linewidths=1.0, zorder=1, alpha=0.5)
        else:
            # CL4とCL2はサイズを少し大きくして目立たせる
            size = 65 if c in [2, 4] else 50
            z_order = 3 if c in [2, 4] else 2
            
            ax.scatter(subset['X'], subset['Y'], label=st['label'], marker=st['marker'], 
                       facecolors=st['fc'], edgecolors=st['ec'], linewidths=1.0, 
                       s=size, zorder=z_order, alpha=0.9)

    # フォントサイズとラベルの設定
    ax.set_title('口縁部 方形・窓枠状の特徴分布\n(t-SNE)', fontsize=16, pad=10)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.tick_params(axis='both', labelsize=11)

    # 凡例の設定 (重なりを防ぎつつ、枠内に収める)
    ax.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "tSNE_ScatterPlot_BW.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"-> 図版保存先: {plot_path}")
    print("=== All Analysis Completed ===")

if __name__ == "__main__":
    main()