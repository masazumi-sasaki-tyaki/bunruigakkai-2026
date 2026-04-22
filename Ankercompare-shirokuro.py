import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import japanize_matplotlib  
import seaborn as sns
import plotly.express as px
import gower
import os
import warnings

from sklearn.manifold import TSNE
import hdbscan
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義
# ==============================================================================

INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'

ANCHOR_SEED = 0            
ANCHOR_CL4_ID = 4          
ANCHOR_CL2_ID = 2          

NUM_SIMULATIONS = 100      
TARGET_K = 5               

MULTI_VALUE_FEATURES = [
    '口縁部_技法_沈線_特徴', '口縁部_技法_磨消縄文_沈線', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_沈線_特徴'
]
MULTI_VALUE_SEPARATOR = ', '

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

ANALYSIS_SCENARIOS = {
    'Hokei_Makeshi_or_Habahiro': {
        'filter_name': 'Hokei_Makeshi_or_Habahiro_UNION',
        'query_list': [
            ("`口縁部_新_主モチーフ` == '区画文_方形・窓枠状'", "主モチーフ: '区画文_方形・窓枠状'"),
            ("`口縁部_技法_分類` in ['磨消縄文', '幅広の沈線']", "技法分類: '磨消縄文' or '幅広の沈線'")
        ],
        'custom_features': FEATURES_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Yano'], 
    }
}

# ==============================================================================
# 2. 関数
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

def calculate_gower_weights(data):
    return np.ones(len(data.columns)) / len(data.columns)

def get_most_similar_cluster(anchor_vec, cluster_vectors_df):
    best_cluster = None
    best_sim = -1.0
    common_cols = anchor_vec.index.intersection(cluster_vectors_df.columns)
    
    for cluster_id, row in cluster_vectors_df.iterrows():
        if cluster_id == -1: continue 
        vec_a = anchor_vec[common_cols].fillna(0).values
        vec_b = row[common_cols].fillna(0).values
        
        sim = 0 if (np.all(vec_a == 0) or np.all(vec_b == 0)) else 1.0 - cosine(vec_a, vec_b)
        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster_id
            
    return best_cluster, best_sim

def calculate_jaccard_overlap(anchor_indices, target_indices):
    """構成メンバーの完全一致率（Jaccard係数）を計算：1.0ならメンバー完全一致"""
    set_a = set(anchor_indices)
    set_t = set(target_indices)
    intersection = len(set_a.intersection(set_t))
    union = len(set_a.union(set_t))
    return intersection / union if union > 0 else 0.0

def get_focus_variables_prevalence(df_dummy_with_cluster, target_cluster_id):
    cluster_data = df_dummy_with_cluster[df_dummy_with_cluster['cluster'] == target_cluster_id]
    if cluster_data.empty: return {}
    
    # 胴部を削除し、口縁部のみに絞る
    focus_queries = {
        '口縁部_反復ナゾリ': ['口縁部', '磨消縄文_沈線', '反復ナゾリ'],
        '口縁部_内面突出': ['口縁部', '磨消縄文_沈線', '内面突出'],
    }
    
    results = {}
    for key, keywords in focus_queries.items():
        matched_cols = [c for c in cluster_data.columns if all(kw in c for kw in keywords) and c != 'cluster']
        results[key] = round(cluster_data[matched_cols[0]].mean() * 100, 2) if matched_cols else 0.0
            
    return results

# ==============================================================================
# 3. メイン処理
# ==============================================================================

def main():
    print("=== Archaeological Cluster Analysis Started ===")
    try:
        df_all = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    for scenario_name, settings in ANALYSIS_SCENARIOS.items():
        sites_str = "_".join([s.replace('\\', '') for s in settings['target_sites']])
        output_dir = f"Result_{settings['filter_name']}_{sites_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        query_parts = [f"({q})" for q, _ in settings['query_list']]
        if settings['parts_filter_mode'] == 'kouen':
            query_parts.append("`口縁部` != 'なし'")
            
        df_scenario = df_all.query(" and ".join(query_parts)).copy()
        df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(df_scenario, settings['custom_features'])
        
        regex_pattern = "|".join([f"({p})" for p in settings['target_sites']])
        df_processed['is_target'] = df_processed[SITE_COLUMN_NAME].str.contains(regex_pattern, regex=True, na=False)

        data_for_report = df_processed[final_features].copy()
        
        print("\n--- Gower距離の計算 ---")
        data_mining = df_processed[final_features].replace('不明', np.nan)
        weights = calculate_gower_weights(data_mining)
        dist_matrix = gower.gower_matrix(data_mining, weight=weights)
        
        df_all_dummy = pd.get_dummies(data_for_report)
        df_all_dummy_target = df_all_dummy[df_processed['is_target']].copy()

        # --- Phase 1: アンカー作成 ---
        print(f"\n--- Phase 1: Seed {ANCHOR_SEED} でのアンカー作成 ---")
        reducer_anchor = TSNE(n_components=2, perplexity=30, metric='precomputed', init='random', random_state=ANCHOR_SEED)
        coords_anchor = reducer_anchor.fit_transform(dist_matrix)
        
        clusterer_anchor = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
        clusters_anchor = clusterer_anchor.fit_predict(coords_anchor.astype(np.float64))
        
        df_all_dummy_target['cluster'] = clusters_anchor[df_processed['is_target'].values]
        mean_vectors = df_all_dummy_target.groupby('cluster').mean()
        
        anchor_vec_4 = mean_vectors.loc[ANCHOR_CL4_ID]
        anchor_vec_2 = mean_vectors.loc[ANCHOR_CL2_ID]
        
        # ★ サンプルのインデックス（構成員）を保存
        target_indices = df_processed[df_processed['is_target']].index
        anchor_members_4 = target_indices[clusters_anchor[df_processed['is_target'].values] == ANCHOR_CL4_ID].tolist()
        anchor_members_2 = target_indices[clusters_anchor[df_processed['is_target'].values] == ANCHOR_CL2_ID].tolist()

        # --- Phase 2: シミュレーション ---
        print(f"\n--- Phase 2: シミュレーションの実行 (Seed 1 ~ {NUM_SIMULATIONS}) ---")
        simulation_results = []
        
        for seed in range(1, NUM_SIMULATIONS + 1):
            reducer = TSNE(n_components=2, perplexity=30, metric='precomputed', init='random', random_state=seed)
            coords = reducer.fit_transform(dist_matrix)
            
            clusterer = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
            clusters = clusterer.fit_predict(coords.astype(np.float64))
            
            unique_labels = set(clusters)
            k = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if k != TARGET_K:
                continue
                
            df_all_dummy_target['cluster'] = clusters[df_processed['is_target'].values]
            sim_mean_vectors = df_all_dummy_target.groupby('cluster').mean()
            
            matched_cl4_id, sim_4 = get_most_similar_cluster(anchor_vec_4, sim_mean_vectors)
            matched_cl2_id, sim_2 = get_most_similar_cluster(anchor_vec_2, sim_mean_vectors)
            
            # ★ 構成メンバーの完全一致度を計算
            sim_members_4 = target_indices[clusters[df_processed['is_target'].values] == matched_cl4_id].tolist()
            sim_members_2 = target_indices[clusters[df_processed['is_target'].values] == matched_cl2_id].tolist()
            overlap_4 = calculate_jaccard_overlap(anchor_members_4, sim_members_4)
            overlap_2 = calculate_jaccard_overlap(anchor_members_2, sim_members_2)
            
            cl4_vars = get_focus_variables_prevalence(df_all_dummy_target, matched_cl4_id)
            cl2_vars = get_focus_variables_prevalence(df_all_dummy_target, matched_cl2_id)
            
            print(f"  [Seed {seed}] CL4相当(類似度 {sim_4:.3f}, メンバー一致 {overlap_4:.1%}) | CL2相当(類似度 {sim_2:.3f}, メンバー一致 {overlap_2:.1%})")
            
            record = {
                'Seed': seed, 'k': k,
                'Matched_CL4_ID': matched_cl4_id,
                'CL4_Profile_Similarity': round(sim_4, 4),
                'CL4_Member_Overlap(%)': round(overlap_4 * 100, 1),
                'CL4_口縁部_反復ナゾリ(%)': cl4_vars.get('口縁部_反復ナゾリ', 0),
                'CL4_口縁部_内面突出(%)': cl4_vars.get('口縁部_内面突出', 0),
                
                'Matched_CL2_ID': matched_cl2_id,
                'CL2_Profile_Similarity': round(sim_2, 4),
                'CL2_Member_Overlap(%)': round(overlap_2 * 100, 1),
                'CL2_口縁部_反復ナゾリ(%)': cl2_vars.get('口縁部_反復ナゾリ', 0),
                'CL2_口縁部_内面突出(%)': cl2_vars.get('口縁部_内面突出', 0),
            }
            simulation_results.append(record)

        if simulation_results:
            df_res = pd.DataFrame(simulation_results)
            csv_path = os.path.join(output_dir, "Simulation_Anchor_Matching_Results.csv")
            df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n-> 保存先: {csv_path}")

            # === グラフ作成処理 (合一のグループ化棒グラフに変更・白黒印刷＆サイズ最適化) ===
            cl4_nazori = df_res['CL4_口縁部_反復ナゾリ(%)'].mean()
            cl4_tosshutsu = df_res['CL4_口縁部_内面突出(%)'].mean()
            cl2_nazori = df_res['CL2_口縁部_反復ナゾリ(%)'].mean()
            cl2_tosshutsu = df_res['CL2_口縁部_内面突出(%)'].mean()

            labels = ['反復ナゾリ', '内面突出']
            cl4_values = [cl4_nazori, cl4_tosshutsu]
            cl2_values = [cl2_nazori, cl2_tosshutsu]

            x = np.arange(len(labels))  # ラベルのX座標位置
            width = 0.35                # バーの幅

            # 白黒印刷時のハッチング(網掛け)の線を少し太くして視認性を高める
            plt.rcParams['hatch.linewidth'] = 1.5

            # サイズと解像度の設定
            # 2.7インチ × 4.0インチ (dpi=200で出力時に540×800pxになる設定)
            fig, ax = plt.subplots(figsize=(2.7, 4.0))
            
            # 【白黒印刷対策】色ではなく明度差とパターン(網掛け)で区別
            # CL4(定型): 濃いグレー塗りつぶし
            ax.bar(x - width/2, cl4_values, width, label='CL4 (定型)', 
                   color='#404040', edgecolor='black')
            # CL2(非定型): 白抜き + 斜線網掛け(//)
            ax.bar(x + width/2, cl2_values, width, label='CL2 (非定型)', 
                   color='white', edgecolor='black', hatch='//')

            # 軸とラベルの設定 (指定されたフォントサイズを適用)
            ax.set_ylabel('割合 (%)', fontsize=14)
            # 横幅が狭いため、タイトルは改行を入れて視認性を確保
            ax.set_title('口縁部の\n磨消縄文の特徴変化', pad=12, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=14)
            
            # メモリの数字サイズ
            ax.tick_params(axis='y', labelsize=11)
            
            # 凡例とY軸の上限設定 (凡例がグラフの棒と被らないよう余裕を持たせる)
            ax.set_ylim(0, 115) 
            # 凡例の背景を透過させず白塗りにする(framealpha=1.0)ことで見やすくする
            ax.legend(loc='upper right', fontsize=11, framealpha=1.0)

            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "Feature_Comparison_BarChart.png")
            # dpi=200で保存 (Photoshopでの50%縮小配置に最適な高解像度)
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"-> 比較グラフ保存先: {plot_path}")

if __name__ == "__main__":
    main()