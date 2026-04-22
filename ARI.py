import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI環境がない場合のエラー回避
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
import seaborn as sns
import gower
import prince
import os
import warnings
import itertools # ★追加: 総当たり組み合わせ用
from scipy.stats import entropy
from tqdm import tqdm  # 進捗バー用

# 次元削減とクラスタリングのインポート
from sklearn.manifold import TSNE
import umap
import hdbscan
# ★追加: クラスタリング評価指標
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 警告の抑制
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義 (CONFIGURATION)
# ==============================================================================

INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'

# シミュレーション設定
# ★テスト時は 5 や 10 に減らして実行時間を確認することをおすすめします
N_ITERATIONS = 100        # シードを変えて実行する回数
TARGET_PERPLEXITY = 30    # t-SNEのperplexity および UMAPのn_neighbors

# 複数選択項目（ワンホット展開用）
MULTI_VALUE_FEATURES = [
    '口縁部_技法_沈線_特徴', '口縁部_技法_磨消縄文_沈線', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_沈線_特徴'
]
MULTI_VALUE_SEPARATOR = ', '

# 特徴量セットの定義
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
FEATURES_MUMON_MAKESHI = [
    '口縁部_形状', '口縁部_直下', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文', '胴部_新_主モチーフ'
]
FEATURES_MUMON_HABAHIRO = ['口縁部_形状','口縁部_直下','胴部_技法_幅広の沈線']

# 統合特徴量リスト
FEATURES_MUMON_UNION = list(set(FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))
FEATURES_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO))
FEATURES_FINAL_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO + FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))

# 分析シナリオ定義
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
        'mca_n_components': 33,
    },
    'Mumon_Makeshi_or_Habahiro': {
        'filter_name': 'Mumon_Makeshi_or_Habahiro_Doka_UNION',
        'query_list': [
            ("`口縁部_新_主モチーフ` == '無文'", "主モチーフ: '無文'"),
            ("`胴部_技法_分類` in ['磨消縄文', '幅広の沈線']", "技法分類: '磨消縄文' or '幅広の沈線'"),
            ("`頸部の傾向` == '同化'", "頸部の傾向: '同化'")
        ],
        'custom_features': FEATURES_MUMON_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Yano'], 
        'mca_n_components': 20,
    },
    'FINAL_UNION_Hokei_vs_Mumon': {
        'filter_name': 'FINAL_UNION_Hokei_vs_Mumon',
        'query_list': [
            (
                "(`口縁部_新_主モチーフ` == '区画文_方形・窓枠状') or "
                "(`口縁部_新_主モチーフ` == '無文' and `頸部の傾向` == '同化')",
                "最終統合: ('方形・窓枠状') または ('無文' かつ '頸部同化')"
            )
        ],
        'custom_features': FEATURES_FINAL_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Yano'], 
        'mca_n_components': 40,
    }
}

ANALYSIS_APPROACHES = [
    {'name': 'Gower_Weighted', 'type': 'gower', 'weight_strategy': 'entropy'},
    {'name': 'Gower_Uniform', 'type': 'gower', 'weight_strategy': 'uniform'},
    {'name': 'MCA_FAMD_Mode', 'type': 'mca', 'impute_strategy': 'mode'}
]

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

def calculate_gower_weights(data, strategy='uniform'):
    if strategy == 'uniform':
        return np.ones(len(data.columns)) / len(data.columns)
    
    weights = []
    for col in data.columns:
        counts = data[col].value_counts(normalize=True, dropna=True)
        if len(counts) > 1:
            weights.append(entropy(counts.values))
        else:
            weights.append(0)
            
    weights = np.array(weights)
    if np.sum(weights) > 0:
        return weights / np.sum(weights)
    else:
        return np.ones(len(data.columns)) / len(data.columns)

def impute_missing_data(data, categorical_cols, encoded_cols, strategy='mode'):
    data_imputed = data.copy()
    data_imputed.replace('不明', np.nan, inplace=True)

    if strategy == 'mode':
        for col in data_imputed.columns:
            mode_val = data_imputed[col].mode()
            if not mode_val.empty:
                data_imputed[col] = data_imputed[col].fillna(mode_val[0])
            else:
                fill_val = 'N/A' if col in categorical_cols else 0
                data_imputed[col] = data_imputed[col].fillna(fill_val)
        return data_imputed

# ==============================================================================
# 3. 分布シミュレーション・安定性評価と可視化
# ==============================================================================

def run_k_distribution_simulation(dim_reduced_data, dist_metric, n_samples, scenario_name, approach_name, output_dir):
    print(f"    -> Running Simulation: {n_samples} samples, {N_ITERATIONS} iterations")
    
    p = min(TARGET_PERPLEXITY, n_samples - 1)
    if p < 1: p = 1
    
    k_results = []
    
    # ★追加: 全イテレーションのクラスタラベルを保存するリスト
    all_labels_tsne = []
    all_labels_umap = []
    
    # 進行状況バーを表示してシミュレーション実行
    for seed in tqdm(range(N_ITERATIONS), desc=f"{approach_name} Simulation", leave=False):
        
        # --- t-SNE ---
        tsne = TSNE(
            n_components=2, perplexity=p, metric=dist_metric, 
            init='random', random_state=seed, n_jobs=-1
        )
        coords_tsne = tsne.fit_transform(dim_reduced_data).astype(np.float64)
        
        clusterer_tsne = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
        labels_tsne = clusterer_tsne.fit_predict(coords_tsne)
        k_tsne = len(set(labels_tsne)) - (1 if -1 in labels_tsne else 0)
        
        k_results.append({'Method': 't-SNE', 'k': k_tsne, 'Seed': seed})
        all_labels_tsne.append(labels_tsne) # ラベルを保存

        # --- UMAP ---
        umap_model = umap.UMAP(
            n_neighbors=p, n_components=2, metric=dist_metric,
            min_dist=0.1, init='random', random_state=seed, n_jobs=-1
        )
        coords_umap = umap_model.fit_transform(dim_reduced_data).astype(np.float64)
        
        clusterer_umap = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
        labels_umap = clusterer_umap.fit_predict(coords_umap)
        k_umap = len(set(labels_umap)) - (1 if -1 in labels_umap else 0)
        
        k_results.append({'Method': 'UMAP', 'k': k_umap, 'Seed': seed})
        all_labels_umap.append(labels_umap) # ラベルを保存

    # ----------------------------------------------------------------------
    # ★追加: 総当たりでの安定性評価 (Pairwise ARI & NMI Calculation)
    # ----------------------------------------------------------------------
    pairwise_results = []
    
    # N回の中から2つを選ぶ全組み合わせのインデックスペアを生成
    pair_indices = list(itertools.combinations(range(N_ITERATIONS), 2))
    
    for i, j in pair_indices:
        # t-SNEペアのスコア計算
        ari_tsne = adjusted_rand_score(all_labels_tsne[i], all_labels_tsne[j])
        nmi_tsne = normalized_mutual_info_score(all_labels_tsne[i], all_labels_tsne[j])
        pairwise_results.append({
            'Method': 't-SNE', 'Pair': f"{i}_vs_{j}", 'ARI': ari_tsne, 'NMI': nmi_tsne
        })
        
        # UMAPペアのスコア計算
        ari_umap = adjusted_rand_score(all_labels_umap[i], all_labels_umap[j])
        nmi_umap = normalized_mutual_info_score(all_labels_umap[i], all_labels_umap[j])
        pairwise_results.append({
            'Method': 'UMAP', 'Pair': f"{i}_vs_{j}", 'ARI': ari_umap, 'NMI': nmi_umap
        })

    df_k = pd.DataFrame(k_results)
    df_stability = pd.DataFrame(pairwise_results)
    
    # ----------------------------------------------------------------------
    # データの保存と可視化
    # ----------------------------------------------------------------------
    
    # 1. Kの分布の保存とプロット (既存)
    csv_path_k = os.path.join(output_dir, f"K_Distribution_{scenario_name}_{approach_name}.csv")
    df_k.to_csv(csv_path_k, index=False)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_k, x='k', hue='Method', palette='Set2', alpha=0.8)
    plt.title(f'HDBSCAN クラスタ数(k)の分布\n{scenario_name} / {approach_name}\n(iterations={N_ITERATIONS})')
    plt.xlabel('クラスタ数 (k)')
    plt.ylabel('頻度')
    plt.legend(title='次元削減手法')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"K_Distribution_{scenario_name}_{approach_name}.png"), bbox_inches='tight')
    plt.close()

    # 2. ★追加: 安定性スコア(ARI/NMI)の保存とプロット
    csv_path_stab = os.path.join(output_dir, f"Stability_Scores_{scenario_name}_{approach_name}.csv")
    df_stability.to_csv(csv_path_stab, index=False)
    
    # ARIとNMIの箱ひげ図を作成して比較
    # 縦長の形式（melt）に変換してSeabornで描きやすくする
    df_melted = df_stability.melt(id_vars=['Method', 'Pair'], value_vars=['ARI', 'NMI'], 
                                  var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Metric', y='Score', hue='Method', palette='Set2')
    plt.title(f'クラスタリングの内部構造の安定性 (Pairwise Scores)\n{scenario_name} / {approach_name}')
    plt.xlabel('評価指標 (1.0に近いほど乱数によらず安定)')
    plt.ylabel('スコア')
    plt.ylim(0, 1.05) # スコアは最大1.0なので固定
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"Stability_Distribution_{scenario_name}_{approach_name}.png"), bbox_inches='tight')
    plt.close()
    
    # コンソールサマリーの出力
    print(f"\n    -> [Summary: K-Distribution] {approach_name}")
    summary_k = df_k.groupby('Method')['k'].value_counts().sort_index().unstack(fill_value=0)
    print(summary_k)
    
    print(f"    -> [Summary: Stability (Mean ± Std)]")
    for method in ['t-SNE', 'UMAP']:
        subset = df_stability[df_stability['Method'] == method]
        mean_ari, std_ari = subset['ARI'].mean(), subset['ARI'].std()
        mean_nmi, std_nmi = subset['NMI'].mean(), subset['NMI'].std()
        print(f"       {method:5s} | ARI: {mean_ari:.3f} (±{std_ari:.3f}) | NMI: {mean_nmi:.3f} (±{std_nmi:.3f})")

# ==============================================================================
# 4. メインルーチン
# ==============================================================================

def main():
    print(f"=== K-Distribution & Stability Simulation Started ({N_ITERATIONS} iterations) ===")
    
    try:
        df_all = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    for scenario_name, settings in ANALYSIS_SCENARIOS.items():
        print(f"\n{'='*60}\n# Scenario: {scenario_name}\n{'='*60}")
        
        output_dir = f"Simulation_K_{settings['filter_name']}"
        os.makedirs(output_dir, exist_ok=True)
        
        # データのフィルタリング
        query_parts = [f"({q})" for q, _ in settings['query_list']]
        if settings['parts_filter_mode'] == 'kouen':
            query_parts.append("`口縁部` != 'なし'")
            
        df_scenario = df_all.query(" and ".join(query_parts)).copy()
        n_samples = len(df_scenario)
        print(f"Filtered Data: {n_samples} rows")
        
        if df_scenario.empty:
            print("Skipping: No data after filtering.")
            continue
            
        # 前処理
        df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(
            df_scenario, settings['custom_features']
        )
        
        for approach in ANALYSIS_APPROACHES:
            approach_name = approach['name']
            print(f"\n  --- Approach: {approach_name} ---")
            
            # --- 距離計算・欠損値補完 ---
            if approach['type'] == 'gower':
                data_mining = df_processed[final_features].replace('不明', np.nan)
                weights = calculate_gower_weights(data_mining, strategy=approach['weight_strategy'])
                dist_matrix = gower.gower_matrix(data_mining, weight=weights)
                
                base_data = dist_matrix
                dist_metric = 'precomputed' 
                
            elif approach['type'] == 'mca':
                data_imputed = impute_missing_data(
                    df_processed[final_features], cat_cols, bool_cols, 
                    strategy=approach['impute_strategy']
                )
                
                n_comps = settings.get('mca_n_components', 10)
                mca = prince.MCA(n_components=n_comps, n_iter=3, random_state=42)
                
                base_data = mca.fit_transform(data_imputed).values
                dist_metric = 'euclidean'

            # シミュレーションと分布の出力
            run_k_distribution_simulation(
                dim_reduced_data=base_data, 
                dist_metric=dist_metric, 
                n_samples=n_samples,
                scenario_name=scenario_name, 
                approach_name=approach_name, 
                output_dir=output_dir
            )

    print("\n=== All Simulations Completed ===")

if __name__ == "__main__":
    main()