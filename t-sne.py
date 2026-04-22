import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI環境がない場合のエラー回避
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
import seaborn as sns
import plotly.express as px
import gower
import prince
import os
import re
import warnings
from scipy.stats import entropy

# 次元削減とクラスタリングのインポート
from sklearn.manifold import TSNE, MDS
import umap
import hdbscan

# 警告の抑制 (FutureWarningなど)
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義 (CONFIGURATION)
# ==============================================================================

INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'

# ★ Direct HDBSCAN の ON/OFF 設定 (True: 実行する, False: 実行しない)
ENABLE_DIRECT_HDBSCAN = False

# 頑健性確認のための乱数シードリスト
RANDOM_SEEDS = [0, 42, 123]

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
        'imputer_max_iter': 20,
        'plot_title_template': "矢野遺跡 - 口縁部方形・窓枠状({name_base}, P={p}, k={k})" 
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
        'imputer_max_iter': 20,
        'plot_title_template': "矢野遺跡 - 口縁部無文・頸部同化({name_base}, P={p}, k={k})" 
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
        'imputer_max_iter': 50,
        'plot_title_template': "矢野遺跡 - 統合({name_base}, P={p}, k={k})" 
    }
}

ANALYSIS_APPROACHES = [
    {'name': 'Gower_Weighted', 'type': 'gower', 'weight_strategy': 'entropy', 'cluster_key': 'gower_weighted'},
    {'name': 'Gower_Uniform', 'type': 'gower', 'weight_strategy': 'uniform', 'cluster_key': 'gower_uniform'},
    {'name': 'MCA_FAMD_Mode', 'type': 'mca', 'impute_strategy': 'mode', 'cluster_key': 'mca_mode'}
]

# ==============================================================================
# 2. データ前処理・加工関数
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

def impute_missing_data(data, categorical_cols, encoded_cols, strategy='mode', max_iter=20):
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
# 3. 可視化・出力関数
# ==============================================================================

def save_scree_plot(mca_model, output_dir, filename_suffix):
    try:
        if hasattr(mca_model, 'eigenvalues_'):
            explained = mca_model.eigenvalues_ / mca_model.total_inertia_
        elif hasattr(mca_model, 'explained_inertia_'):
            explained = mca_model.explained_inertia_
        else:
            return

        cumulative = np.cumsum(explained)
        n_comps = len(explained)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_comps + 1), explained, alpha=0.6, label='寄与率')
        plt.plot(range(1, n_comps + 1), cumulative, 'r-o', label='累積寄与率')
        plt.axhline(y=0.8, color='c', linestyle='--', label='80%')
        plt.axhline(y=0.9, color='b', linestyle='--', label='90%')
        plt.xlabel('主成分')
        plt.ylabel('寄与率')
        plt.title(f'MCA スクリープロット ({filename_suffix})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"Scree_{filename_suffix}.png"))
        plt.close()
    except Exception as e:
        print(f"Scree plot error: {e}")

def assign_plot_markers(df, scenario_name):
    df_plot = df.copy()
    has_doubu = ~df_plot['胴部'].isin(['なし', '不明', np.nan]) if '胴部' in df_plot.columns else False
    
    if scenario_name == 'FINAL_UNION_Hokei_vs_Mumon':
        is_hokei = df_plot['口縁部_新_主モチーフ'] == '区画文_方形・窓枠状'
        is_mumon = (df_plot['口縁部_新_主モチーフ'] == '無文') & (df_plot['頸部の傾向'] == '同化')
        
        df_plot['group'] = '不明'
        df_plot.loc[is_hokei, 'group'] = '方形'
        df_plot.loc[is_mumon, 'group'] = '無文'
        
        def get_label(row):
            g, d = row['group'], row['has_doubu']
            if g == '方形': return '方形 (胴部あり)' if d else '方形 (胴部なし)'
            if g == '無文': return '無文 (胴部あり)' if d else '無文 (胴部なし)'
            return 'その他'
            
        df_plot['has_doubu'] = has_doubu
        df_plot['marker_label'] = df_plot.apply(get_label, axis=1)
        
        symbol_map = {
            '方形 (胴部あり)': 'circle', '方形 (胴部なし)': 'circle-open',
            '無文 (胴部あり)': 'diamond', '無文 (胴部なし)': 'diamond-open',
            'その他': 'cross'
        }
        category_order = ['方形 (胴部あり)', '方形 (胴部なし)', '無文 (胴部あり)', '無文 (胴部なし)', 'その他']
        
    else:
        base_label = '方形' if 'Hokei' in scenario_name else '無文'
        symbol_closed = 'circle' if 'Hokei' in scenario_name else 'diamond'
        symbol_open = 'circle-open' if 'Hokei' in scenario_name else 'diamond-open'
        
        df_plot['has_doubu'] = has_doubu
        df_plot['marker_label'] = np.where(has_doubu, f'{base_label} (胴部あり)', f'{base_label} (胴部なし)')
        
        symbol_map = {
            f'{base_label} (胴部あり)': symbol_closed,
            f'{base_label} (胴部なし)': symbol_open
        }
        category_order = [f'{base_label} (胴部あり)', f'{base_label} (胴部なし)']

    return df_plot, symbol_map, category_order

def create_cluster_profile_excel(data_analyzed, df_result, cluster_col, output_path, cat_cols, bool_cols):
    print(f"   -> レポート作成中: {output_path}")
    
    target_mask = df_result['is_target']
    if not target_mask.any():
        return

    data_target = data_analyzed.loc[target_mask].copy()
    data_target['cluster'] = df_result.loc[target_mask, cluster_col]
    
    unique_clusters = sorted(data_target['cluster'].unique())
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for i in unique_clusters:
            cluster_data = data_target[data_target['cluster'] == i]
            if cluster_data.empty: continue
            
            # HDBSCANのノイズ(-1)への対応
            sheet_name = f'Cluster_{i}_Profile' if i != -1 else 'Cluster_Noise'
            start_row = 1
            
            cat_list = []
            for col in cat_cols:
                vc = cluster_data[col].value_counts(normalize=True).mul(100).reset_index()
                vc.columns = ['Category', 'Percentage']
                vc['Feature'] = col
                cat_list.append(vc)
            
            if cat_list:
                pd.concat(cat_list)[['Feature', 'Category', 'Percentage']].to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=start_row
                )
                
            if bool_cols:
                bool_stats = cluster_data[bool_cols].mean().mul(100).round(1)
                bool_stats = bool_stats[bool_stats > 0].sort_values(ascending=False).reset_index()
                bool_stats.columns = ['Technique', 'Prevalence(%)']
                bool_stats.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=4)

        comp = data_target['cluster'].value_counts().reset_index()
        comp.columns = ['Cluster', 'Count']
        comp['Percentage'] = (comp['Count'] / comp['Count'].sum() * 100).round(1)
        comp.to_excel(writer, sheet_name='Summary_Composition', index=False)

# ==============================================================================
# 4. メイン分析ループ
# ==============================================================================

def main():
    print("=== Archaeological Cluster Analysis Started ===")
    
    try:
        df_all = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    for scenario_name, settings in ANALYSIS_SCENARIOS.items():
        print(f"\n{'#'*60}\n# Scenario: {scenario_name}\n{'#'*60}")
        
        sites_str = "_".join([s.replace('\\', '') for s in settings['target_sites']])
        output_dir = f"Result_{settings['filter_name']}_{sites_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        query_parts = [f"({q})" for q, _ in settings['query_list']]
        if settings['parts_filter_mode'] == 'kouen':
            query_parts.append("`口縁部` != 'なし'")
            
        df_scenario = df_all.query(" and ".join(query_parts)).copy()
        print(f"Filtered Data: {len(df_scenario)} rows")
        
        if df_scenario.empty:
            print("Skipping: No data after filtering.")
            continue
            
        df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(
            df_scenario, settings['custom_features']
        )
        
        regex_pattern = "|".join([f"({p})" for p in settings['target_sites']])
        df_processed['is_target'] = df_processed[SITE_COLUMN_NAME].str.contains(regex_pattern, regex=True, na=False)

        data_for_report = df_processed[final_features].copy()
        
        for approach in ANALYSIS_APPROACHES:
            approach_name = approach['name']
            print(f"\n--- Approach: {approach_name} ---")
            
            # --- 距離計算・欠損値補完 (乱数非依存) ---
            if approach['type'] == 'gower':
                data_mining = df_processed[final_features].replace('不明', np.nan)
                weights = calculate_gower_weights(data_mining, strategy=approach['weight_strategy'])
                dist_matrix = gower.gower_matrix(data_mining, weight=weights)
                dim_reduced_data_base = dist_matrix
                # t-SNE, MDS, UMAP, HDBSCAN(Direct) への入力は距離行列
                dist_metric = 'precomputed' 
            
            elif approach['type'] == 'mca':
                data_imputed = impute_missing_data(
                    df_processed[final_features], cat_cols, bool_cols, 
                    strategy=approach['impute_strategy'], 
                    max_iter=settings.get('imputer_max_iter', 20)
                )
                # t-SNE, MDS, UMAP, HDBSCAN(Direct) への入力は生データ(ユークリッド空間)
                dist_metric = 'euclidean'

            # --- シードごとのループ ---
            for seed in RANDOM_SEEDS:
                # MCA (シード依存)
                if approach['type'] == 'mca':
                    n_comps = settings.get('mca_n_components', 10)
                    mca = prince.MCA(n_components=n_comps, n_iter=3, random_state=seed)
                    mca.fit(data_imputed)
                    save_scree_plot(mca, output_dir, f"{approach_name}_seed{seed}")
                    # numpy arrayとして取得
                    dim_reduced_data = mca.transform(data_imputed).values
                else:
                    dim_reduced_data = dim_reduced_data_base

                # ★ Direct HDBSCAN の実行制御
                if ENABLE_DIRECT_HDBSCAN:
                    print(f"  -> Running Direct HDBSCAN (seed={seed}, metric={dist_metric})")
                    
                    # エラー対策: HDBSCANの内部C拡張(Cython)がfloat64(double)を要求するため型を変換
                    dim_reduced_data = dim_reduced_data.astype(np.float64)
                    
                    clusterer_direct = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7, metric=dist_metric)
                    clusters_direct = clusterer_direct.fit_predict(dim_reduced_data)
                    
                    # 直接クラスタリングのクラスター数(-1のノイズを除外)
                    unique_labels_direct = set(clusters_direct)
                    n_clusters_direct = len(unique_labels_direct) - (1 if -1 in unique_labels_direct else 0)

                # perplexity/n_neighborsの拡張：20, 30, 40, 50
                perplexities = [p for p in [30] if p < len(df_scenario)]
                if not perplexities: perplexities = [max(1, len(df_scenario)-1)]

                # t-SNE、UMAP、NMDSの実行設定
                mapping_configs = []
                for p in perplexities:
                    mapping_configs.append({'method': 't-SNE', 'p': p})
                    mapping_configs.append({'method': 'UMAP', 'p': p})
                mapping_configs.append({'method': 'NMDS', 'p': 'N/A'})

                # --- 次元圧縮(可視化マッピング)のループ ---
                for config in mapping_configs:
                    mapping_method = config['method']
                    p = config['p']
                    
                    if approach['type'] == 'mca' and mapping_method == 'NMDS':
                        print(f"  -> Skipping: MCA + NMDS pipeline (seed={seed})")
                        continue
                    
                    if mapping_method == 't-SNE':
                        run_name = f"{approach_name}_tSNE_perp{p}_seed{seed}"
                        print(f"  -> Running Mapping: {run_name}")
                        reducer = TSNE(
                            n_components=2, perplexity=p, metric=dist_metric, 
                            init='random' if dist_metric=='precomputed' else 'pca', 
                            random_state=seed
                        )
                        coords = reducer.fit_transform(dim_reduced_data)
                        
                    elif mapping_method == 'UMAP':
                        run_name = f"{approach_name}_UMAP_nNeighbors{p}_seed{seed}"
                        print(f"  -> Running Mapping: {run_name}")
                        reducer = umap.UMAP(
                            n_neighbors=p, n_components=2, metric=dist_metric,
                            min_dist=0.1, random_state=seed
                        )
                        coords = reducer.fit_transform(dim_reduced_data)

                    elif mapping_method == 'NMDS':
                        run_name = f"{approach_name}_NMDS_seed{seed}"
                        print(f"  -> Running Mapping: {run_name}")
                        reducer = MDS(
                            n_components=2, metric=False, dissimilarity=dist_metric, 
                            random_state=seed, max_iter=300, n_init=4
                        )
                        coords = reducer.fit_transform(dim_reduced_data)

                    # --- 2D空間に対するHDBSCAN (比較用・従来処理) ---
                    # エラー対策: 念のため座標データもfloat64にしておく
                    coords = coords.astype(np.float64)
                    clusterer_2d = hdbscan.HDBSCAN(min_cluster_size=21, min_samples=7)
                    clusters_2d = clusterer_2d.fit_predict(coords)
                    
                    unique_labels_2d = set(clusters_2d)
                    n_clusters_2d = len(unique_labels_2d) - (1 if -1 in unique_labels_2d else 0)

                    # === プロット出力ループ ===
                    clustering_results = []
                    
                    # Direct が ON の場合のみリストに追加
                    if ENABLE_DIRECT_HDBSCAN:
                        clustering_results.append({
                            'type': 'Direct_HDBSCAN',
                            'clusters': clusters_direct,
                            'n_clusters': n_clusters_direct,
                            'title_suffix': 'Direct'
                        })
                        
                    # 2D は常にリストに追加
                    clustering_results.append({
                        'type': '2D_HDBSCAN',
                        'clusters': clusters_2d,
                        'n_clusters': n_clusters_2d,
                        'title_suffix': '2D'
                    })

                    for result_info in clustering_results:
                        c_type = result_info['type']
                        
                        df_result = df_scenario[['名称', '場所']].copy()
                        if '口縁部_新_主モチーフ' in df_scenario.columns:
                            df_result = pd.concat([df_result, df_scenario[['口縁部_新_主モチーフ', '頸部の傾向', '胴部']]], axis=1)
                        
                        df_result['X'] = coords[:, 0]
                        df_result['Y'] = coords[:, 1]
                        df_result['cluster'] = result_info['clusters']
                        df_result['is_target'] = df_processed['is_target'].values
                        
                        df_plot, symbol_map, category_orders = assign_plot_markers(df_result, scenario_name)
                        
                        title = settings['plot_title_template'].format(
                            name_base=f"{approach_name}({mapping_method}), Seed={seed}", 
                            p=p, k=f"{result_info['n_clusters']}({result_info['title_suffix']})"
                        )
                        
                        fig = px.scatter(
                            df_plot, x='X', y='Y',
                            color=df_plot['cluster'].astype(str),
                            symbol='marker_label', symbol_map=symbol_map,
                            category_orders={'marker_label': category_orders},
                            hover_data=['名称', '場所'],
                            title=title,
                            opacity=0.7
                        )
                        
                        fig.for_each_trace(
                            lambda t: t.update(marker=dict(opacity=0.3, size=5)) 
                            if not any(target in t.name for target in category_orders) else None
                        )
                        
                        html_path = os.path.join(output_dir, f"Plot_{run_name}_{c_type}.html")
                        fig.write_html(html_path)
                        
                        excel_path = os.path.join(output_dir, f"Profile_{run_name}_{c_type}.xlsx")
                        create_cluster_profile_excel(
                            data_for_report, df_result, 'cluster', excel_path, cat_cols, bool_cols
                        )

    print("\n=== All Analysis Completed ===")

if __name__ == "__main__":
    main()