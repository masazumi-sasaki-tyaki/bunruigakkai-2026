import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import gower
import prince
import os
import warnings
from scipy.stats import entropy
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義 (本番コードと同一)
# ==============================================================================
INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'

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
FEATURES_MUMON_MAKESHI = [
    '口縁部_形状', '口縁部_直下', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文', '胴部_新_主モチーフ'
]
FEATURES_MUMON_HABAHIRO = ['口縁部_形状','口縁部_直下','胴部_技法_幅広の沈線']

FEATURES_MUMON_UNION = list(set(FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))
FEATURES_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO))
FEATURES_FINAL_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO + FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))

ANALYSIS_SCENARIOS = {
    'Hokei_Makeshi_or_Habahiro': {
        'query_list': [
            ("`口縁部_新_主モチーフ` == '区画文_方形・窓枠状'", ""),
            ("`口縁部_技法_分類` in ['磨消縄文', '幅広の沈線']", "")
        ],
        'custom_features': FEATURES_UNION, 
        'parts_filter_mode': 'kouen',
        'mca_n_components': 33,
    },
    'Mumon_Makeshi_or_Habahiro': {
        'query_list': [
            ("`口縁部_新_主モチーフ` == '無文'", ""),
            ("`胴部_技法_分類` in ['磨消縄文', '幅広の沈線']", ""),
            ("`頸部の傾向` == '同化'", "")
        ],
        'custom_features': FEATURES_MUMON_UNION, 
        'parts_filter_mode': 'kouen',
        'mca_n_components': 20,
    },
    'FINAL_UNION_Hokei_vs_Mumon': {
        'query_list': [
            ("(`口縁部_新_主モチーフ` == '区画文_方形・窓枠状') or (`口縁部_新_主モチーフ` == '無文' and `頸部の傾向` == '同化')", "")
        ],
        'custom_features': FEATURES_FINAL_UNION, 
        'parts_filter_mode': 'kouen',
        'mca_n_components': 40,
    }
}

# ==============================================================================
# 2. 前処理関数 (一部抜粋)
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
    return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(data.columns)) / len(data.columns)

def impute_missing_data(data, categorical_cols, encoded_cols):
    data_imputed = data.copy()
    data_imputed.replace('不明', np.nan, inplace=True)
    for col in data_imputed.columns:
        mode_val = data_imputed[col].mode()
        if not mode_val.empty:
            data_imputed[col] = data_imputed[col].fillna(mode_val[0])
        else:
            fill_val = 'N/A' if col in categorical_cols else 0
            data_imputed[col] = data_imputed[col].fillna(fill_val)
    return data_imputed

# ==============================================================================
# 3. 診断と可視化処理
# ==============================================================================
def plot_distance_distribution(distances_dict, scenario_name, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'ペアワイズ距離分布の比較 (次元の呪い検証): {scenario_name}', fontsize=16)

    for ax, (title, distances) in zip(axes, distances_dict.items()):
        sns.histplot(distances, bins=50, kde=True, color='blue', alpha=0.6, ax=ax)
        
        # 統計量の計算
        d_max, d_min, d_mean, d_std = distances.max(), distances.min(), distances.mean(), distances.std()
        # 相対コントラスト (最大-最小) / 平均：次元の呪いではこれが0に近づく
        contrast = (d_max - d_min) / d_mean if d_mean > 0 else 0
        
        stat_text = (
            f"最大: {d_max:.4f}\n"
            f"最小: {d_min:.4f}\n"
            f"平均: {d_mean:.4f}\n"
            f"標準偏差: {d_std:.4f}\n"
            f"コントラスト: {contrast:.4f}"
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # テキストボックスで統計量をグラフ内に表示
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'Distance_Distribution_{scenario_name}.png'))
    plt.close()

def main():
    print("=== Dimensionality Curse Check Started ===")
    output_dir = "Result_Diagnostic_Distances"
    os.makedirs(output_dir, exist_ok=True)
    
    df_all = pd.read_csv(INPUT_FILE)

    for scenario_name, settings in ANALYSIS_SCENARIOS.items():
        print(f"\n--- Checking Scenario: {scenario_name} ---")
        
        query_parts = [f"({q})" for q, _ in settings['query_list']]
        if settings['parts_filter_mode'] == 'kouen':
            query_parts.append("`口縁部` != 'なし'")
            
        df_scenario = df_all.query(" and ".join(query_parts)).copy()
        if df_scenario.empty:
            continue
            
        df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(
            df_scenario, settings['custom_features']
        )
        
        data_mining = df_processed[final_features].replace('不明', np.nan)
        
        distances_dict = {}
        
        # 1. Gower_Uniform の距離
        weights_uni = calculate_gower_weights(data_mining, strategy='uniform')
        dist_mat_uni = gower.gower_matrix(data_mining, weight=weights_uni)
        # 上三角行列からペアワイズ距離を抽出（自分自身の0距離や重複を排除）
        dist_uni = dist_mat_uni[np.triu_indices_from(dist_mat_uni, k=1)]
        distances_dict['Gower (Uniform Weight)'] = dist_uni

        # 2. Gower_Weighted (エントロピー) の距離
        weights_ent = calculate_gower_weights(data_mining, strategy='entropy')
        dist_mat_ent = gower.gower_matrix(data_mining, weight=weights_ent)
        dist_ent = dist_mat_ent[np.triu_indices_from(dist_mat_ent, k=1)]
        distances_dict['Gower (Entropy Weight)'] = dist_ent

        # 3. MCA空間でのユークリッド距離
        data_imputed = impute_missing_data(data_mining, cat_cols, bool_cols)
        mca = prince.MCA(n_components=settings['mca_n_components'], n_iter=3, random_state=42)
        dim_reduced_mca = mca.fit_transform(data_imputed).values
        # pdistは自動的にペアワイズ距離の1次元配列を返す
        dist_mca = pdist(dim_reduced_mca, metric='euclidean')
        distances_dict['MCA (Euclidean)'] = dist_mca

        # プロットして保存
        plot_distance_distribution(distances_dict, scenario_name, output_dir)
        
        # コンソールにもサマリーを出力
        for name, dists in distances_dict.items():
            contrast = (dists.max() - dists.min()) / dists.mean()
            print(f"[{name}] コントラスト((Max-Min)/Mean): {contrast:.4f}  (Std: {dists.std():.4f})")

    print(f"\nDone! Please check the graphs in '{output_dir}' directory.")

if __name__ == "__main__":
    main()