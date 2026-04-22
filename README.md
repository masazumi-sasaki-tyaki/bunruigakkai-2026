欠測を伴う混合データからの潜在的構造の抽出

Gower距離を用いた縄文土器破片の教師なし分類

本リポジトリは、欠測を含む質的データ（混合データ）から潜在的構造を客観的に抽出するための探索的データ分析（EDA）パイプラインのコードを公開しています。
本研究では、縄文土器破片のデータを対象に、欠測を許容する Gower 距離と、局所的類似性を強調する非線形次元圧縮（t-SNE等）、および密度ベースのクラスタリング（HDBSCAN）を組み合わせた分析を行っています。

分析の概要

対象データ: 縄文土器破片 計1418点

変数: 56項目の属性変数（名義変数、二値変数、順序変数が混在）

主要な手法:

距離計算: Gower 距離 (gower)

次元圧縮: t-SNE, UMAP (scikit-learn, umap-learn)

クラスタリング: HDBSCAN (hdbscan)

評価指標: ARI, NMI 等

リポジトリのファイル構成

本リポジトリに含まれる主なスクリプトの構成と役割は以下の通りです。

.
├── requirements.txt          # 実行環境構築用のパッケージ一覧
│
├── 前処理 (Data Cleaning)
│   ├── cleaning1.py          # データクレンジング処理 1
│   ├── cleaning2.py          # データクレンジング処理 2
│   └── cleaning3.py          # データクレンジング処理 3
│
├── 次元圧縮・クラスタリング (Dimensionality Reduction & Clustering)
│   ├── t-sne.py              # MCA, NMDS, Gower距離の計算やt-SNE, UMAPによる次元圧縮と可視化の実行
│   └── t-SNE-shirokuro.py    # 論文用図版（白黒）のt-SNEプロット出力
│
├── 評価・比較 (Evaluation & Comparison)
│   ├── jigen.py              # クラスターの構造確認
│   ├── k-check.py            # クラスタ数(k)のシミュレーションと安定性評価
│   ├── ARI.py                # ARI (Adjusted Rand Index) 等による再現性・妥当性評価
│   ├── Ankercompare.py       # クラスターごとの属性比較・型式変遷の分析
│   └── Ankercompare-shirokuro.py # 比較結果の論文用図版（白黒）出力


環境構築

このプロジェクトを実行するには、Python環境が必要です。以下の手順で必要なパッケージをインストールしてください。

リポジトリのクローン

git clone [https://github.com/masazumi-sasaki-tyaki/bunruigakkai-2026.git](https://github.com/masazumi-sasaki-tyaki/bunruigakkai-2026.git)
cd bunruigakkai-2026


仮想環境の作成と有効化 (推奨)

Windows の場合:

python -m venv .venv
.venv\Scripts\activate


macOS / Linux の場合:

python -m venv .venv
source .venv/bin/activate


依存パッケージのインストール

pip install -r requirements.txt


実行順序（推奨）

分析を再現する場合は、おおむね以下の順序でスクリプトを実行してください。
(※データファイルの配置場所や入出力の設定は、各スクリプト内のパス設定をご確認ください)

データの前処理
cleaning1.py → cleaning2.py → cleaning3.py の順に実行し、分析用データを整備します。

次元圧縮とクラスタリング
Gower距離の計算とt-SNEによるマッピング、HDBSCANによるクラスタリングを行います。

評価と図版作成
jigen.py, k-check.py や ARI.py で結果の妥当性を評価します。
論文用の白黒図版が必要な場合は、t-SNE-shirokuro.py や Ankercompare-shirokuro.py を実行してください。

著者

佐々木 真純
早稲田大学 文学学術院 文学研究科 考古学専攻

連絡先

E-mail: tukushimap@suou.waseda.jp
