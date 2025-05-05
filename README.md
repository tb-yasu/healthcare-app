# LLM-based helthcare-app

# ヘルスケアアプリケーション

このアプリケーションは日々の健康管理を目的として、体組成データ、食事記録、筋トレ記録を管理・分析するためのStreamlitベースのヘルスケアアプリです。

GPT-4oモデルを利用して、ユーザーの入力したデータに基づき食事評価やアドバイス、総合的な健康評価を提供します。

## 機能

* **ユーザー情報管理**

  * 年齢、性別、身長の管理

* **食事記録**

  * テキストまたは画像を用いて食事内容を記録
  * GPT-4oを用いた栄養情報の自動推定

* **体組成記録**

  * 体重、体脂肪率、筋肉量、骨量、体水分率を記録

* **筋トレ記録**

  * 種目、重量、セット数、レップ数、運動時間を記録

* **データ表示・可視化**

  * 記録されたデータをグラフや表で表示
  * 栄養素ごとの摂取状況、体組成の推移を確認可能

* **評価とアドバイス**

  * ハリス・ベネディクト方程式を利用した基礎代謝計算
  * GPT-4oを利用した食事、筋トレ、体組成の個別評価
  * 総合評価による具体的な改善提案

* **LLM相談機能**

  * 最近1ヶ月の食事、体組成、筋トレデータを踏まえた健康相談が可能

## 使用技術

* Streamlit
* Python
* SQLAlchemy (SQLite)
* GPT-4o API (OpenAI)
* matplotlib, pandas

## インストール方法

1. リポジトリをクローンします。

```bash
git clone https://github.com/<your_username>/healthcare-app.git
```

2. 必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

3. Streamlitアプリを実行します。

```bash
streamlit run app.py
```

## 環境変数

OpenAI APIキーをStreamlitのシークレットに設定します。

`.streamlit/secrets.toml`

```toml
OPENAI_API_KEY="your_openai_api_key"
```

## ライセンス

本プロジェクトは[GNU General Public License v3.0](LICENSE)のもとで公開されています。
