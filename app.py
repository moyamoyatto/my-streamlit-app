import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# .envファイルをロードして環境変数を設定
load_dotenv()

# .envファイルはapp.pyと同じディレクトリに配置し、以下のように環境変数を定義する
# export GEMINI_API_KEY="取得したAPIキーの文字列"

# APIキーを環境変数から取得
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。Google CloudのAPIキーを設定してください。")
    st.stop()

genai.configure(api_key=api_key)

# Geminiモデルを取得する関数を実装してください。
@st.cache_resource       # モデルの読み込みを1度だけ行い、同じモデルを使い回す
def get_gemini_model():
    return genai.GenerativeModel("gemini-2.5-flash")


# CSVファイルを読み込む関数を実装してください。
@st.cache_data           # 読み込んだデータを保存し、使い回す
def load_data(file_name):
    csv_file_path = os.path.join(os.path.dirname(__file__), file_name)
    return pd.read_csv(csv_file_path)


# TF-IDFモデルを構築する関数を実装してください。
@st.cache_resource
def build_tfidf_model(texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

# SentenceTransformerの埋め込みモデルを取得する関数を実装してください。
@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

# テキストデータをベクトル化する関数を実装してください。
# Embedding ・・・単語や文章の意味をベクトル化
@st.cache_resource
def build_embedding_model(texts):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings

# ハイブリッド検索を行う関数を実装してください。
def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings):
    # TF-IDFモデルを使用して類似度を算出
    query_tfidf = tfidf_vectorizer.transform([query])   # 質問文のベクトルを計算（単語ごとの重要度をベクトル化）
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]   # ユーザー質問(query)とデータベース(yahoo記事csvファイル)を比較し、類似度を算出

    # Embeddingによりベクトル化した数値から類似度を算出
    embedding_model = get_embedding_model()   # Embeddingモデルを読み込み
    query_embeddings = embedding_model.encode([query], convert_to_numpy=True)   # ユーザー質問チャットをEmbeddingモデルを使ってベクトル化（文章全体の意味をベクトル化）
    embedding_scores = cosine_similarity(query_embeddings, embeddings)[0]   # ユーザー質問(query)とデータベース(yahoo記事csvファイル)を比較し、類似度を算出

    # ハイブリットスコア
    hybrid_scores = (0.5 * tfidf_scores + 0.5 * embedding_scores)   # TF-IDF類似度、Embedding類似度それぞれ5割ずつ

    return np.argsort(hybrid_scores)[::-1]   # 類似度の高い順に並べ替え（argsort: 類似度と記事の紐付けを保ったままソート）


# チャット履歴を初期化する関数を実装してください。
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# チャット履歴を表示する関数を実装してください。
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Geminiモデルを使って応答を生成する関数を実装してください。
def respond_with_gemini(query, results, texts, top_n=3):
    model = get_gemini_model()

    context = "\n\n".join([
        f'記事{i+1}\n{texts[idx]}'
        for i, idx in enumerate(results[:top_n])
    ])

    # Geminiに投げる質問文
    prompt_to_LLM = f"""
    あなたはニュース解説アシスタントです。
    以下のユーザーの質問に対して、【参考記事】の内容を根拠にして回答してください。

    【質問】
    {query}

    【参考記事】
    {context}

    【回答】
    """

    response = model.generate_content(prompt_to_LLM)

    return response.text


# Streamlitアプリのメイン
st.title("RAG System")

# 必要なデータをロードし、処理するコードを実装してください。
file_name = "yahoo_news_articles_preprocessed.csv"
df = load_data(file_name)   # ←これがデータベースとなる

# texts = []  # 適切なデータを抽出してリストに変換してください。
texts = df['text_mod'].fillna('').tolist()                # テキストデータを正規化＆表記ゆれ統一したデータ　→回答生成用に使う
search_texts = df['text_tokenized'].fillna('').tolist()   # テキストデータを形態素解析までしたデータ　→検索用に使う

# テキストデータをベクトルデータに変換
# tfidf_matrix, tfidf_vectorizer = None, None  # TF-IDFモデルを構築してください。
# embeddings = None  # 埋め込みモデルを構築してください。
tfidf_vectorizer, tfidf_matrix = build_tfidf_model(search_texts)
embeddings = build_embedding_model(search_texts)


# チャット履歴を初期化
init_chat_history()

# チャット履歴を表示
display_chat_history()

# チャットボット
if prompt := st.chat_input("質問を入力してください"):   # ユーザーquery入力欄を定義
    # ユーザー(role: user)質問チャット表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # ユーザー(role: user)質問チャット履歴保存
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ハイブリット検索
    results = hybrid_search(prompt, tfidf_matrix, tfidf_vectorizer, embeddings)

    # Gemini回答生成
    answer = respond_with_gemini(prompt, results, texts)

    # 回答(role: assistant)チャット表示
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # 回答(role: assistant)の履歴保存
    st.session_state.messages.append({"role": "assistant", "content": answer})
