import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage

# 環境変数の読み込み
load_dotenv('.env')

def create_llm_model(use_model: str):
    """指定されたモデル名に基づいてLLMモデルのインスタンスを生成する"""
    if use_model == "ChatGPT4":
        return ChatOpenAI(model="gpt-4-turbo-2024-04-09")
    elif use_model == "ChatGPT4o":
        return ChatOpenAI(model="gpt-4o-2024-08-06")
    elif use_model == "Claude Haiku":
        return ChatAnthropic(model_name="claude-3-haiku-20240307",temperature=0.5, max_tokens=4096)
    elif use_model == "Claude Sonnet":
        return ChatAnthropic(model_name="claude-3-5-sonnet-20240620",temperature=0.5, max_tokens=4096)
    elif use_model == "Claude Opus":
        return ChatAnthropic(model_name="claude-3-opus-20240229",temperature=0.5, max_tokens=4096)
    elif use_model == "Gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro-exp-0801")
    else:
        raise ValueError("Unsupported LLM model type")

def setup_ui():
    """Streamlit UIを設定する"""
    st.title("Langchain Chat")
    st.caption("Powered by Langchain. Created in Leaders Advance 2024.")
    
    # モデル選択のプルダウンメニュー
    model_choice = st.selectbox(
        "Select the LLM model:",
        ("ChatGPT4", "ChatGPT4o", "Claude Haiku", "Claude Sonnet", "Claude Opus", "Gemini"),
        key="model_choice"
    )
    
    text_input = st.text_area("Enter your message...", key="input")  # 複数行入力に変更
    submit_button = st.button("Submit", key="submit")
    return model_choice, text_input, submit_button

def run_llm_model(chain: ConversationChain, text_input: str, model_name: str):
    """モデルを実行してチャット履歴を更新する"""
    # モデル実行
    response = chain.run(text_input)

    # セッションにチャット履歴を保存
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append((HumanMessage(content=text_input), AIMessage(content=response), model_name))

def display_chat_history():
    """チャット履歴を表示する"""
    if "chat_history" in st.session_state:
        latest_pair = st.session_state.chat_history[-1]  # 最新のメッセージペアを取得
        past_pairs = st.session_state.chat_history[:-1]  # 過去のメッセージペアを取得

        # 最新のメッセージペアをそのまま表示
        human_message, ai_message, model_name = latest_pair
        message(f"**Model: {model_name}**\n\n{ai_message.content}", is_user=False, key="latest_ai_message")
        message(human_message.content, is_user=True, key="latest_human_message")

        # 過去のメッセージペアをまとめて折りたたむ
        if past_pairs:
            with st.expander("Past Conversations"):
                for index, (human_message, ai_message, model_name) in enumerate(reversed(past_pairs)):
                    message(f"**Model: {model_name}**\n\n{ai_message.content}", is_user=False, key=f"past_ai_message_{index}")
                    message(human_message.content, is_user=True, key=f"past_human_message_{index}")

def main():
    """メイン関数"""
    # UI設定
    model_choice, text_input, submit_button = setup_ui()
    
    # メモリとチェインの初期化
    memory = st.session_state.get("memory", ConversationBufferMemory(return_messages=True))
    llm = create_llm_model(model_choice)  # 選択されたモデルを使用
    chain = ConversationChain(llm=llm, memory=memory)

    # チャット履歴の処理
    if submit_button and text_input:
        with st.spinner('AI is generating a response...'):
            run_llm_model(chain, text_input, model_choice)
            st.rerun()  # UIを更新

    # チャット履歴の表示
    display_chat_history()

if __name__ == "__main__":
    main()
