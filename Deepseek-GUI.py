import streamlit as st
from openai import OpenAI
import os
import json
import re
from datetime import datetime
import base64

# Create client object
client = OpenAI(
    base_url='https://api.siliconflow.cn/v1/', #Just example, use the service platform as your will
    api_key='...' #put your api-key here
)

HISTORY_DIR = "ChatHistory" #Default path, change as you want
os.makedirs(HISTORY_DIR, exist_ok=True)
name_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
vlm_model = "deepseek-ai/deepseek-vl2"

def init_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_convo' not in st.session_state:
        st.session_state.current_convo = None
    if 'convo_list' not in st.session_state:
        st.session_state.convo_list = []
    # parameter initialization
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "deepseek-ai/DeepSeek-R1"
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 2048
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 1.0
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 1.0

    # Loading Chat History
    for msg in st.session_state.get('messages', []):
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item["type"] == "image_url":
                    item["image_url"]["url"] = str(item["image_url"]["url"])


def generate_filename(content):

    text_content = ""
    if isinstance(content, list):
        texts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
        text_content = " ".join(texts)
    else:
        text_content = str(content)

    clean_content = ""
    response = client.chat.completions.create(
        model=name_model,
        messages=[
            {"role": "system", "content": "You are a dialog naming assistant that helps extract dialog keywords as dialog record file names, within fifteen words."},
            {"role": "user", "content": "Extract the theme of the dialog(only the words of theme themselves):" + text_content}],
        stream=True,
        max_tokens=1024
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            clean_content += chunk.choices[0].delta.content

    clean_content = re.sub(r'[\n\r\t\\/*?:"<>|]', "", clean_content)[:15]
    timestamp = datetime.now().strftime("%m%d%H%M")
    return f"{clean_content}_{timestamp}.json" if clean_content else f"Untitled_{timestamp}.json"


def convert_messages_for_api(messages, use_vlm):
    # Convert messages for calling VLM
    converted = []
    for msg in messages:
        if use_vlm:
            if isinstance(msg["content"], list):
                content = [{
                    "type": item["type"],
                    "image_url": {"url": item["image_url"]["url"]} if item["type"] == "image_url" else None,
                    "text": item.get("text", "")} for item in msg["content"]
                ]
            else:
                content = [{"type": "text", "text": str(msg["content"])}]
        else:
            if isinstance(msg["content"], list):
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                content = " ".join(text_parts)
            else:
                content = msg["content"]
        converted.append({"role": msg["role"], "content": content})
    return converted

def refresh_convo_list():
    st.session_state.convo_list = [
        f for f in os.listdir(HISTORY_DIR)
        if f.endswith('.json') and os.path.getsize(os.path.join(HISTORY_DIR, f)) > 0
    ]

def new_conversation():
    st.session_state.messages = []
    st.session_state.current_convo = None

def load_conversation(filename):
    path = os.path.join(HISTORY_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        st.session_state.messages = json.load(f)
    st.session_state.current_convo = filename

def save_conversation():
    if st.session_state.current_convo and st.session_state.messages:
        path = os.path.join(HISTORY_DIR, st.session_state.current_convo)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

init_session()

#Defining Sidebar
with st.sidebar:
    st.title("Dialogs")

    # 新增模型设置区域
    st.subheader("Model Settings")
    st.session_state.selected_model = st.selectbox(
        "Dialog Model",
        ["deepseek-ai/DeepSeek-R1",
         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
         "Pro/deepseek-ai/DeepSeek-R1"],
        index=0
    )

    # Supported parameters
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.max_tokens = st.slider(
            "Max_tokens",
            1024, 16384, 8192,
            help="Limiting the max length of response"
        )
    with col2:
        st.session_state.max_tokens = st.number_input(
            "Input value",
            min_value=1024,
            max_value=16384,
            value=8192,
            step=128,
            key="max_tokens"
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.6, 0.1,
            help="More random as temperature rises"
        )
    with col2:
        st.session_state.temperature = st.number_input(
            "Input value",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.1,
            key="temp_input",
            format="%.1f"
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.top_p = st.slider(
            "Top_p",
            0.0, 1.0, 0.95, 0.01,
            help="Controlling the content diversity"
        )
    with col2:
        st.session_state.top_p = st.number_input(
            "Input value",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            key="top_p_input",
            format="%.2f"
        )

    if st.button("➕ Create New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_convo = None
        st.rerun()

    st.subheader("History Conversation")
    refresh_convo_list()
    for convo in st.session_state.convo_list:
        cols = st.columns([3, 1])
        with cols[0]:
            if st.button(convo[:-5], key=f"btn_{convo}", use_container_width=True):
                load_conversation(convo)
                st.rerun()
        with cols[1]:
            if st.button("×", key=f"del_{convo}", type='primary'):
                os.remove(os.path.join(HISTORY_DIR, convo))
                if st.session_state.current_convo == convo:
                    new_conversation()
                st.rerun()

# 主界面布局
st.title("AI Agent Helper")

# 显示聊天记录（支持多模态）
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "image_url":
                    try:
                        base64_str = item["image_url"]["url"].split(",")[1]
                        st.image(base64.b64decode(base64_str), use_column_width=True)
                    except:
                        st.error("Fail to load image file")
                elif item["type"] == "text" and item["text"].strip():
                    st.markdown(item["text"])
        else:
            st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🧠 Reasoning Content(Click to unfold)"):
                st.markdown(msg["reasoning"])

# 处理用户输入（新增图片上传）
uploaded_files = st.file_uploader(
    "📤 Upload Image File",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="file_uploader"
)

if prompt := st.chat_input("Input your question or description"):
    # Create message
    message_content = []

    # Processing upload file
    for uploaded_file in uploaded_files:
        if uploaded_file:
            base64_str = base64.b64encode(uploaded_file.read()).decode("utf-8")
            mime_type = uploaded_file.type
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_str}"
                }
            })
            uploaded_file.seek(0)  # Reset pointer

    # Process text input
    if prompt.strip():
        message_content.append({
            "type": "text",
            "text": prompt.strip()
        })

    # Save message
    user_message = {
        "role": "user",
        "content": message_content if len(message_content) > 1 else prompt
    }
    st.session_state.messages.append(user_message)

    # Generate Filename
    filename_content = prompt.strip() if prompt.strip() else "VLM Chat"
    if not st.session_state.current_convo:
        st.session_state.current_convo = generate_filename(filename_content)

    # Display User Message
    with st.chat_message("user", avatar="🧑"):
        for item in message_content:
            if item["type"] == "image_url":
                try:
                    base64_str = item["image_url"]["url"].split(",")[1]
                    st.image(base64.b64decode(base64_str), use_column_width=True)
                except:
                    st.error("Fail to display image")
            elif item["type"] == "text":
                st.markdown(item["text"])

    # Choose Model
    use_vlm = any(
        isinstance(msg.get("content"), list) and
        any(item.get("type") == "image_url" for item in msg.get("content", []))
        for msg in st.session_state.messages[-1:]
    )

    # Creating api request
    try:
        with st.chat_message("assistant", avatar="🤖"):
            answer_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            full_answer = ""
            full_reasoning = ""

            # Convert message
            api_messages = convert_messages_for_api(st.session_state.messages, use_vlm)

            # API request
            stream = client.chat.completions.create(
                model=vlm_model if use_vlm else st.session_state.selected_model,
                messages=api_messages,
                stream=True,
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p
#                top_k=st.session_state.top_k
            )

            # Streaming Mode
            for chunk in stream:
                # Content
                if chunk.choices[0].delta.content:
                    full_answer += chunk.choices[0].delta.content or ""
                    answer_placeholder.markdown(full_answer + "▌")

                # Reasoning
                if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                    reasoning = chunk.choices[0].delta.reasoning_content or ""
                    full_reasoning += reasoning
                    with reasoning_placeholder.expander("🤔 Real-time Reasoning:"):
                        st.markdown(full_reasoning)

            # Save response
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "reasoning": full_reasoning.strip()
            })

    except Exception as e:
        st.error(f"Fail to get response: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Response Failure",
            "reasoning": f"Error: {str(e)}"
        })

    # Save chat history
    if st.session_state.current_convo:
        save_conversation()
        refresh_convo_list()

# Scrolling
st.markdown("""
<script>
window.addEventListener('DOMContentLoaded', () => {
    const scrollToBottom = () => {
        window.scrollTo(0, document.body.scrollHeight);
    };
    scrollToBottom();
});
</script>
""", unsafe_allow_html=True)
