import streamlit as st

# App title
st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Silicon GPT")

# Custom CSS for reducing the sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 200px; /* Adjust this value to control the width */
        max-width: 200px; /* Ensures the width is fixed */
        background-color: #f4f4f4; /* Optional: Sidebar background color */
        padding-top: 20px;
    }
    .hover-link {
        color: #2b6777; 
        font-size: 14px;
        margin-bottom: 10px;
        text-decoration: none;
        transition: color 0.3s ease-in-out;
    }
    .hover-link:hover {
        color: #21a0a0;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Recent Conversations")

# Hoverable text links for recent conversations
recent_conversations = [
    "Die Yield Calculator",
    "AI Semiconductor Rephrasings",
    "Tanh Derivative: 1 - 1 Tanh^2",
    "HCI Course Teaching Support",
]

for topic in recent_conversations:
    st.sidebar.markdown(f'<div class="hover-link">{topic}</div>', unsafe_allow_html=True)

# Chat container
st.markdown("## What can I help with?")
chat_container = st.container()

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# User input with arrow send icon
st.markdown(
    """
    <style>
    input[type="text"] {
        padding-left: 10px;
    }
    .send-arrow {
        position: absolute;
        right: 10px;
        top: 8px;
        color: #888;
        font-size: 18px;
        cursor: pointer;
    }
    .send-arrow:hover {
        color: #21a0a0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Collect user input and send when Enter is pressed
input_container = st.container()
user_input = st.text_input(
    "",
    key="user_input",
    placeholder="Message SiliconGPT",
    label_visibility="collapsed",
)

arrow_clicked = st.markdown('<div class="send-arrow">➡️</div>', unsafe_allow_html=True)

if user_input.strip():
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Bot response (placeholder for real AI interaction)
    bot_response = "This is a placeholder response. Replace with AI logic."
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Clear input box
    st.sessi
