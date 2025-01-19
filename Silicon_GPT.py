import streamlit as st

# App title
st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Silicon GPT")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Recent Conversations")
st.sidebar.button("Conversation 1")
st.sidebar.button("Conversation 2")
st.sidebar.button("New Conversation")

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

# User input
user_input = st.text_input("", key="user_input", placeholder="Message SiliconGPT")

# Send button
if st.button("Send"):
    if user_input.strip():
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Bot response (placeholder for real AI interaction)
        bot_response = "This is a placeholder response. Replace with AI logic."
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Clear input box
        st.session_state.user_input = ""

# Optional Footer Buttons
st.markdown("---")
cols = st.columns(5)
cols[0].button("Create image")
cols[1].button("Get advice")
cols[2].button("Brainstorm")
cols[3].button("Help me write")
cols[4].button("Summarize text")
