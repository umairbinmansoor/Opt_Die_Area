import streamlit as st

# App title
st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Silicon GPT")

# Custom CSS for reducing the sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px; /* Adjust this value to control the width */
        max-width: 300px; /* Ensures the width is fixed */
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

# Input box with send button inside
st.markdown(
    """
    <div class="input-container">
        <input type="text" id="user_input" class="input-box" placeholder="Message SiliconGPT">
        <span class="send-button" onclick="sendMessage()">➡️</span>
    </div>

    <script>
    const input = document.getElementById("user_input");
    input.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userInput = document.getElementById("user_input").value;
        if (userInput.trim() !== "") {
            // Pass the input back to Streamlit (placeholder, implement real logic)
            document.getElementById("user_input").value = ""; // Clear input
            console.log("Message Sent: " + userInput); // Debug in console
        }
    }
    </script>
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
    st.session_state.user_input = ""

# Optional Footer Buttons
st.markdown("---")
cols = st.columns(5)
cols[0].button("Create image")
cols[1].button("Get advice")
cols[2].button("Brainstorm")
cols[3].button("Help me write")
cols[4].button("Summarize text")
