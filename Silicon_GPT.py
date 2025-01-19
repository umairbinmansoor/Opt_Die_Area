import streamlit as st

# App title
st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Silicon GPT")

# Custom CSS for sidebar and input styling
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px; /* Adjust this value to control the width */
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
    .input-container {
        display: flex;
        align-items: center;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 5px 10px;
        background-color: #fff;
        width: 100%;
    }
    .input-box {
        flex-grow: 1;
        border: none;
        outline: none;
        font-size: 16px;
    }
    .send-arrow {
        color: #888;
        font-size: 18px;
        cursor: pointer;
        margin-left: 10px;
        transition: color 0.3s ease-in-out;
    }
    .send-arrow:hover {
        color: #21a0a0;
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

# Input box with embedded send button
st.markdown(
    """
    <div class="input-container">
        <input type="text" id="user_input" class="input-box" placeholder="Message SiliconGPT">
        <span class="send-arrow" onclick="sendMessage()">➡️</span>
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
            // Simulate sending a message to Streamlit (for display only)
            const messageContainer = document.createElement("div");
            messageContainer.textContent = "You: " + userInput;
            document.body.appendChild(messageContainer);
            document.getElementById("user_input").value = ""; // Clear input
        }
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# Optional Footer Buttons
st.markdown("---")
cols = st.columns(5)
cols[0].button("Create image")
cols[1].button("Get advice")
cols[2].button("Brainstorm")
cols[3].button("Help me write")
cols[4].button("Summarize text")
