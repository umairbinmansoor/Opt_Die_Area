import streamlit as st

# App title
st.set_page_config(page_title="Chatbot Interface", layout="wide")
st.title("Silicon GPT")

# Custom CSS for sidebar width, input styling, and hover effects
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px; /* Adjust this value to control the sidebar width */
        max-width: 300px;
        background-color: #f4f4f4; /* Optional: Sidebar background color */
        padding-top: 20px;
    }

    /* Style for hoverable chat topics in the sidebar */
    .hover-link {
        font-size: 16px;
        color: #333;
        margin: 5px 0;
        padding: 8px 10px;
        border-radius: 5px;
        transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
    }
    .hover-link:hover {
        background-color: #21a0a0; /* Highlight background color */
        color: white; /* Highlight text color */
    }

    /* Style for the input box with embedded arrow */
    .input-container {
        display: flex;
        align-items: center;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 5px 10px;
        background-color: #fff;
    }
    .input-box {
        flex-grow: 1;
        border: none;
        outline: none;
        font-size: 16px;
    }
    .send-button {
        cursor: pointer;
        font-size: 18px;
        color: #888;
        margin-left: 8px;
        transition: color 0.3s ease-in-out;
    }
    .send-button:hover {
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

# Optional Footer Buttons
st.markdown("---")
cols = st.columns(5)
cols[0].button("Create image")
cols[1].button("Get advice")
cols[2].button("Brainstorm")
cols[3].button("Help me write")
cols[4].button("Summarize text")
