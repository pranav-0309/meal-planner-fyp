# All imports for this file
import streamlit as st
import agentic_rag
from dotenv import load_dotenv
from mem0 import MemoryClient

# Load the environment variables
load_dotenv()

# Setting the title
st.title('Meal Planner using Agentic RAG (Powered by Qdrant, Qwen2.5 Coder 32B and Mem0)')

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Initialize all session states at the start
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llama_model" not in st.session_state:
    st.session_state["llama_model"] = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Create two columns for input and button
col1, col2 = st.columns([3, 1])

# Add user ID input in first column
with col1:
    user_id = st.text_input("Enter your User ID:", key="user_id")

# Add clear chat button in second column
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize the memory client
mem0_memory = MemoryClient(api_key="get your own api key from: https://app.mem0.ai/dashboard/api-keys")

# Function to add messages to the memory
def add_to_memory(user_input: str, ai_response: str, user_id: str):
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ]
        mem0_memory.add(messages, user_id=user_id)

# Function to search from the memory
def search(query: str, user_id: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        return f"Chat History from Memory:\n{mem0_memory.search(query, user_id=user_id)}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
        st.write(message["content"])

# User input and chat logic
if user_msg := st.chat_input("Enter your ingredients, any dietary restrictions, allergy information and daily protein target..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_msg})
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(user_msg)

    # Search from memory
    ctx = search(user_msg, user_id=user_id)

    # Final Message to the bot
    final_msg = f"Context:\n{ctx}\nUser Message:\n{user_msg}"

    # Get bot response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response = agentic_rag.meal_planner_agent.run(final_msg)
        st.write(response)
    
    # Add to memory
    add_to_memory(user_msg, str(response), user_id=user_id)

    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})