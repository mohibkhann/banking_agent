import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import your enhanced router (adjust path as needed)
try:
    from banking_agent.agents.agent_router import EnhancedPersonalFinanceRouter
except ImportError:
    st.error("Could not import PersonalFinanceRouter. Please check the file path.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GX Bank Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Updated to consistent dark theme color system */
    :root {
        --primary-purple: #A855F7;
        --primary-purple-dark: #9333EA;
        --accent-green: #10B981;
        --accent-orange: #F59E0B;
        --accent-blue: #3B82F6;
        --accent-red: #EF4444;
        --neutral-white: #FFFFFF;
        --neutral-light: #F8FAFC;
        --neutral-medium: #64748B;
        --neutral-dark: #1E293B;
        --neutral-darker: #0F172A;
        --bg-primary: #1a1a1a;
        --bg-secondary: #2a2a2a;
        --bg-card: #333333;
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0B0;
        --shadow-light: 0 1px 3px rgba(0, 0, 0, 0.3);
        --shadow-medium: 0 4px 6px rgba(0, 0, 0, 0.4);
        --shadow-large: 0 10px 25px rgba(0, 0, 0, 0.5);
        --border-radius: 12px;
        --border-radius-large: 20px;
    }
    
    /* Dark theme main background */
    .main > div {
        background: var(--bg-primary);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Updated header with dark theme and purple branding */
    .header-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
        padding: 2.5rem 2rem;
        border-radius: var(--border-radius-large);
        margin-bottom: 2rem;
        color: var(--text-primary);
        text-align: center;
        box-shadow: var(--shadow-large);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-purple), var(--accent-green), var(--accent-orange));
    }
    
    .logo-container {
        margin-bottom: 1.5rem;
    }
    
    .gx-logo {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-purple-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        margin: 0;
        position: relative;
    }
    
    .gx-logo::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-purple), var(--primary-purple-dark));
        border-radius: 2px;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .client-info {
        font-size: 1rem;
        margin-top: 0.5rem;
        opacity: 0.8;
        font-weight: 400;
        color: var(--text-secondary);
    }
    
    /* Custom scrollbar for dark theme */
    .main > div::-webkit-scrollbar {
        width: 6px;
    }
    
    .main > div::-webkit-scrollbar-track {
        background: var(--bg-card);
        border-radius: 3px;
    }
    
    .main > div::-webkit-scrollbar-thumb {
        background: var(--primary-purple);
        border-radius: 3px;
    }
    
    .main > div::-webkit-scrollbar-thumb:hover {
        background: var(--primary-purple-dark);
    }
    
    /* Updated message bubbles for dark theme */
    .user-message {
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-purple-dark));
        color: var(--text-primary);
        padding: 1.25rem 1.75rem;
        border-radius: var(--border-radius-large) var(--border-radius-large) 8px var(--border-radius-large);
        margin: 1rem 0 1rem auto;
        max-width: 75%;
        box-shadow: var(--shadow-medium);
        font-weight: 500;
        line-height: 1.6;
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .assistant-message {
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 1.75rem;
        border-radius: var(--border-radius-large) var(--border-radius-large) var(--border-radius-large) 8px;
        margin: 1rem auto 1rem 0;
        max-width: 85%;
        border-left: 4px solid var(--accent-green);
        box-shadow: var(--shadow-light);
        animation: slideInLeft 0.3s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .assistant-header {
        color: var(--accent-green);
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .assistant-content {
        line-height: 1.7;
        font-weight: 400;
        color: var(--text-primary);
    }
    
    /* Dark theme input container */
    .input-container {
        background: var(--bg-secondary);
        border-radius: var(--border-radius-large);
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: var(--primary-purple);
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2);
    }
    
    /* Hide streamlit form elements */
    .stForm {
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stForm > div {
        border: none !important;
        padding: 0 !important;
    }
    
    /* Dark theme input styling */
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 2px solid transparent !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        height: 56px !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-purple) !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2) !important;
        background: var(--bg-secondary) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-secondary) !important;
        font-weight: 400 !important;
    }
    
    .stTextInput > div {
        height: 56px !important;
        margin-bottom: 0 !important;
    }
    
    .stTextInput {
        margin-bottom: 0 !important;
        flex: 1;
    }
    
    /* Updated send button with purple theme */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--primary-purple-dark)) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        height: 56px !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-medium) !important;
        min-width: 120px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-large) !important;
        background: linear-gradient(135deg, var(--primary-purple-dark), #7C3AED) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Dark theme action buttons */
    .action-button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-light) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .action-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-medium) !important;
        border-color: var(--primary-purple) !important;
        color: var(--primary-purple) !important;
    }
    
    /* Dark theme typing indicator */
    .typing-indicator {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--border-radius-large) var(--border-radius-large) var(--border-radius-large) 8px;
        margin: 1rem auto 1rem 0;
        max-width: 280px;
        border-left: 4px solid var(--accent-green);
        animation: pulse 1.5s infinite;
        box-shadow: var(--shadow-light);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    .typing-dots {
        display: flex;
        gap: 6px;
        margin-top: 0.75rem;
        justify-content: center;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    .typing-dot:nth-child(3) { animation-delay: 0; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    
    /* Dark theme welcome message */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--bg-secondary);
        border-radius: var(--border-radius-large);
        border: 2px solid rgba(168, 85, 247, 0.3);
        max-width: 600px;
        margin: 0 auto;
        color: var(--text-primary);
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    /* Dark theme feature cards with colored accents */
    .feature-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border-left: 4px solid var(--accent-green);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    .feature-card:nth-child(1) { border-left-color: var(--accent-green); }
    .feature-card:nth-child(2) { border-left-color: var(--accent-orange); }
    .feature-card:nth-child(3) { border-left-color: var(--accent-blue); }
    .feature-card:nth-child(4) { border-left-color: var(--accent-red); }
    .feature-card:nth-child(5) { border-left-color: var(--primary-purple); }
    
    .feature-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Dark theme conversation history */
    .conversation-item {
        background: var(--bg-card) !important;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 0.75rem;
        border-left: 3px solid var(--primary-purple);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .conversation-time {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 600;
    }
    .conversation-query {
        color: var(--text-primary);
        font-weight: 600;
        margin: 0.25rem 0;
    }
    .conversation-preview {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gx-logo {
            font-size: 2.5rem;
        }
        
        .header-title {
            font-size: 1.5rem;
        }
        
        .user-message, .assistant-message {
            max-width: 95%;
        }
        
        .header-container {
            padding: 2rem 1.5rem;
        }
        
        .welcome-container {
            padding: 2rem 1.5rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_client" not in st.session_state:
    st.session_state.current_client = 430
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False
if "router" not in st.session_state:
    # Initialize router - adjust paths as needed
    try:
        st.session_state.router = EnhancedPersonalFinanceRouter(
            client_csv_path="Banking_Data.csv",
            overall_csv_path="overall_data.csv"
        )
    except Exception as e:
        st.error(f"Failed to initialize router: {e}")
        st.stop()

# Helper functions
def format_response(response_text, response_data=None):
    """Format assistant response with rich content"""
    formatted_html = f"""
    <div class="assistant-message">
        <div class="assistant-header">
            🤖 GX Assistant
        </div>
        <div class="assistant-content">
            {response_text}
        </div>
    </div>
    """
    return formatted_html

def display_welcome_message():
    """Display enhanced welcome message with modern design"""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🏦</div>
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #10B981; font-size: 1.5rem; margin-bottom: 1rem;">🤖 Welcome to GX Bank Assistant!</h2>
            <p style="color: #FFFFFF; font-size: 1.1rem; line-height: 1.6;">
                I'm your intelligent financial companion, ready to help you manage your money smarter. Here's what I can do for you:
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">💰 Spending Analysis</div>
            <div class="feature-description">"How much did I spend on restaurants last month?"</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Budget Management</div>
            <div class="feature-description">"Create a $800 monthly budget for groceries"</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">💳 Banking Products</div>
            <div class="feature-description">"What credit cards and accounts do you offer?"</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📈 Financial Insights</div>
            <div class="feature-description">"How do I compare to similar customers?"</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Smart Recommendations</div>
            <div class="feature-description">"Which savings account is best for my goals?"</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(16, 185, 129, 0.1)); border-radius: 12px; border: 1px solid rgba(168, 85, 247, 0.3);">
        <div style="font-weight: 700; color: #A855F7; margin-bottom: 0.5rem;">💡 Pro Tip</div>
        <div style="color: #FFFFFF;">Ask me anything about your finances using natural language - I understand context and can help with complex queries!</div>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    header_html = """
    <div class="header-container">
        <div class="logo-container">
            <h1 class="gx-logo">GXBank</h1>
        </div>
        <div class="header-content">
            <h2 class="header-title">AI-Powered Banking Assistant</h2>
            <div class="client-info">Intelligent Financial Management • Secure & Personalized</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Main chat interface
    col1 = st.columns([1])[0]
    
    with col1:
        # Chat content
        chat_placeholder = st.empty()
        
        with chat_placeholder.container():
            if not st.session_state.messages:
                display_welcome_message()
            
            # Display conversation messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    user_html = f'<div class="user-message">{message["content"]}</div>'
                    st.markdown(user_html, unsafe_allow_html=True)
                else:
                    assistant_html = format_response(message["content"])
                    st.markdown(assistant_html, unsafe_allow_html=True)
        
        # Input area - Single unified input with proper form handling
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Message",
                    placeholder="💬 Ask me about your spending, budgets, or banking products...",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col_send:
                send_button = st.form_submit_button("Send ➤", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle form submission (triggered by Enter key or Send button)
        if send_button and user_input.strip():
            st.session_state.processing_query = True
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            
            # Clear and rerun to show typing indicator
            st.rerun()
        
        # Handle processing state
        if st.session_state.processing_query:
            # Show typing indicator
            with chat_placeholder.container():
                # Display all previous messages
                if not st.session_state.messages:
                    display_welcome_message()
                
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        user_html = f'<div class="user-message">{message["content"]}</div>'
                        st.markdown(user_html, unsafe_allow_html=True)
                    else:
                        assistant_html = format_response(message["content"])
                        st.markdown(assistant_html, unsafe_allow_html=True)
                
                # Enhanced typing indicator
                st.markdown("""
                <div class="typing-indicator">
                    <div style="color: #10B981; font-weight: bold; display: flex; align-items: center; gap: 0.5rem;">
                        🤖 GX Assistant is analyzing your request
                    </div>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Get the last user message for processing
            last_user_message = None
            for message in reversed(st.session_state.messages):
                if message["role"] == "user":
                    last_user_message = message["content"]
                    break
            
            # Get response from router
            if last_user_message:
                try:
                    response = st.session_state.router.chat(
                        client_id=st.session_state.current_client,
                        user_query=last_user_message
                    )
                    
                    assistant_response = response.get("response", "I apologize, but I'm having trouble processing your request right now. Please try again.")
                    
                    # Add assistant response with data
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "data": response
                    })
                    
                    # Save to conversation history
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "query": last_user_message[:50] + "..." if len(last_user_message) > 50 else last_user_message,
                        "preview": assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
                    })
                    
                    # Reset processing state
                    st.session_state.processing_query = False
                    
                    # Rerun to update display
                    st.rerun()
                    
                except Exception as e:
                    error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    st.session_state.processing_query = False
                    st.rerun()
    
    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
    
    # New conversation and additional controls
    col_new, col_stats, col_help = st.columns([2, 2, 2])
    
    with col_new:
        if st.button("🆕 New Conversation", use_container_width=True, key="new_conv", help="Start a fresh conversation"):
            st.session_state.messages = []
            st.session_state.processing_query = False
            st.rerun()
    
    with col_stats:
        if st.button("📊 Quick Summary", use_container_width=True, key="quick_stats", help="Get an overview of your finances"):
            # Add a quick stats query
            if not st.session_state.processing_query:
                st.session_state.processing_query = True
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "Show me a quick summary of my spending this month with key insights"
                })
                st.rerun()
    
    with col_help:
        if st.button("💡 Examples", use_container_width=True, key="help_tips", help="See example questions you can ask"):
            # Add a help query
            if not st.session_state.processing_query:
                help_message = """Here are some powerful questions you can ask me:

**💰 Spending Intelligence:**
• "Analyze my spending patterns for the last 3 months"
• "Which categories am I overspending in?"
• "Show me unusual transactions this week"

**📊 Budget Optimization:**  
• "Create an optimized budget based on my spending history"
• "How can I save $500 more per month?"
• "Track my progress against financial goals"

**💳 Product Recommendations:**
• "Which GX Bank credit card maximizes my rewards?"
• "Compare savings accounts for my situation"
• "What investment options suit my risk profile?"

**📈 Financial Health:**
• "How does my financial health compare to peers?"
• "Predict my spending for next month"
• "Identify opportunities to improve my credit score"

**🎯 Smart Insights:**
• "What's the best time to make large purchases?"
• "How can I optimize my cash flow?"
• "Create a personalized financial action plan"

💡 **Remember:** I learn from our conversation context, so feel free to ask follow-up questions and dive deeper into any topic!"""

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": help_message
                })
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.conversation_history:
        with st.expander("💬 Recent Conversations", expanded=False):
            st.markdown("""
            <style>
            .conversation-item {
                background: var(--bg-card) !important;
                padding: 1rem;
                border-radius: var(--border-radius);
                margin-bottom: 0.75rem;
                border-left: 3px solid var(--primary-purple);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .conversation-time {
                color: var(--text-secondary);
                font-size: 0.85rem;
                font-weight: 600;
            }
            .conversation-query {
                color: var(--text-primary);
                font-weight: 600;
                margin: 0.25rem 0;
            }
            .conversation-preview {
                color: var(--text-secondary);
                font-size: 0.9rem;
                line-height: 1.4;
            }
            </style>
            """, unsafe_allow_html=True)
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
                st.markdown(f"""
                <div class="conversation-item">
                    <div class="conversation-time">{conv['timestamp']}</div>
                    <div class="conversation-query">{conv['query']}</div>
                    <div class="conversation-preview">{conv['preview']}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
