import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np
import re
from textblob import TextBlob # type: ignore
import matplotlib.pyplot as plt # type: ignore
from wordcloud import WordCloud # type: ignore
import time
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* College card styling */
    .college-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .college-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Sidebar metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Animation keyframes */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 10px 20px;
    }
    
    /* Sidebar header */
    .sidebar-header {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 🔄 Load college data with caching
@st.cache_data
def load_data():
    """Load college data from CSV file"""
    try:
        # Try to load the CSV file
        df = pd.read_csv('data/colleges.csv')
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure required columns exist, if not create default values
        required_columns = ['College Name', 'Stream', 'Location', 'Rank', 'Fees', 'Exams Accepted', 'Website']
        
        for col in required_columns:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found in CSV. Please check your data structure.")
        
        # Add sentiment score if not present
        if 'Sentiment Score' not in df.columns:
            df['Sentiment Score'] = np.random.uniform(0.7, 0.95, len(df))
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Could not find 'data/colleges.csv'. Please ensure the file exists.")
        st.info("Expected CSV structure:")
        st.code("""
College Name,Stream,Location,Rank,Fees,Exams Accepted,Website
IIT Bombay,Engineering,Mumbai,1,200000,JEE Advanced,https://www.iitb.ac.in/
AIIMS Delhi,Medical,Delhi,1,100000,NEET,https://www.aiims.edu/
        """)
        
        # Return empty dataframe to prevent crashes
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")
        return pd.DataFrame()

# Load the data
college_df = load_data()

# Check if data was loaded successfully
if college_df.empty:
    st.stop()  # Stop execution if no data is available

# 🧠 NLP Functions
def analyze_sentiment(query):
    """Analyze sentiment of user query"""
    try:
        analysis = TextBlob(query)
        return analysis.sentiment.polarity
    except:
        return 0.0

def extract_keywords(query):
    """Extract keywords from user query"""
    keywords = {
        'streams': [],
        'locations': [],
        'exams': [],
        'fees': None,
        'rank': None
    }
    
    stream_keywords = ['engineering', 'medical', 'commerce', 'law', 'arts', 'science', 'management']
    for stream in stream_keywords:
        if re.search(rf'\b{stream}\b', query, re.IGNORECASE):
            keywords['streams'].append(stream)
    
    locations = college_df['Location'].str.lower().unique()
    for loc in locations:
        if re.search(rf'\b{loc}\b', query, re.IGNORECASE):
            keywords['locations'].append(loc.title())
    
    exams = ['JEE', 'NEET', 'CUET', 'CLAT', 'BITSAT', 'VITEEE']
    for exam in exams:
        if re.search(rf'\b{exam}\b', query, re.IGNORECASE):
            keywords['exams'].append(exam)
    
    if re.search(r'\b(low|cheap|affordable)\b', query, re.IGNORECASE):
        keywords['fees'] = 'low'
    elif re.search(r'\b(high|expensive|premium)\b', query, re.IGNORECASE):
        keywords['fees'] = 'high'
    
    if re.search(r'\b(top|best|high)\b', query, re.IGNORECASE):
        keywords['rank'] = 'high'
    elif re.search(r'\b(average|moderate)\b', query, re.IGNORECASE):
        keywords['rank'] = 'medium'
    
    return keywords

def recommend_colleges(query, df):
    sentiment = analyze_sentiment(query)
    keywords = extract_keywords(query)
    
    result_df = df.copy()
    
    if keywords['streams']:
        result_df = result_df[result_df['Stream'].str.lower().isin(keywords['streams'])]
    
    if keywords['locations']:
        result_df = result_df[result_df['Location'].isin(keywords['locations'])]
    
    if keywords['exams']:
        exam_filter = result_df['Exams Accepted'].apply(
            lambda x: any(exam in x for exam in keywords['exams'])
        )
        result_df = result_df[exam_filter]
    
    if keywords['fees'] == 'low':
        result_df = result_df[result_df['Fees'] < result_df['Fees'].median()]
    elif keywords['fees'] == 'high':
        result_df = result_df[result_df['Fees'] > result_df['Fees'].median()]
    
    if keywords['rank'] == 'high':
        result_df = result_df[result_df['Rank'] <= 3]
    elif keywords['rank'] == 'medium':
        result_df = result_df[result_df['Rank'].between(4, 7)]
    
    if 'Sentiment Score' in result_df.columns:
        result_df = result_df.sort_values(by=['Sentiment Score', 'Rank'], ascending=[False, True])
    else:
        result_df = result_df.sort_values(by='Rank')
    
    return result_df.head(5), sentiment, keywords

def display_chat_message(role, content):
    """Display chat message with beautiful styling"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🎓 AI Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def generate_response(user_input):
    """Generate AI response with beautiful formatting"""
    time.sleep(0.5)
    
    results, sentiment, keywords = recommend_colleges(user_input, college_df)
    
    response = ""
    
    if sentiment > 0.3:
        greeting = "🌟 Great to hear your enthusiasm! "
    elif sentiment < -0.3:
        greeting = "💙 I understand this can be stressful. "
    else:
        greeting = "✨ "
    
    response += f"{greeting}Here are some colleges that match your preferences:"
    
    # Show keyword understanding
    if any(keywords.values()):
        response += "\n\n🔍 **I detected these preferences:**"
        if keywords['streams']:
            response += f"\n• **Streams:** {', '.join(keywords['streams']).title()}"
        if keywords['locations']:
            response += f"\n• **Locations:** {', '.join(keywords['locations'])}"
        if keywords['exams']:
            response += f"\n• **Exams:** {', '.join(keywords['exams'])}"
        if keywords['fees']:
            response += f"\n• **Fee preference:** {keywords['fees']}"
        if keywords['rank']:
            response += f"\n• **Rank preference:** {keywords['rank']}"
    
    # Add college recommendations
    if not results.empty:
        response += "\n\n🏫 **Top Recommendations:**"
        for i, row in results.iterrows():
            response += f"""
            
<div class="college-card">
<h3>🎓 {row['College Name']}</h3>
<p><strong>📍 Location:</strong> {row['Location']}</p>
<p><strong>🧪 Stream:</strong> {row['Stream']}</p>
<p><strong>📝 Exams:</strong> {row['Exams Accepted']}</p>
<p><strong>🏆 Rank:</strong> #{row['Rank']}</p>
<p><strong>💰 Fees:</strong> ₹{int(row['Fees']):,}/year</p>
<p><strong>🔗 Website:</strong> <a href="{row['Website']}" target="_blank" style="color: white;">Visit Website</a></p>
</div>
            """
    else:
        response += "\n\n😞 I couldn't find colleges matching all your criteria. Try broadening your search or ask me about specific streams!"
    
    response += "\n\n💬 **How else can I assist with your college search?**"
    
    return response

# 🧠 Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {"role": "assistant", "content": "Hi there! 👋 I'm your AI College Assistant. I can help you find the perfect college in India after 12th grade. Tell me about your preferences - what stream are you interested in? (Engineering, Medical, Commerce, etc.)"}
    ]

# 📊 Sidebar for additional features
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 College Insights</div>', unsafe_allow_html=True)
    
    if not college_df.empty:
        # Stream distribution with Plotly
        st.subheader("🎯 Popular Streams")
        stream_counts = college_df['Stream'].value_counts()
        
        fig_pie = px.pie(
            values=stream_counts.values,
            names=stream_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
            title="Stream Distribution"
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Location distribution
        st.subheader("📍 Top Locations")
        location_counts = college_df['Location'].value_counts().head(5)
        
        fig_bar = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            color=location_counts.values,
            color_continuous_scale='Viridis',
            title="Colleges by Location"
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Fees distribution
        st.subheader("💰 Fee Range Analysis")
        fig_hist = px.histogram(
            college_df,
            x='Fees',
            nbins=10,
            color_discrete_sequence=['#667eea'],
            title="Fee Distribution"
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Sentiment distribution
        if 'Sentiment Score' in college_df.columns:
            st.subheader("😊 Student Satisfaction")
            avg_sentiment = college_df['Sentiment Score'].mean()
            
            # Create a gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_sentiment * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Satisfaction (%)"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Quick stats
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Total Colleges", len(college_df))
            with col2:
                st.metric("🎯 Avg Rank", f"{college_df['Rank'].mean():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No data available. Please check your CSV file.")

# 🧠 Main Chat Interface
st.markdown('<h1 class="main-title">🎓 AI College Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your personalized guide to finding the perfect college in India</p>', unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display conversation history
for message in st.session_state.conversation:
    display_chat_message(message["role"], message["content"])

st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("🗨️ Ask about colleges...")

if user_input:
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    # Generate and display AI response
    with st.spinner("🤔 Analyzing your preferences..."):
        ai_response = generate_response(user_input)
        st.session_state.conversation.append({"role": "assistant", "content": ai_response})
    
    # Auto-scroll to bottom
    st.rerun()

# Reset button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.conversation = [
            {"role": "assistant", "content": "Hi there! 👋 I'm your AI College Assistant. Where shall we start today?"}
        ]
        st.rerun()