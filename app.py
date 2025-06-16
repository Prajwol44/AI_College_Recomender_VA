import streamlit as st
import pandas as pd

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="AI College Assistant", page_icon="🎓", layout="centered")

# 🔄 Load college data
@st.cache_data
def load_data():
    df = pd.read_csv("data/colleges.csv")
    return df

college_df = load_data()

# 🎯 Filter logic based on keywords
def recommend_colleges(query, df):
    query = query.lower()
    result_df = df.copy()

    # Filter by stream
    if "engineering" in query:
        result_df = result_df[result_df['Stream'].str.lower() == "engineering"]
    elif "medical" in query:
        result_df = result_df[result_df['Stream'].str.lower() == "medical"]
    elif "commerce" in query:
        result_df = result_df[result_df['Stream'].str.lower() == "commerce"]
    elif "law" in query:
        result_df = result_df[result_df['Stream'].str.lower() == "law"]

    # Filter by location (dynamic)
    for city in df['Location'].unique():
        if city.lower() in query:
            result_df = result_df[result_df['Location'].str.lower() == city.lower()]

    # Sort by rank and return top 5
    return result_df.sort_values(by="Rank").head(5)

# 🧠 UI Begins
st.title("🎓 AI College Assistant")
st.markdown("""
Welcome to the AI College Assistant!  
This tool helps you discover colleges in India based on your stream, location, and entrance exams.

**Examples**:
- "Top engineering colleges in Delhi"
- "Best medical colleges in Vellore"
- "Commerce colleges accepting CUET"
""")

st.divider()

# 📝 User input
user_input = st.text_input("🗨️ What are you looking for?", placeholder="Type your query here...")

# 📊 Output response
if user_input:
    st.info("🔍 Searching for colleges that match your preferences...")
    results = recommend_colleges(user_input, college_df)

    if not results.empty:
        st.success(f"✅ Found {len(results)} matching college(s):")
        for i, row in results.iterrows():
            st.markdown(f"""
            ---
            ### 🏫 {row['College Name']}
            - 📍 **Location**: {row['Location']}
            - 🎓 **Stream**: {row['Stream']}
            - 🧪 **Exam Accepted**: {row['Exams Accepted']}
            - 🏆 **Rank**: {row['Rank']}
            - 💰 **Fees**: ₹{int(row['Fees']):,}/year
            """)
    else:
        st.warning("❌ Sorry, no matching colleges found for your query.")

st.divider()
st.caption("Made with ❤️ using Streamlit")
