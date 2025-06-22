import streamlit as st
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Any
import os

# Initialize the sentence transformer model
@st.cache_resource
def load_model():
    # return SentenceTransformer('all-MiniLM-L6-v2')
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_data
def load_college_data(file_path='final_college_dataset.json'):
    """Load college data from JSON file and convert ratings to floats"""
    try:
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            st.success(f"✅ Successfully loaded data from {file_path}")
        print(f"load_college_data {len(data_dict.keys())}")
        return data_dict
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return {}

# Sample data structure (replace with your actual data loading)
@st.cache_data
def load_college_data_old():
    # This is sample data - replace with your actual JSON data
    sample_data = {
        "676878": {
            "college_name": "All India Institute of Medical Sciences - [AIIMS]",
            "website": "http://www.aiims.edu",
            "phone": "011-26588500",
            "email": "director@aiims.edu",
            "rating_value": 4.3,
            "review_count": "245",
            "worst_rating": 0,
            "best_rating": 5,
            "city": "New Delhi",
            "state": "Delhi NCR",
            "approvals": "MCI, UGC",
            "courses": [
                {"course_name": "Operation Theatre Technology", "fee_amount": 3385},
                {"course_name": "Medical Radiology & Imaging Technology", "fee_amount": 4480}
            ]
        },
        "232688": {
            "college_name": "Delhi Institute of Medical Sciences - [AIIM]",
            "website": "http://www.aiims.edu",
            "phone": "011-56677888",
            "email": "director@aiims.edu",
            "rating_value": 4.3,
            "review_count": "245",
            "worst_rating": 0,
            "best_rating": 5,
            "city": "Delhi",
            "state": "NCR",
            "approvals": "MCI, UGC",
            "courses": [
                {"course_name": "IT Technology", "fee_amount": 3385},
                {"course_name": "Imaging Technology", "fee_amount": 4480}
            ]
        }
    }
    return sample_data

class CollegeChatbot:
    def __init__(self, college_data: Dict, model):
        self.college_data = college_data
        self.model = model
        self.college_descriptions = self._create_college_descriptions()
        self.embeddings = self._create_embeddings()
    
    def _create_college_descriptions(self) -> List[str]:
        """Create searchable text descriptions for each college"""
        descriptions = []
        self.college_ids = []
        
        for college_id, college in self.college_data.items():
            # Create a comprehensive description
            courses_text = ", ".join([course["course_name"] for course in college.get("courses", [])])
            fees_text = ", ".join([f"{course['course_name']} fee {course['fee_amount']}" for course in college.get("courses", [])])
            
            # print(fees_text) # distinugish course and its fee
            print("....."*30)
            # break 
            
            description_old= f"""
            College: {college['college_name']} located in city:{college['city']} and state:{college['state']}
            Rating: {college['rating_value']} stars
            Courses: {courses_text}
            Fees: {fees_text}
            Approvals: {college.get('approvals', '')}
            Contact: {college.get('phone', '')} {college.get('email', '')}
            Website: {college.get('website', '')}
            """
            
            description = f"""
            College: {college['college_name']}
            City:{college['city']}
            State: {college['state']}
            Ratings: {college['rating_value']} 
            Fee: {fees_text}
            Aprovals: {college.get('approvals', '')}
            """
            # This is a {'highly rated' if college['rating_value'] >= 4.0 else 'good' if college['rating_value'] >= 3.5 else 'average'} college.
            
            
            # print(f"description: {description}")
            
            descriptions.append(description.strip())
            self.college_ids.append(college_id)
        
        return descriptions
    
    def _create_embeddings(self):
        """Create embeddings for all college descriptions"""
        return self.model.encode(self.college_descriptions)
    
    def extract_query_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters from user query"""
        params = {
            'city': None,
            'course': None,
            'fee_limit': None,
            'rating_min': None,
            'approval': None
        }
        print(f"extract_query_params: {query}")
        # Extract city names (common Indian cities)
        cities = ['mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune']
        for city in cities:
            if city.lower() in query.strip().lower():
                params['city'] = city.title()
                break
        
        # Extract course names
        courses = ['computer science', 'engineering', 'medical', 'technology', 'it', 'radiology', 'operation theatre']
        for course in courses:
            if course.lower() in query.lower():
                params['course'] = course
                break
        
        # Extract fee limits
        fee_patterns = [
            r'fee.*?under.*?(\d+)',r'fee\s+(\d+)',
            r'fee.*?below.*?(\d+)',
            r'fee.*?less than.*?(\d+)',
            r'under.*?(\d+).*?fee',
            r'with.*?(\d+).*?fee',
            r'below.*?(\d+).*?fee',
            r'less than.*?(\d+).*?fee',
            r'fee.*?(\d+).*?or less',
            r'maximum fee.*?(\d+)',
            r'fee limit.*?(\d+)'
        ]
        
        
        for pattern in fee_patterns:
            fee_match = re.search(pattern, query.lower())
            if fee_match:
                print(f"--fee match {fee_match}")
                params['fee_limit'] = int(fee_match.group(1))
                break
        
        # Extract rating requirements
        rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:star|rating)', query.lower())
        if rating_match:
            params['rating_min'] = float(rating_match.group(1))
        
        return params
    
    def filter_colleges(self, params: Dict[str, Any]) -> List[str]:
        """Filter colleges based on extracted parameters"""
        filtered_ids = []
        print(f"filter_colleges {params}")
        for college_id in self.college_ids:
            college = self.college_data[college_id]
            
            # City filter
            if params['city'] and params['city'].lower() not in college['city'].lower():
                continue
            
            # Rating filter
            if params['rating_min'] and college['rating_value'] < params['rating_min']:
                continue
            
            # Course filter
            if params['course']:
                course_match = any(params['course'].lower() in course['course_name'].lower() 
                                 for course in college.get('courses', []))
                if not course_match:
                    continue
            
            # Fee filter
            if params['fee_limit']:
                affordable_courses = [course for course in college.get('courses', []) 
                                    if course['fee_amount'] <= int(params['fee_limit'])]
                if not affordable_courses:
                    continue
            
            filtered_ids.append(college_id)
        
        return filtered_ids
    
    def search_colleges(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for colleges based on user query"""
        if self.model is None or self.embeddings is None:
            print("Restart the system")
            return []
        
        query = query.strip()
        # Extract parameters and filter
        params = self.extract_query_params(query)
        filtered_ids = self.filter_colleges(params)
        
        # print(f"filtered_id-2: {filtered_ids}")
        
        # Create query embedding
        query_embedding = self.model.encode([query]) # prompt embedding
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of filtered colleges
        # if filtered_ids:
        #     filtered_indices = [self.college_ids.index(cid) for cid in filtered_ids]
        #     filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
        # else:
        #     filtered_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        
        filtered_similarities = [(i, sim) for i, sim in enumerate(similarities)]

        # Sort by similarity and get top results
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = filtered_similarities[:top_k]
        
        # Prepare results
        results = []
        for idx, similarity in top_results:
            print(f"top_results: {idx} - {similarity}")
            
            college_id = self.college_ids[idx]
            college = self.college_data[college_id].copy()
            college['id'] = college_id
            college['similarity_score'] = similarity
            college['extracted_params'] = params
            results.append(college)
        
        return results

def display_college_card(college: Dict):
    """Display a college in a card format"""
    with st.container():
        st.markdown(f"### 🎓 {college['college_name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"📍 **Location:** {college['city']}, {college['state']}")
            st.markdown(f"⭐ **Rating:** {college['rating_value']}/5.0")
            st.markdown(f"✅ **Approvals:** {college.get('approvals', 'N/A')}")
            st.markdown(f"📞 **Phone:** {college.get('phone', 'N/A')}")
            st.markdown(f"📧 **Email:** {college.get('email', 'N/A')}")
            st.markdown(f"🌐 **Website:** {college.get('website', 'N/A')}")
        
        with col2:
            st.markdown(f"**Similarity Score:** {college['similarity_score']:.3f}")
        
        # Display courses
        if college.get('courses'):
            st.markdown("**📚 Available Courses:**")
            for course in college['courses']:
                st.markdown(f"- {course['course_name']}: ₹{course['fee_amount']:}")
        
        st.markdown("---")

def create_similarity_plot(similarities: List[float], college_names: List[str]):
    """Create a plot showing similarity scores"""
    df = pd.DataFrame({
        'College': college_names,
        'Similarity Score': similarities
    })
    
    fig = px.bar(df, x='Similarity Score', y='College', orientation='h',
                 title='College Similarity Scores',
                 color='Similarity Score',
                 color_continuous_scale='viridis')
    
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    return fig

def main():
    st.set_page_config(page_title="College Recommendation Chatbot", page_icon="🎓", layout="wide")
    
    st.title("🎓 College Recommendation Chatbot")
    st.markdown("Find the perfect college based on your preferences using AI-powered search!")
    
    # Load model and data
    model = load_model()
    college_data = load_college_data()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CollegeChatbot(college_data, model)
    
    # Sidebar for sample queries
    with st.sidebar:
        st.header("📝 Sample Queries")
        sample_queries = [
            "Best colleges in Delhi for medical courses",
            "Engineering colleges with good ratings",
            "IT courses under ₹50000 fee",
            "Colleges with MCI approval",
            "Top rated colleges in Mumbai",
            "Best medical colleges in Delhi with good ratings",
            "Engineering colleges in Mumbai under 200000 fee",
            "Computer science courses in Bangalore",
            "Top rated colleges for MBBS",
            "IT engineering colleges in Pune under 100000",
            "Dental colleges in Chennai",
            "Mechanical engineering with AICTE approval"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.user_query = query
    
    # Main chat interface
    user_query = st.text_input("Enter your query about colleges:", 
                              value=st.session_state.get('user_query', ''),
                              placeholder="e.g., 'Show me medical colleges in Delhi with good ratings'")
    
    if st.button("🔍 Search Colleges") and user_query.strip():
        with st.spinner("Searching for colleges..."):
            results = st.session_state.chatbot.search_colleges(user_query.strip(), top_k=5)
            
            if results:
                st.success(f"Found {len(results)} relevant colleges!")
                
                # Display results
                st.header("🏫 Recommended Colleges")
                for college in results:
                    display_college_card(college)
                
                # Create and display similarity plot
                similarities = [college['similarity_score'] for college in results]
                college_names = [college['college_name'][:30] + "..." if len(college['college_name']) > 30 
                               else college['college_name'] for college in results]
                
                # st.header("📊 Similarity Analysis")
                # fig = create_similarity_plot(similarities, college_names)
                # st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.header("🔍 Search Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_rating = np.mean([college['rating_value'] for college in results])
                    st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
                
                with col2:
                    avg_similarity = np.mean(similarities)
                    st.metric("Average Similarity", f"{avg_similarity:.3f}")
                
                with col3:
                    total_courses = sum(len(college.get('courses', [])) for college in results)
                    st.metric("Total Courses", total_courses)
            else:
                st.warning("No colleges found matching your criteria. Try a different query!")
    
    # Instructions
    with st.expander("ℹ️ How to use this chatbot"):
        st.markdown("""
        **This chatbot can help you find colleges based on:**
        - 🏙️ **City/Location**: "colleges in Delhi", "Mumbai colleges"
        - ⭐ **Ratings**: "good ratings", "4 star colleges"
        - 📚 **Courses**: "medical courses", "engineering", "IT technology"
        - 💰 **Fee Structure**: "fee under 50000", "affordable colleges"
        - ✅ **Approvals**: "MCI approved", "UGC approved"
        
        **Sample queries:**
        - "Can you recommend colleges in Delhi with good ratings that offer medical courses?"
        - "What are the best-rated colleges for computer science?"
        - "List colleges for IT courses with fee under 50000"
        """)

if __name__ == "__main__":
    main()