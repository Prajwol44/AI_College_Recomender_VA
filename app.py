import streamlit as st
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Any, Union
import os

BASE_PATH = 'data/'

# Initialize the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
    # return SentenceTransformer("BAAI/bge-base-en-v1.5")

@st.cache_data
def load_college_data(file_path= BASE_PATH + 'final_college_dataset.json'):
    """Load college data from JSON file and convert ratings to floats"""
    try:
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                for college in data_dict.values():
                    try:
                        college['rating_value'] = float(college['rating_value'])
                    except (ValueError, TypeError):
                        college['rating_value'] = 0.0

            st.success(f"College data recieved. You may beign your search.")
        print(f"load_college_data {len(data_dict.keys())}")
        return data_dict
    except Exception as e:
        st.error(f" Error loading data: {str(e)}")
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
            # break 
            
           
           #this is the perviously used description
           
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
        
        # Extract city names (common Indian cities)
        cities = [
            'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune',
            'delhi', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur',
            'nagpur', 'visakhapatnam', 'bhopal', 'patna', 'vadodara', 'indore',
            'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot',
            'varanasi', 'amritsar', 'ranchi', 'coimbatore', 'guwahati', 'allahabad'
        ]

        for city in cities:
            if city.lower() in query.strip().lower():
                params['city'] = city.title()
                break
        
        # Extract course names
        courses = [
            'computer science', 'engineering', 'medical', 'technology', 'it',
            'radiology', 'operation theatre', 'nursing', 'pharmacy', 'dentistry',
            'law', 'business administration', 'commerce', 'economics', 'accounting',
            'finance', 'architecture', 'civil engineering', 'mechanical engineering',
            'electrical engineering', 'aerospace engineering', 'data science',
            'artificial intelligence', 'psychology', 'sociology', 'english literature',
            'history', 'political science', 'mass communication', 'hotel management',
            'fashion designing', 'graphic designing', 'animation', 'agriculture',
            'veterinary science', 'education', 'physical education', 'biotechnology'
        ]

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
            r'fee limit.*?(\d+)',
            r'fee not more than.*?(\d+)',
            r'cost.*?under.*?(\d+)',
            r'cost.*?less than.*?(\d+)',
            r'budget.*?(\d+).*?for fee',
            r'courses.*?within.*?(\d+)',
            r'fees up to.*?(\d+)',
            r'fees not exceeding.*?(\d+)',
            r'charge.*?up to.*?(\d+)',
            r'price less than.*?(\d+)',
            r'courses.*?under.*?(\d+)',
            r'pay only.*?(\d+)',
            r'max.*?(\d+).*?fee',
            r'around.*?(\d+).*?fee',
            r'fee approx.*?(\d+)',
            r'less than rupees.*?(\d+)',
            r'fee.*?no more than.*?(\d+)'
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
        
        print(f"extract_query_params: {query}")
        
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
                affordable_courses = [
                    course for course in college.get('courses', [])
                    if str(course.get('fee_amount', '')).isdigit()
                    and int(course['fee_amount']) <= int(params['fee_limit'])
                ]
                if not affordable_courses:
                    continue
            
            filtered_ids.append(college_id)
        
        return filtered_ids
    
    def handle_general_queries(self, query: str) -> Union[str, None]:
        q = query.lower().strip()

        greetings = ["hello", "hi", "namaste", "hey"]
        if any(greet in q for greet in greetings):
            return "Namaste! I'm your College Recommender. How can I assist you today?"

        if "how are you" in q:
            return "I'm doing great! Ready to help you find the best colleges across India. What are you looking for?"

        help_phrases = [
            "what can you help",
            "what do you do",
            "help me",
            "what can you help me with"
        ]
        if any(phrase in q for phrase in help_phrases):
            return (
                "I can help you with:\n\n"
                "- Recommending colleges based on your interests\n"
                "- Filter by city or state\n"
                "- Find specific courses (e.g., engineering, medical, IT, etc.)\n"
                "- Budget-based college search\n"
                "- Filter by approvals like MCI, UGC, AICTE\n"
                "- Ratings-based recommendations\n\n"
                "Try asking something like:\n"
                "`Top engineering colleges in Pune under ₹1 lakh`\n"
                "or\n"
                "`Best rated medical colleges in Delhi`"
            )

        return None


    def search_colleges(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for colleges based on user query"""
        if self.model is None or self.embeddings is None:
            print("Restart the system")
            return " Model not loaded. Please try again later."
        
        query = query.strip()
        
        #Handle general responses
        general_response = self.handle_general_queries(query)
        if general_response is not None:
            return general_response
        
        
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
        #     filtered_similarities = [(i, sim) for i, sim in enumerate(similarities)] # orig
        
        filtered_similarities = [(i, sim) for i, sim in enumerate(similarities)] # added

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
    """Display a college in a cleaner, compact card format"""
    with st.container():
        st.markdown(f"### 🎓 {college['college_name']}")
        
        col1, col2 = st.columns([3, 1])

        # Left Column - Info
        with col1:
            st.markdown(
                f"""
                 **Location:** {college['city']}, {college['state']}  
                 **Rating:** {college['rating_value']}/5.0  
                 **Approvals:** {college.get('approvals', 'N/A')}  
                 **Phone:** {", ".join(college.get('phone', [])) if isinstance(college.get('phone'), list) else college.get('phone', 'N/A')}  
                 **Email:** {college.get('email', 'N/A')}  
                 **Website:** [{college.get('website', 'N/A')}]({college.get('website', '#')})
                """,
                unsafe_allow_html=True
            )

        # Right Column - Score
        with col2:
            st.metric(label="Similarity Score", value=f"{college['similarity_score']:.3f}")

        # Courses Table
        if college.get('courses'):
            st.markdown("**📚 Available Courses:**")
            # Safely format fee
            course_data = {
                "Course": [course['course_name'] for course in college['courses']],
                "Fee (₹)": [
                    f"₹{int(course['fee_amount']):,}" if str(course['fee_amount']).isdigit()
                    else course['fee_amount']
                    for course in college['courses']
                ]
            }
            course_df = pd.DataFrame(course_data)
            st.dataframe(course_df, use_container_width=True, hide_index=True)

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
    st.set_page_config(page_title="Zenthra - College Recommender", page_icon="🎓", layout="wide")
    st.title("Zenthra - Your Personal College Recommender")
    st.markdown("Find your perfect college and start your journey!")

    # Load model and data
    model = load_model()
    college_data = load_college_data()

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CollegeChatbot(college_data, model)

    # Sidebar with sample queries
    with st.sidebar:
        st.header("📝 Sample Queries")
        sample_queries = [
            "What can you help me with?",
            "Engineering colleges with good ratings",
            "IT courses under ₹50000 fee",
            "Colleges with MCI approval",
            "Top rated colleges in Mumbai",
            "Best medical colleges in Delhi",
            "Engineering colleges in Mumbai",
            "Computer science courses in Bangalore",
            "Top rated colleges for MBBS",
            "IT engineering colleges in Pune",
            "Dental colleges in Chennai",
        ]

        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.user_query = query

    st.divider()

    # Container for chat-like input box at the bottom
    with st.form(key="college_query_form", clear_on_submit=False):
        col1, col2 = st.columns([18, 1])  # Input box and button side-by-side

        with col1:
            user_query = st.text_input(
                "Ask me about colleges:", 
                value=st.session_state.get('user_query', ''), 
                placeholder="Hello, Start Searching for Collges", 
                label_visibility="collapsed",  # Hide label like a chat
                key="query_input"
            )

        with col2:
            submitted = st.form_submit_button("⬆")

    if submitted and user_query.strip():
        st.session_state.user_query = user_query.strip()

        with st.spinner("Searching for colleges..."):
            results = st.session_state.chatbot.search_colleges(user_query.strip(), top_k=5)

            if isinstance(results, str):
                st.info(results)

            elif results:
                st.success(f"Found {len(results)} relevant colleges!")

                st.header("Recommended Colleges")
                for college in results:
                    display_college_card(college)

                similarities = [college['similarity_score'] for college in results]
                college_names = [
                    college['college_name'][:30] + "..." if len(college['college_name']) > 30 
                    else college['college_name']
                    for college in results
                ]

                # st.header("📊 Similarity Analysis")
                # fig = create_similarity_plot(similarities, college_names)
                # st.plotly_chart(fig, use_container_width=True)

                st.header("🔍 Search Insights")
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_rating = np.mean([
                        float(college['rating_value']) 
                        for college in results 
                        if isinstance(college['rating_value'], (int, float, str)) and str(college['rating_value']).replace('.', '', 1).isdigit()
                    ])
                    st.metric("Average Rating", f"{avg_rating:.2f}/5.0")

                with col2:
                    avg_similarity = np.mean(similarities)
                    st.metric("Average Similarity", f"{avg_similarity:.3f}")

                with col3:
                    total_courses = sum(len(college.get('courses', [])) for college in results)
                    st.metric("Total Courses", total_courses)

            else:
                st.warning("No colleges found matching your criteria. Try a different query!")


if __name__ == "__main__":
    main()