import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CollegeChatbot:
    def __init__(self):
        self.colleges_data = []
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Intent patterns
        self.intent_patterns = {
            'find_colleges': [
                r'recommend.*colleges?.*in\s+(\w+)',
                r'colleges?.*in\s+(\w+)',
                r'find.*colleges?.*(\w+)',
                r'suggest.*colleges?.*in\s+(\w+)',
                r'best.*colleges?.*in\s+(\w+)'
            ],
            'course_specific': [
                r'colleges?.*offer.*(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*).*course',
                r'study.*(\w+(?:\s+\w+)*)',
                r'colleges?.*for.*(\w+(?:\s+\w+)*)'
            ],
            'rating_based': [
                r'good\s+rating',
                r'best\s+rated',
                r'high\s+rating',
                r'top\s+rated',
                r'(\d+(?:\.\d+)?)\s*(?:star|rating)'
            ],
            'fee_based': [
                r'fee.*(\d+)',
                r'cost.*(\d+)',
                r'cheap.*college',
                r'affordable.*college',
                r'budget.*(\d+)'
            ],
            'approval_based': [
                r'approval.*from\s+(\w+)',
                r'approved.*by\s+(\w+)',
                r'(\w+)\s+approved'
            ]
        }
    
    def load_data(self, data):
        """Load college data from JSON"""
        if isinstance(data, str):
            with open(data, 'r') as f:
                data_dict = json.load(f)
        else:
            data_dict = data
        
        # Convert dictionary format to list format
        self.colleges_data = []
        for college_id, college_info in data_dict.items():
            # Add the college ID to the college info
            college_info['college_id'] = college_id
            self.colleges_data.append(college_info)
        
        # Create searchable text for each college
        self.create_search_corpus()
    
    def create_search_corpus(self):
        """Create searchable text corpus for each college"""
        self.search_corpus = []
        for college in self.colleges_data:
            # Combine all searchable fields safely
            text_parts = []
            
            # Add basic string fields
            text_parts.append(college.get('college_name', ''))
            text_parts.append(college.get('city', ''))
            text_parts.append(college.get('state', ''))
            text_parts.append(college.get('approvals', ''))
            text_parts.append(college.get('college_type', ''))
            text_parts.append(college.get('positive_notes', ''))
            
            # Add streams (handle list safely)
            streams = college.get('streams', [])
            if isinstance(streams, list):
                text_parts.extend(streams)
            elif isinstance(streams, str):
                text_parts.append(streams)
            
            # Add courses
            courses = college.get('courses', [])
            if isinstance(courses, list):
                for course in courses:
                    if isinstance(course, dict):
                        text_parts.append(course.get('course_name', ''))
                    elif isinstance(course, str):
                        text_parts.append(course)
            
            # Filter out empty strings and non-string items
            text_parts = [str(part) for part in text_parts if part]
            
            combined_text = ' '.join(text_parts).lower()
            self.search_corpus.append(combined_text)
        
        # Fit vectorizer
        if self.search_corpus:
            self.corpus_vectors = self.vectorizer.fit_transform(self.search_corpus)
    
    def preprocess_query(self, query):
        """Preprocess user query"""
        query = query.lower().strip()
        tokens = word_tokenize(query)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens), tokens
    
    def extract_entities(self, query):
        """Extract entities from query using regex patterns"""
        entities = {
            'city': None,
            'course': None,
            'rating': None,
            'fee': None,
            'approval': None,
            'intent': []
        }
        
        # Extract city names (basic approach - you can enhance this)
        city_pattern = r'in\s+([A-Z][a-zA-Z\s]+?)(?:\s+with|\s+that|\s+for|$)'
        city_match = re.search(city_pattern, query, re.IGNORECASE)
        if city_match:
            entities['city'] = city_match.group(1).strip()
        
        # Extract course names
        course_patterns = [
            r'offer(?:ing)?\s+([A-Z][a-zA-Z\s&]+?)(?:\s+in|\s+with|$)',
            r'for\s+([A-Z][a-zA-Z\s&]+?)(?:\s+in|\s+with|$)',
            r'study\s+([A-Z][a-zA-Z\s&]+?)(?:\s+in|\s+with|$)'
        ]
        for pattern in course_patterns:
            course_match = re.search(pattern, query, re.IGNORECASE)
            if course_match:
                entities['course'] = course_match.group(1).strip()
                break
        
        # Extract ratings
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:star|rating)',
            r'rating.*?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:to|\-)\s*(\d+(?:\.\d+)?)\s*(?:star|rating)'
        ]
        for pattern in rating_patterns:
            rating_match = re.search(pattern, query, re.IGNORECASE)
            if rating_match:
                entities['rating'] = float(rating_match.group(1))
                break
        
        # Extract fee information
        fee_match = re.search(r'fee.*?(\d+)', query, re.IGNORECASE)
        if fee_match:
            entities['fee'] = int(fee_match.group(1))
        
        # Extract approval bodies
        approval_match = re.search(r'approval.*?from\s+(\w+)', query, re.IGNORECASE)
        if approval_match:
            entities['approval'] = approval_match.group(1).upper()
        
        return entities
    
    def filter_colleges(self, entities, query_vector):
        """Filter colleges based on extracted entities"""
        filtered_colleges = []
        
        for i, college in enumerate(self.colleges_data):
            score = 0
            match_reasons = []
            should_include = True
            
            # If no specific filters, use semantic search
            has_specific_filters = any([entities['city'], entities['course'], entities['rating'], entities['approval'], entities['fee']])
            
            # City filter - flexible matching
            if entities['city']:
                city_match = False
                college_city = college.get('city', '').lower()
                college_state = college.get('state', '').lower()
                search_city = entities['city'].lower()
                
                if (search_city in college_city or 
                    search_city in college_state or
                    college_city in search_city):
                    score += 0.25
                    match_reasons.append(f"📍 Located in {college.get('city')}")
                    city_match = True
                
                # If city specified but no match, still include but with lower score
                if not city_match and has_specific_filters:
                    score -= 0.1
            
            # Course filter - comprehensive matching
            if entities['course']:
                course_found = False
                search_course = entities['course'].lower()
                
                for course in college.get('courses', []):
                    course_name = course.get('course_name', '').lower()
                    if (search_course in course_name or 
                        any(word in course_name for word in search_course.split()) or
                        course_name in search_course):
                        score += 0.3
                        match_reasons.append(f"📚 Offers {course.get('course_name')}")
                        course_found = True
                        break
                
                # Check streams as well
                streams = college.get('streams', [])
                if not course_found and isinstance(streams, list):
                    for stream in streams:
                        if stream and search_course in str(stream).lower():
                            score += 0.25
                            match_reasons.append(f"🎯 Stream: {stream}")
                            course_found = True
                            break
                
                # If specific course requested but not found, significantly reduce score
                if not course_found:
                    score -= 0.2
            
            # Rating filter - flexible threshold
            if entities['rating']:
                college_rating = college.get('rating_value', 0)
                if isinstance(college_rating, (int, float)):
                    if college_rating >= entities['rating']:
                        score += 0.2
                        match_reasons.append(f"⭐ Rating: {college_rating}/5")
                    elif college_rating >= entities['rating'] - 0.5:  # Close enough
                        score += 0.1
                        match_reasons.append(f"⭐ Rating: {college_rating}/5 (close match)")
            
            # Fee filter
            if entities['fee']:
                fee_found = False
                for course in college.get('courses', []):
                    course_fee = course.get('fee_amount', 0)
                    if isinstance(course_fee, (int, float)) and course_fee > 0:
                        if course_fee <= entities['fee']:
                            score += 0.15
                            match_reasons.append(f"💰 Affordable course: {course.get('course_name')} - ₹{course_fee:,}")
                            fee_found = True
                            break
                
                if not fee_found:
                    score -= 0.1
            
            # Approval filter - flexible matching
            if entities['approval']:
                approvals = college.get('approvals', '').upper()
                search_approval = entities['approval'].upper()
                
                if (search_approval in approvals or 
                    any(search_approval in approval.strip() for approval in approvals.split(','))):
                    score += 0.15
                    match_reasons.append(f"✅ Approved by {entities['approval']}")
                else:
                    score -= 0.05
            
            # Semantic similarity - always include for general queries
            if query_vector is not None:
                similarity = cosine_similarity(query_vector, self.corpus_vectors[i:i+1])[0][0]
                score += similarity * 0.2
                
                if similarity > 0.1:
                    match_reasons.append(f"🔍 Semantic match (similarity: {similarity:.2f})")
            
            # Quality bonus - prefer higher rated colleges
            college_rating = college.get('rating_value', 0)
            if isinstance(college_rating, (int, float)) and college_rating > 0:
                score += (college_rating / 5) * 0.1  # Bonus based on rating
            
            # Include college if it has any positive score or matches semantic search
            min_threshold = 0.05 if has_specific_filters else 0.02
            if score > min_threshold:
                filtered_colleges.append({
                    'college': college,
                    'score': score,
                    'match_reasons': match_reasons
                })
        
        # Sort by score
        filtered_colleges.sort(key=lambda x: x['score'], reverse=True)
        return filtered_colleges[:8]  # Return top 8 for better variety
    
    def format_response(self, filtered_colleges, query):
        """Format the response for display with collapsible sections"""
        if not filtered_colleges:
            return "I couldn't find any colleges matching your criteria. Please try with different parameters."
        
        # Start with a summary
        response = f"🎯 **Found {len(filtered_colleges)} colleges matching your criteria:**\n\n"
        
        for i, item in enumerate(filtered_colleges, 1):
            college = item['college']
            reasons = item['match_reasons']
            match_score = item['score']
            
            # Create collapsible section with summary
            college_name = college.get('college_name', 'N/A')
            city = college.get('city', 'N/A')
            rating = college.get('rating_value', 'N/A')
            
            response += f"### {i}. {college_name}\n"
            response += f"**📍 {city} | ⭐ {rating}/5 | 🎯 Match Score: {match_score:.2f}**\n\n"
            
            # Quick highlights
            if reasons:
                response += f"**🎯 Why this matches:** {', '.join(reasons)}\n\n"
            
            # Detailed information in expandable format
            response += f"<details>\n"
            response += f"<summary><strong>📋 View Complete Details</strong></summary>\n\n"
            
            # Basic Information
            response += f"**🏫 Basic Information:**\n"
            response += f"- **College ID:** {college.get('college_id', 'N/A')}\n"
            response += f"- **Location:** {college.get('city', 'N/A')}, {college.get('state', 'N/A')}\n"
            response += f"- **Type:** {college.get('college_type', 'Not specified')}\n"
            response += f"- **Website:** {college.get('website', 'N/A')}\n"
            response += f"- **Phone:** {college.get('phone', 'N/A')}\n"
            response += f"- **Email:** {college.get('email', 'N/A')}\n\n"
            
            # Ratings & Reviews
            response += f"**⭐ Ratings & Reviews:**\n"
            response += f"- **Overall Rating:** {college.get('rating_value', 'N/A')}/5\n"
            response += f"- **Review Count:** {college.get('review_count', 'N/A')}\n"
            response += f"- **Score:** {college.get('score', 'N/A')}\n"
            response += f"- **Rating Range:** {college.get('worst_rating', 'N/A')} - {college.get('best_rating', 'N/A')}\n\n"
            
            # Approvals & Accreditation
            response += f"**✅ Approvals & Accreditation:**\n"
            response += f"- **Approved by:** {college.get('approvals', 'N/A')}\n\n"
            
            # Courses & Fees
            courses = college.get('courses', [])
            if courses:
                response += f"**📚 Courses & Fees:**\n"
                for j, course in enumerate(courses, 1):
                    course_name = course.get('course_name', 'N/A')
                    fee = course.get('fee_amount', 'N/A')
                    if isinstance(fee, (int, float)) and fee > 0:
                        response += f"{j}. **{course_name}** - ₹{fee:,}\n"
                    else:
                        response += f"{j}. **{course_name}** - Fee: {fee}\n"
                response += "\n"
            else:
                response += f"**📚 Courses:** No course information available\n\n"
            
            # Streams
            streams = college.get('streams', [])
            if streams and isinstance(streams, list):
                valid_streams = [stream for stream in streams if stream and str(stream).strip()]
                if valid_streams:
                    response += f"**🎯 Available Streams:**\n"
                    for stream in valid_streams:
                        response += f"- {stream}\n"
                    response += "\n"
            
            # Additional Notes
            if college.get('positive_notes'):
                response += f"**💡 Additional Notes:**\n"
                response += f"{college.get('positive_notes')}\n\n"
            
            # Cutoffs (if available)
            cutoffs = college.get('cutoffs', [])
            if cutoffs:
                response += f"**📊 Cutoffs:** Available\n\n"
            
            response += f"</details>\n\n"
            response += "---\n\n"
        
        # Add helpful tips at the end
        response += "💡 **Tips:**\n"
        response += "- Click on 'View Complete Details' to see full information\n"
        response += "- Colleges are ranked by match score (higher is better)\n"
        response += "- Try different search terms for more results\n"
        
        return response
    
    def process_query(self, query):
        """Main method to process user query"""
        # Preprocess query
        processed_query, tokens = self.preprocess_query(query)
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Create query vector for semantic search
        try:
            query_vector = self.vectorizer.transform([processed_query])
        except:
            query_vector = None
        
        # Filter colleges
        filtered_colleges = self.filter_colleges(entities, query_vector)
        
        # Format response
        response = self.format_response(filtered_colleges, query)
        
        return response, entities

# Streamlit App
def main():
    st.set_page_config(
        page_title="College Recommendation Chatbot",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 College Recommendation Chatbot")
    st.markdown("Ask me about colleges, courses, ratings, and more!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CollegeChatbot()
        
        # Sample data (replace with your actual data loading)
        sample_data = {
            "25446": {
                "college_name": "All India Institute of Medical Sciences - [AIIMS]",
                "streams": [],
                "website": "http://www.aiims.edu",
                "phone": "011-26588500",
                "college_type": "",
                "email": "director@aiims.edu",
                "rating_value": 4.3,
                "review_count": "245",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "",
                "courses": [
                    {"course_name": "Operation Theatre Technology", "fee_amount": 3385},
                    {"course_name": "Medical Radiology & Imaging Technology", "fee_amount": 4480},
                    {"course_name": "MBBS", "fee_amount": 50000}
                ],
                "cutoffs": [],
                "city": "New Delhi",
                "state": "Delhi NCR",
                "rating": "10.0",
                "score": 4.3,
                "approvals": "MCI, UGC"
            },
            "25703": {
                "college_name": "IIT Bombay - Indian Institute of Technology - [IITB]",
                "streams": [],
                "website": "http://www.iitb.ac.in",
                "phone": "022-25722545",
                "college_type": "",
                "email": "gateoffice@iitb.ac.in",
                "rating_value": 4.4,
                "review_count": "397",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "",
                "courses": [
                    {"course_name": "Computer Science Engineering", "fee_amount": 200000},
                    {"course_name": "Mechanical Engineering", "fee_amount": 200000},
                    {"course_name": "Electrical Engineering", "fee_amount": 200000}
                ],
                "cutoffs": [],
                "city": "Mumbai",
                "state": "Maharashtra",
                "rating": "10.0",
                "score": 4.4,
                "approvals": "AICTE, UGC"
            }
        }
        
        st.session_state.chatbot.load_data(sample_data)
    
    # File upload section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your college dataset (JSON)", type=['json'])
    
    if uploaded_file is not None:
        try:
            data_dict = json.load(uploaded_file)
            st.session_state.chatbot.load_data(data_dict)
            st.sidebar.success(f"✅ Loaded {len(data_dict)} colleges successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Example queries
    st.sidebar.header("💡 Example Queries")
    example_queries = [
        "Can you recommend colleges in New Delhi with good ratings that offer Medical courses?",
        "What are the best-rated colleges in Mumbai for Engineering?",
        "Find colleges offering MBBS in Delhi with 4+ rating",
        "Show me affordable colleges under 50000 fees in Maharashtra",
        "Which colleges in Delhi are approved by UGC and offer Computer Science?",
        "Find colleges with Operation Theatre Technology course and good ratings",
        "Show me IIT colleges with engineering courses",
        "Best colleges in Chennai for MBA with fees under 100000"
    ]
    
    for query in example_queries:
        if st.sidebar.button(query, key=f"example_{hash(query)}"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Process query
            response, entities = st.session_state.chatbot.process_query(query)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me about colleges..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching for colleges..."):
                response, entities = st.session_state.chatbot.process_query(prompt)
                
                # Use HTML rendering for collapsible sections
                st.markdown(response, unsafe_allow_html=True)
                
                # Show extracted entities in sidebar for debugging
                if st.sidebar.checkbox("🔍 Show Query Analysis"):
                    st.sidebar.json(entities)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Statistics
    if st.session_state.chatbot.colleges_data:
        st.sidebar.header("📊 Dataset Statistics")
        total_colleges = len(st.session_state.chatbot.colleges_data)
        
        # Calculate some basic stats
        cities = set()
        courses = set()
        streams = set()
        avg_rating = 0
        total_rating_count = 0
        
        for college in st.session_state.chatbot.colleges_data:
            if college.get('city'):
                cities.add(college['city'])
            
            # Handle courses safely
            college_courses = college.get('courses', [])
            if isinstance(college_courses, list):
                for course in college_courses:
                    if isinstance(course, dict) and course.get('course_name'):
                        courses.add(course['course_name'])
            
            # Handle streams safely
            college_streams = college.get('streams', [])
            if isinstance(college_streams, list):
                for stream in college_streams:
                    if stream and str(stream).strip():
                        streams.add(str(stream))
            
            # Handle rating safely
            rating = college.get('rating_value')
            if rating and isinstance(rating, (int, float)):
                avg_rating += rating
                total_rating_count += 1
        
        avg_rating = avg_rating / total_rating_count if total_rating_count > 0 else 0
        
        st.sidebar.metric("Total Colleges", total_colleges)
        st.sidebar.metric("Cities Covered", len(cities))
        st.sidebar.metric("Unique Courses", len(courses))
        st.sidebar.metric("Unique Streams", len(streams))
        st.sidebar.metric("Average Rating", f"{avg_rating:.2f}")

if __name__ == "__main__":
    main()