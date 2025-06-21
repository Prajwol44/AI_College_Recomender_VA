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
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set page config - MUST BE FIRST AND ONLY CALL
st.set_page_config(
    page_title="Zenthra - College Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Custom CSS for better UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --card-background: #FFFFFF;
        --border-color: #E1E8ED;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Chat interface styling */
    .chat-container {
        background: var(--card-background);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    /* College card styling */
    .college-card {
        background: var(--card-background);
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 15px 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .college-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .college-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    .college-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .college-rating {
        background: linear-gradient(135deg, var(--accent-color), #FF6B35);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .college-location {
        color: var(--text-color);
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .match-reasons {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(162, 59, 114, 0.1));
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .course-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .course-item {
        background: rgba(46, 134, 171, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(46, 134, 171, 0.2);
        transition: all 0.3s ease;
    }
    
    .course-item:hover {
        background: rgba(46, 134, 171, 0.1);
        transform: translateY(-1px);
    }
    
    .course-name {
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .course-fee {
        color: var(--secondary-color);
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Example query buttons */
    .example-query {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(162, 59, 114, 0.1));
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .example-query:hover {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        transform: translateX(5px);
    }
    
    /* Statistics cards */
    .stat-card {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .stat-label {
        color: var(--text-color);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Loading animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid rgba(46, 134, 171, 0.1);
        border-left: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Alert boxes */
    .alert-info {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(162, 59, 114, 0.1));
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(241, 143, 1, 0.1), rgba(255, 107, 53, 0.1));
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .college-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        
        .course-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    </style>
    """, unsafe_allow_html=True)

class ZenthraChatbot:
    def __init__(self):
        self.colleges_data = []
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced intent patterns
        self.intent_patterns = {
            'find_colleges': [
                r'recommend.*colleges?.*in\s+(\w+(?:\s+\w+)*)',
                r'colleges?.*in\s+(\w+(?:\s+\w+)*)',
                r'find.*colleges?.*in\s+(\w+(?:\s+\w+)*)',
                r'suggest.*colleges?.*in\s+(\w+(?:\s+\w+)*)',
                r'best.*colleges?.*in\s+(\w+(?:\s+\w+)*)',
                r'top\s+colleges?.*in\s+(\w+(?:\s+\w+)*)'
            ],
            'course_specific': [
                r'colleges?.*offer.*(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*).*course',
                r'study.*(\w+(?:\s+\w+)*)',
                r'colleges?.*for.*(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*).*program',
                r'(\w+(?:\s+\w+)*).*degree'
            ],
            'rating_based': [
                r'good\s+rating',
                r'best\s+rated',
                r'high\s+rating',
                r'top\s+rated',
                r'(\d+(?:\.\d+)?)\s*(?:star|rating)',
                r'rating\s+above\s+(\d+(?:\.\d+)?)',
                r'rating\s+at\s+least\s+(\d+(?:\.\d+)?)'
            ],
            'fee_based': [
                r'fee.*?(\d[\d,.]*)',
                r'cost.*?(\d[\d,.]*)',
                r'cheap.*college',
                r'affordable.*college',
                r'budget.*?(\d[\d,.]*)',
                r'fees?\s+under\s+(\d[\d,.]*)',
                r'fees?\s+less\s+than\s+(\d[\d,.]*)'
            ],
            'approval_based': [
                r'approval.*from\s+(\w+)',
                r'approved.*by\s+(\w+)',
                r'(\w+)\s+approved',
                r'accreditation.*by\s+(\w+)',
                r'(\w+)\s+accredited'
            ],
            'stream_based': [
                r'streams?.*in\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*).*stream',
                r'programs?.*in\s+(\w+(?:\s+\w+)*)'
            ]
        }
        
        # Enhanced course aliases with more comprehensive matching
        self.course_aliases = {
            'engineering': ['engineering', 'engg', 'b.tech', 'btech', 'be', 'b.e', 'technical'],
            'medical': ['medical', 'mbbs', 'bds', 'bams', 'bhms', 'bums', 'medicine', 'doctor'],
            'management': ['management', 'mba', 'pgdm', 'bms', 'bba', 'business', 'finance', 'marketing'],
            'computer science': ['computer science', 'cs', 'cse', 'it', 'information technology', 'software', 'programming'],
            'commerce': ['commerce', 'b.com', 'bcom', 'accounting', 'finance'],
            'arts': ['arts', 'ba', 'b.a', 'humanities', 'literature'],
            'science': ['science', 'b.sc', 'bsc', 'physics', 'chemistry', 'biology', 'mathematics'],
            'design': ['design', 'fashion', 'interior', 'graphic', 'product'],
            'law': ['law', 'llb', 'legal', 'advocate', 'judiciary'],
            'pharmacy': ['pharmacy', 'b.pharm', 'pharm.d', 'pharmaceutical']
        }
    
    def load_data_from_file(self, file_path='final_college_dataset.json'):
        """Load college data from JSON file and convert ratings to floats"""
        try:
            # Try multiple possible locations for the file
            possible_paths = [
                file_path,
                f'./{file_path}',
                f'../{file_path}',
                f'./data/{file_path}',
                f'../data/{file_path}'
            ]
            
            data_dict = None
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data_dict = json.load(f)
                    st.success(f"✅ Successfully loaded data from {path}")
                    break
            
            if data_dict is None:
                # Fallback to sample data if file not found
                st.warning("⚠️ final_college_dataset.json not found. Using sample data.")
                data_dict = self.get_sample_data()
            
            # Convert dictionary format to list format and ensure ratings are floats
            self.colleges_data = []
            for college_id, college_info in data_dict.items():
                # Convert rating_value to float if it's a string
                if 'rating_value' in college_info:
                    try:
                        college_info['rating_value'] = float(college_info['rating_value'])
                    except (ValueError, TypeError):
                        college_info['rating_value'] = 0.0
                else:
                    college_info['rating_value'] = 0.0
                
                college_info['college_id'] = college_id
                self.colleges_data.append(college_info)
            
            # Create searchable text for each college
            self.create_search_corpus()
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            # Use sample data as fallback
            self.colleges_data = []
            for college_id, college_info in self.get_sample_data().items():
                college_info['college_id'] = college_id
                self.colleges_data.append(college_info)
            self.create_search_corpus()
            return False
    
    def get_sample_data(self):
        """Enhanced sample data with more diverse colleges"""
        return {
            "25446": {
                "college_name": "All India Institute of Medical Sciences - [AIIMS]",
                "streams": ["Medical"],
                "website": "http://www.aiims.edu",
                "phone": "011-26588500",
                "college_type": "Government",
                "email": "director@aiims.edu",
                "rating_value": 4.8,
                "review_count": "245",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Premier medical institute with world-class facilities and renowned faculty",
                "courses": [
                    {"course_name": "MBBS", "fee_amount": 50000},
                    {"course_name": "MD - General Medicine", "fee_amount": 60000},
                    {"course_name": "MS - General Surgery", "fee_amount": 60000},
                    {"course_name": "DM - Cardiology", "fee_amount": 70000}
                ],
                "cutoffs": ["NEET: 99.5+ percentile"],
                "city": "New Delhi",
                "state": "Delhi",
                "rating": "10.0",
                "score": 4.8,
                "approvals": "MCI, UGC, NAAC A+"
            },
            "25703": {
                "college_name": "IIT Bombay - Indian Institute of Technology - [IITB]",
                "streams": ["Engineering", "Technology", "Science"],
                "website": "http://www.iitb.ac.in",
                "phone": "022-25722545",
                "college_type": "Government",
                "email": "gateoffice@iitb.ac.in",
                "rating_value": 4.7,
                "review_count": "397",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Top engineering institute with excellent placement records and research facilities",
                "courses": [
                    {"course_name": "Computer Science Engineering", "fee_amount": 200000},
                    {"course_name": "Mechanical Engineering", "fee_amount": 200000},
                    {"course_name": "Electrical Engineering", "fee_amount": 200000},
                    {"course_name": "Chemical Engineering", "fee_amount": 200000},
                    {"course_name": "Civil Engineering", "fee_amount": 200000}
                ],
                "cutoffs": ["JEE Advanced: Top 500 rank"],
                "city": "Mumbai",
                "state": "Maharashtra",
                "rating": "10.0",
                "score": 4.7,
                "approvals": "AICTE, UGC, NBA"
            },
            "25704": {
                "college_name": "St. Xavier's College",
                "streams": ["Arts", "Science", "Commerce"],
                "website": "http://www.xaviers.edu",
                "phone": "022-22620661",
                "college_type": "Private",
                "email": "principal@xaviers.edu",
                "rating_value": 4.5,
                "review_count": "286",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Prestigious college with strong academic reputation and excellent faculty",
                "courses": [
                    {"course_name": "B.Sc Computer Science", "fee_amount": 75000},
                    {"course_name": "B.Com", "fee_amount": 65000},
                    {"course_name": "B.A. Economics", "fee_amount": 60000},
                    {"course_name": "B.Sc Physics", "fee_amount": 70000},
                    {"course_name": "B.A. English Literature", "fee_amount": 55000}
                ],
                "cutoffs": ["HSC: 90%+"],
                "city": "Mumbai",
                "state": "Maharashtra",
                "rating": "9.5",
                "score": 4.5,
                "approvals": "UGC, NAAC A"
            },
            "25705": {
                "college_name": "Indian Institute of Management - [IIM Ahmedabad]",
                "streams": ["Management", "Business"],
                "website": "http://www.iima.ac.in",
                "phone": "079-66323456",
                "college_type": "Government",
                "email": "director@iima.ac.in",
                "rating_value": 4.9,
                "review_count": "324",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Premier B-school with 100% placement record and top corporate tie-ups",
                "courses": [
                    {"course_name": "MBA", "fee_amount": 2500000},
                    {"course_name": "PGP - Management", "fee_amount": 2400000},
                    {"course_name": "Executive MBA", "fee_amount": 1800000},
                    {"course_name": "FPM - Doctoral Program", "fee_amount": 800000}
                ],
                "cutoffs": ["CAT: 99.5+ percentile"],
                "city": "Ahmedabad",
                "state": "Gujarat",
                "rating": "10.0",
                "score": 4.9,
                "approvals": "AICTE, UGC, AACSB"
            },
            "25706": {
                "college_name": "National Institute of Fashion Technology - [NIFT Delhi]",
                "streams": ["Design", "Fashion", "Technology"],
                "website": "http://www.nift.ac.in",
                "phone": "011-26542100",
                "college_type": "Government",
                "email": "director@nift.ac.in",
                "rating_value": 4.3,
                "review_count": "187",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Leading fashion institute with industry connections and modern facilities",
                "courses": [
                    {"course_name": "Fashion Design", "fee_amount": 350000},
                    {"course_name": "Textile Design", "fee_amount": 320000},
                    {"course_name": "Fashion Communication", "fee_amount": 300000},
                    {"course_name": "Fashion Technology", "fee_amount": 330000}
                ],
                "cutoffs": ["NIFT Entrance Exam: Top 1000 rank"],
                "city": "New Delhi",
                "state": "Delhi",
                "rating": "9.0",
                "score": 4.3,
                "approvals": "UGC, AICTE"
            },
            "25707": {
                "college_name": "Jamia Millia Islamia University",
                "streams": ["Engineering", "Arts", "Science", "Management"],
                "website": "http://www.jmi.ac.in",
                "phone": "011-26981717",
                "college_type": "Government",
                "email": "registrar@jmi.ac.in",
                "rating_value": 4.2,
                "review_count": "423",
                "worst_rating": 0,
                "best_rating": 5,
                "positive_notes": "Central university with diverse courses and good placement record",
                "courses": [
                    {"course_name": "Computer Science Engineering", "fee_amount": 120000},
                    {"course_name": "MBA", "fee_amount": 180000},
                    {"course_name": "B.A. English", "fee_amount": 45000},
                    {"course_name": "Civil Engineering", "fee_amount": 115000},
                    {"course_name": "Mass Communication", "fee_amount": 95000}
                ],
                "cutoffs": ["JEE Main: 85+ percentile"],
                "city": "New Delhi",
                "state": "Delhi",
                "rating": "8.5",
                "score": 4.2,
                "approvals": "UGC, AICTE, NAAC A"
            }
        }
    
    def create_search_corpus(self):
        """Create searchable text corpus for each college"""
        self.search_corpus = []
        for college in self.colleges_data:
            text_parts = []
            
            # Add basic string fields
            text_parts.append(college.get('college_name', ''))
            text_parts.append(college.get('city', ''))
            text_parts.append(college.get('state', ''))
            text_parts.append(college.get('approvals', ''))
            text_parts.append(college.get('college_type', ''))
            text_parts.append(college.get('positive_notes', ''))
            
            # Add streams
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
            
            text_parts = [str(part) for part in text_parts if part]
            combined_text = ' '.join(text_parts).lower()
            self.search_corpus.append(combined_text)
        
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
            'stream': None,
            'intent': []
        }
        
        # Extract city names
        city_pattern = r'in\s+([a-zA-Z\s]+?)(?:\s+with|\s+that|\s+for|$)'
        city_match = re.search(city_pattern, query, re.IGNORECASE)
        if city_match:
            entities['city'] = city_match.group(1).strip()
        
        # Extract course names with enhanced patterns
        course_patterns = [
            r'offer(?:ing)?\s+([a-zA-Z\s&.-]+?)(?:\s+in|\s+with|\s+course|$)',
            r'for\s+([a-zA-Z\s&.-]+?)(?:\s+in|\s+with|\s+course|$)',
            r'study\s+([a-zA-Z\s&.-]+?)(?:\s+in|\s+with|\s+course|$)',
            r'colleges?.*for\s+([a-zA-Z\s&.-]+?)(?:\s+in|\s+with|$)',
            r'([a-zA-Z\s&.-]+?)\s+course',
            r'([a-zA-Z\s&.-]+?)\s+program',
            r'([a-zA-Z\s&.-]+?)\s+degree'
        ]
        
        for pattern in course_patterns:
            course_match = re.search(pattern, query, re.IGNORECASE)
            if course_match:
                course_candidate = course_match.group(1).strip()
                # Filter out common words that aren't courses
                if course_candidate.lower() not in ['best', 'top', 'good', 'cheap', 'affordable']:
                    entities['course'] = course_candidate
                    break
        
        # Extract ratings
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:star|rating)',
            r'rating.*?(\d+(?:\.\d+)?)',
            r'rating\s+above\s+(\d+(?:\.\d+)?)',
            r'rating\s+at\s+least\s+(\d+(?:\.\d+)?)'
        ]
        for pattern in rating_patterns:
            rating_match = re.search(pattern, query, re.IGNORECASE)
            if rating_match:
                try:
                    entities['rating'] = float(rating_match.group(1))
                except:
                    pass
                break
        
        # Extract fee information
        fee_patterns = [
            r'fee.*?(\d[\d,.]*)\s*(?:lakh|thousand|k)?',
            r'cost.*?(\d[\d,.]*)\s*(?:lakh|thousand|k)?',
            r'budget.*?(\d[\d,.]*)\s*(?:lakh|thousand|k)?',
            r'under.*?(\d[\d,.]*)\s*(?:lakh|thousand|k)?'
        ]
        for pattern in fee_patterns:
            fee_match = re.search(pattern, query, re.IGNORECASE)
            if fee_match:
                try:
                    fee_str = fee_match.group(1).replace(',', '')
                    fee_value = float(fee_str)
                    # Convert lakhs to actual amount
                    if 'lakh' in query.lower():
                        fee_value *= 100000
                    elif 'thousand' in query.lower() or 'k' in query.lower():
                        fee_value *= 1000
                    entities['fee'] = fee_value
                except:
                    pass
                break
        
        # Extract approval bodies
        approval_patterns = [
            r'approval.*?from\s+(\w+)',
            r'approved.*?by\s+(\w+)',
            r'(\w+)\s+approved',
            r'accredited.*?by\s+(\w+)'
        ]
        for pattern in approval_patterns:
            approval_match = re.search(pattern, query, re.IGNORECASE)
            if approval_match:
                entities['approval'] = approval_match.group(1).upper()
                break
        
        # Check for course aliases and expand terms
        if entities['course']:
            for canonical, aliases in self.course_aliases.items():
                if any(alias in entities['course'].lower() for alias in aliases):
                    entities['course'] = canonical
                    break
        
        return entities

    def find_matching_colleges(self, query):
        """Find matching colleges based on user query"""
        if not hasattr(self, 'corpus_vectors'):
            return []
        
        # Preprocess query
        preprocessed_query, tokens = self.preprocess_query(query)
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([preprocessed_query])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()
        
        # Combine results with college data
        results = []
        for idx, similarity in enumerate(cosine_similarities):
            college = self.colleges_data[idx]
            results.append({
                'college': college,
                'similarity': similarity,
                'match_reasons': []
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter based on entities
        filtered_results = []
        for result in results:
            college = result['college']
            include = True
            
            # City filter
            if entities['city']:
                city = entities['city'].lower()
                if city not in college.get('city', '').lower():
                    include = False
                else:
                    result['match_reasons'].append(f"Located in {entities['city']}")
            
            # Course filter
            if entities['course']:
                course_found = False
                for course in college.get('courses', []):
                    course_name = course.get('course_name', '').lower()
                    if entities['course'].lower() in course_name:
                        course_found = True
                        break
                if not course_found:
                    include = False
                else:
                    result['match_reasons'].append(f"Offers {entities['course']} courses")
            
            # Rating filter
            if entities['rating']:
                college_rating = college.get('rating_value', 0)
                if college_rating < entities['rating']:
                    include = False
                else:
                    result['match_reasons'].append(f"Rating {college_rating} >= {entities['rating']}")
            
            # Fee filter
            if entities['fee']:
                min_fee = min(course.get('fee_amount', float('inf')) for course in college.get('courses', []))
                if min_fee > entities['fee']:
                    include = False
                else:
                    result['match_reasons'].append(f"Has courses under ₹{entities['fee']:,}")
            
            # Approval filter
            if entities['approval']:
                approvals = college.get('approvals', '').upper()
                if entities['approval'] not in approvals:
                    include = False
                else:
                    result['match_reasons'].append(f"{entities['approval']} approved")
            
            if include:
                filtered_results.append(result)
        
        return filtered_results[:5]  # Return top 5 results
    
    def get_relevant_courses(self, college, course_query=None):
        """Get relevant courses based on query"""
        courses = college.get('courses', [])
        
        if not course_query:
            return courses[:3]  # Return first 3 courses if no specific query
        
        # Preprocess course query
        preprocessed_course, _ = self.preprocess_query(course_query)
        
        # Filter courses that match the query
        relevant_courses = []
        for course in courses:
            course_name = course.get('course_name', '').lower()
            if preprocessed_course in course_name:
                relevant_courses.append(course)
        
        return relevant_courses
    
    def display_college(self, college, course_query=None):
        """Display college information with relevant courses"""
        with st.container():
            st.markdown(f"<div class='college-card'>", unsafe_allow_html=True)
            
            # College header
            st.markdown(f"<div class='college-header'>", unsafe_allow_html=True)
            st.markdown(f"<h3 class='college-name'>{college['college_name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='college-rating'>⭐ {college['rating_value']}/5</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # Close college-header
            
            # Location
            st.markdown(f"<div class='college-location'>📍 {college['city']}, {college['state']}</div>", unsafe_allow_html=True)
            
            # College type and approvals
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Type:** {college['college_type']}")
            with col2:
                st.markdown(f"**Approvals:** {college['approvals']}")
            
            # Positive notes
            st.markdown(f"<div style='margin: 1rem 0;'>{college['positive_notes']}</div>", unsafe_allow_html=True)
            
            # Courses section
            st.subheader("Relevant Courses & Fees")
            relevant_courses = self.get_relevant_courses(college, course_query)
            
            if relevant_courses:
                for course in relevant_courses:
                    with st.container():
                        st.markdown(f"<div class='course-item'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='course-name'>{course['course_name']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='course-fee'>💸 Annual Fee: ₹{course['fee_amount']:,}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No matching courses found in this college")
            
            # Additional info
            with st.expander("More Information"):
                st.markdown(f"**Website:** [{college['website']}]({college['website']})")
                st.markdown(f"**Contact:** {college['phone']}")
                st.markdown(f"**Email:** {college['email']}")
                if college.get('cutoffs'):
                    st.markdown("**Cutoffs:**")
                    for cutoff in college['cutoffs']:
                        st.markdown(f"- {cutoff}")
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close college-card

# Main application
def main():
    # Initialize chatbot
    chatbot = ZenthraChatbot()
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Load data
    with st.spinner("Loading college data..."):
        data_loaded = chatbot.load_data_from_file()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Zenthra College Finder</h1>
        <p>Discover the perfect college with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>Get Started</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "Find engineering colleges in Mumbai",
            "Suggest medical colleges with rating above 4.5",
            "Top MBA colleges under 5 lakh fees",
            "Computer science colleges in Delhi approved by UGC"
        ]
        
        for query in example_queries:
            if st.button(query, key=query, use_container_width=True, type='secondary'):
                st.session_state.user_query = query
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistics")
        if data_loaded:
            col1, col2 = st.columns(2)
            col1.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{len(chatbot.colleges_data)}</p>
                <p class="stat-label">Colleges</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate average rating - all ratings are now floats
            ratings = [college.get('rating_value', 0) for college in chatbot.colleges_data]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            col2.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{avg_rating:.1f}/5</p>
                <p class="stat-label">Avg Rating</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Data not loaded properly")
    
    # Chat interface
    st.markdown("### 🔍 Find Your Perfect College")
    user_query = st.text_input(
        "Ask about colleges (e.g., 'Engineering colleges in Mumbai'):",
        key="user_query",
        placeholder="Type your query here..."
    )
    
    # Process query when submitted
    if st.button("Find Colleges") or user_query:
        if not user_query:
            st.warning("Please enter a query")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Extract course query for filtering courses
        entities = chatbot.extract_entities(user_query)
        course_query = entities['course']
        
        # Show thinking animation
        with st.spinner("Searching for the best colleges..."):
            # Get matching colleges
            results = chatbot.find_matching_colleges(user_query)
            
            # Display assistant response
            if results:
                st.success(f"🎉 Found {len(results)} colleges matching your criteria")
                
                # Display each college
                for result in results:
                    chatbot.display_college(result['college'], course_query)
                    
                    # Show match reasons if any
                    if result['match_reasons']:
                        with st.expander("Why this college matches your query"):
                            for reason in result['match_reasons']:
                                st.markdown(f"- {reason}")
            else:
                st.warning("No colleges found matching your criteria. Try broadening your search.")
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("### 💬 Conversation History")
        for message in st.session_state.messages:
            role = "👤 You" if message["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {message['content']}")

# Run the app
if __name__ == "__main__":
    main()