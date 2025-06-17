import requests
from pyquery import PyQuery as pq
import json
import time
import re
import random
import os

# Load college dataset
if os.path.exists("college_dataset.json"):
    with open("college_dataset.json", "r", encoding="utf-8") as f:
        college_data = json.load(f)
else:
    print("Error: college_dataset.json not found!")
    exit(1)

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183"
]

# Base URL for colleges
BASE_URL = "https://collegedunia.com/"

# Final extracted data dictionary
all_extracted_data = {}

def extract_data_from_html(html_text):
    """Extract college data from HTML content"""
    src = pq(html_text)
    jsonData = src.find('script:last')
    final = pq(jsonData).text()
    
    try:
        data_dict = json.loads(final)
    except json.JSONDecodeError:
        return None
    
    # Extract required parameters
    extracted_data = {}
    
    # Extract college name
    try:
        college_name = data_dict['props']['initialProps']['pageProps']['data']['college_name']
        extracted_data['college_name'] = college_name
    except (KeyError, IndexError, TypeError):
        extracted_data['college_name'] = None
    
    # Extract stream details
    streams_info = []
    try:
        streams = data_dict['props']['initialProps']['pageProps']['data']['streamGoalData']['streams']
        for stream in streams:
            stream_name = stream.get('stream', '')
            slugs = stream.get('slugs', [])
            slug = slugs[0].get('slug', '') if slugs else ''
            streams_info.append({
                'stream_name': stream_name,
                'slug': slug
            })
    except (KeyError, TypeError):
        pass
    extracted_data['streams'] = streams_info
    
    # Extract website, phone, college_type
    try:
        basic_info = data_dict['props']['initialProps']['pageProps']['data']['basic_info']
        website = basic_info.get('website', '')
        phone = basic_info.get('mobile', [])
        college_type = basic_info.get('type_of_college', '')
        extracted_data['website'] = website
        extracted_data['phone'] = phone
        extracted_data['college_type'] = college_type
    except (KeyError, TypeError):
        extracted_data['website'] = ''
        extracted_data['phone'] = []
        extracted_data['college_type'] = ''
    
    # Extract email
    extracted_data['email'] = ''
    try:
        schema_json_str = data_dict['props']['initialProps']['pageProps']['data']['metadata']['schemaJsonLd']['2']
        try:
            schema_data = json.loads(schema_json_str)
            email = schema_data.get('email', '')
            extracted_data['email'] = email
        except json.JSONDecodeError:
            email_match = re.search(r'"email":"([^"]+)"', schema_json_str)
            if email_match:
                extracted_data['email'] = email_match.group(1)
    except (KeyError, TypeError):
        pass
    
    # Extract course fees
    courses_info = []
    try:
        full_time_courses = data_dict['props']['initialProps']['pageProps']['data']['new_compare_courses']['full_time']
        for course in full_time_courses:
            stream_list = course.get('stream', [])
            for stream_item in stream_list:
                course_name = stream_item.get('name', '')
                fee_amount = stream_item.get('total_current_fee', {}).get('general', {}).get('amount', '')
                
                courses_info.append({
                    'course_name': course_name,
                    'fee_amount': fee_amount
                })
    except (KeyError, TypeError):
        pass
    extracted_data['courses'] = courses_info
    
    # Extract cutoff details
    cutoffs_info = []
    try:
        course_data = data_dict['props']['initialProps']['pageProps']['data']['course_data']
        courses_list = course_data['courses']
        cutoff_dict = course_data.get('cutoff', {})
        
        # Create course ID to name mapping
        id_to_name = {}
        for course in courses_list:
            try:
                id_to_name[course['id']] = course['name']
            except (KeyError, TypeError):
                continue
        
        for course_id, cutoff_list in cutoff_dict.items():
            try:
                course_name = id_to_name.get(int(course_id), 'Unknown Course')
                for cutoff in cutoff_list:
                    cutoffs_info.append({
                        'course_name': course_name,
                        'cutoff': cutoff.get('cutoff', ''),
                        'cutoff_type': cutoff.get('cutoff_type', ''),
                        'exam': cutoff.get('exam', '')
                    })
            except (ValueError, TypeError):
                continue
        
        # Fallback to stream cutoffs if needed
        if not cutoffs_info:
            try:
                full_time_courses = data_dict['props']['initialProps']['pageProps']['data']['new_compare_courses']['full_time']
                for course in full_time_courses:
                    stream_list = course.get('stream', [])
                    for stream_item in stream_list:
                        cutoff_data = stream_item.get('cutoff', {})
                        if cutoff_data and isinstance(cutoff_data, dict):
                            course_name = stream_item.get('name', '')
                            cutoffs_info.append({
                                'course_name': course_name,
                                'cutoff': cutoff_data.get('cutoff', ''),
                                'cutoff_type': cutoff_data.get('cutoff_type', ''),
                                'exam': cutoff_data.get('exam', '')
                            })
            except (KeyError, TypeError):
                pass
    except (KeyError, TypeError):
        pass
    extracted_data['cutoffs'] = cutoffs_info
    
    return extracted_data

def process_colleges():
    """Process all colleges in the dataset"""
    global all_extracted_data
    
    # Initialize variables
    request_count = 0
    user_agent = random.choice(USER_AGENTS)
    total_colleges = len(college_data)
    processed_count = 0
    max_colleges = 40  # Set maximum number of colleges to process
    
    print(f"Starting to process first {max_colleges} colleges")
    
    for college_id, college_info in college_data.items():
        # Break if we've reached the maximum number of colleges
        if processed_count >= max_colleges:
            break
            
        # Rotate user agent every 10 requests
        if request_count % 10 == 0 and request_count > 0:
            user_agent = random.choice(USER_AGENTS)
            print(f"Rotating user agent to: {user_agent[:50]}...")
            time.sleep(5)  # Sleep before making next batch of requests
        
        # Configure headers
        HEADERS = {
            "User-Agent": user_agent,
            "Accept": "application/json"
        }
        
        # Create full URL from the path in the dataset
        full_url = BASE_URL + college_info['url']
        print(f"\nProcessing college {processed_count+1}/{max_colleges}: {college_id}")
        print(f"URL: {full_url}")
        
        try:
            response = requests.get(full_url, headers=HEADERS, timeout=30)
            request_count += 1
            processed_count += 1
            
            if response.status_code == 200:
                extracted_data = extract_data_from_html(response.text)
                
                if extracted_data:
                    # Add basic info from original dataset
                    extracted_data['city'] = college_info.get('city', '')
                    extracted_data['state'] = college_info.get('state', '')
                    extracted_data['rating'] = college_info.get('rating', '')
                    extracted_data['score'] = college_info.get('score', '')
                    extracted_data['approvals'] = college_info.get('approvals', [])
                    
                    all_extracted_data[college_id] = extracted_data
                    print(f"Extracted: {extracted_data.get('college_name', 'Unknown')}")
                else:
                    print(f"Failed to extract data for college ID: {college_id}")
            else:
                print(f"Failed to load page: Status {response.status_code} for ID {college_id}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for ID {college_id}: {str(e)}")
            processed_count += 1  # Increment even on failure to count attempt
        except Exception as e:
            print(f"Error processing ID {college_id}: {str(e)}")
            processed_count += 1  # Increment even on failure to count attempt
        
        # Save progress after each college
        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(all_extracted_data, f, indent=4)
        print(f"Saved progress after college {processed_count}")
    
    # Final save
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_data, f, indent=4)
    print(f"\nCompleted! Extracted data for {len(all_extracted_data)}/{processed_count} colleges")
    
    # Final save
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_data, f, indent=4)
    print(f"\nCompleted! Extracted data for {len(all_extracted_data)}/{total_colleges} colleges")

if __name__ == "__main__":
    process_colleges()