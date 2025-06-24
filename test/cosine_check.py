import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

def visualize_similarity_clusters(colleges_data, model):
    """Visualize college similarity clusters using PCA and Seaborn"""
    if not colleges_data or len(colleges_data) == 0:
        st.warning("No college data available for visualization")
        return
    
    with st.spinner("Creating similarity clusters visualization..."):
        # Generate college embeddings
        college_texts = []
        for college in colleges_data:
            text_parts = []
            text_parts.append(college.get('college_name', ''))
            text_parts.append(college.get('city', ''))
            text_parts.append(college.get('state', ''))
            text_parts.append(college.get('approvals', ''))
            text_parts.append(college.get('college_type', ''))
            text_parts.append(college.get('positive_notes', ''))
            
            streams = college.get('streams', [])
            if isinstance(streams, list):
                text_parts.extend(streams)
            elif isinstance(streams, str):
                text_parts.append(streams)
            
            courses = college.get('courses', [])
            if isinstance(courses, list):
                for course in courses:
                    if isinstance(course, dict):
                        text_parts.append(course.get('course_name', ''))
                    elif isinstance(course, str):
                        text_parts.append(course)
            
            combined_text = ' '.join([str(part) for part in text_parts if part])
            college_texts.append(combined_text)
        
        # Generate embeddings
        college_embeddings = model.encode(
            college_texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        college_embeddings_np = college_embeddings.cpu().numpy()
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(college_embeddings_np)
        
        # Cluster colleges
        n_clusters = min(8, len(colleges_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(college_embeddings_np)
        
        # Prepare data for plotting
        college_names = [c['college_name'] for c in colleges_data]
        states = [c.get('state', 'N/A') for c in colleges_data]
        ratings = [c.get('rating_value', 0) for c in colleges_data]
        types = [c.get('college_type', 'Unknown') for c in colleges_data]
        
        # Create DataFrame
        plot_data = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'college': college_names,
            'cluster': clusters,
            'state': states,
            'rating': ratings,
            'type': types
        })
        
        # Create scatter plot
        sns.scatterplot(
            data=plot_data,
            x='x',
            y='y',
            hue='cluster',
            palette='viridis',
            size='rating',
            sizes=(50, 300),
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Add college labels (only for high-rated colleges to reduce clutter)
        top_colleges = plot_data.nlargest(15, 'rating')
        for i, row in top_colleges.iterrows():
            plt.annotate(
                row['college'], 
                (row['x'], row['y']),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Add plot decorations
        plt.title(f'College Similarity Clusters (n={len(colleges_data)})', fontsize=16)
        plt.xlabel('PCA Dimension 1', fontsize=12)
        plt.ylabel('PCA Dimension 2', fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.2)
        
        # Show the plot in Streamlit
        st.pyplot(plt)
        
        # Add cluster analysis
        st.subheader("Cluster Analysis")
        st.write("Colleges are grouped based on similarity in their features, courses, and descriptions.")
        
        # Show cluster characteristics
        for cluster_id in range(n_clusters):
            cluster_df = plot_data[plot_data['cluster'] == cluster_id]
            cluster_colleges = [c for c, cl in zip(colleges_data, clusters) if cl == cluster_id]
            
            with st.expander(f"Cluster {cluster_id} - {len(cluster_df)} colleges"):
                # Top colleges in cluster
                st.write(f"**Top colleges in this cluster:**")
                top_in_cluster = cluster_df.nlargest(3, 'rating')
                for _, row in top_in_cluster.iterrows():
                    st.markdown(f"- {row['college']} (Rating: {row['rating']:.1f})")
                
                # Common characteristics
                common_streams = get_common_features(cluster_colleges, 'streams')
                common_courses = get_common_features(cluster_colleges, 'courses')
                
                st.write("**Common characteristics:**")
                if common_streams:
                    st.markdown(f"- **Streams:** {', '.join(common_streams)}")
                if common_courses:
                    st.markdown(f"- **Courses:** {', '.join(common_courses)}")
                
                # College type distribution
                type_counts = cluster_df['type'].value_counts().to_dict()
                st.markdown(f"- **College Types:** {', '.join([f'{k} ({v})' for k, v in type_counts.items()])}")
                
                # Average rating
                avg_rating = cluster_df['rating'].mean()
                st.markdown(f"- **Average Rating:** {avg_rating:.1f}/5")

def get_common_features(colleges, feature_type):
    """Get common features in a cluster of colleges"""
    feature_counter = defaultdict(int)
    
    for college in colleges:
        features = college.get(feature_type, [])
        if isinstance(features, str):
            features = [features]
        
        for feature in features:
            if isinstance(feature, dict) and 'course_name' in feature:
                feature_counter[feature['course_name']] += 1
            elif isinstance(feature, str):
                feature_counter[feature] += 1
    
    # Get top 5 most common features
    sorted_features = sorted(feature_counter.items(), key=lambda x: x[1], reverse=True)
    return [feat for feat, count in sorted_features[:5]]