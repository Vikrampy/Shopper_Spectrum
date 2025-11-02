import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pickle # Used for simulating model loading

# --- 0. Data and Product List Definitions ---

# The specific list of products provided by the user for easy selection
SELECTION_PRODUCTS = [
    '--- Select a Product ---', # Placeholder for initial selection
    'WHITE HANGING HEART T-LIGHT HOLDER',
    'RED METAL BOX TOP SECRET',
    'SMALL PARISIENNE HEART PHOTO FRAME',
    'RED PAPER PARASOL',
    'LOVE SEAT ANTIQUE WHITE METAL',
    'PINK HEARTS LIGHT CHAIN',
    'PINK/BLUE DISC/MIRROR STRING',
    'BISCUIT TIN VINTAGE LEAF'
]

# Map cluster labels to meaningful segment names (Based on standard RFM interpretation)
CLUSTER_LABELS = {
    0: 'High-Value Customer', # Low Recency, High Frequency, High Monetary
    1: 'At-Risk Customer',    # High Recency, Low Frequency, Low Monetary
    2: 'Regular Customer',    # Medium values
    3: 'Occasional Shopper'   # Low Frequency, Low Monetary, older Recency
}

# --- Shared Data Loading and Model Functions ---

@st.cache_data
def load_and_prepare_data():
    # Load your actual DataFrame here. For demonstration, we'll use a sample.
    data = pd.read_csv('online_retail.csv')
    df = pd.DataFrame(data)

    # Clean the data by removing cancelled invoices and non-positive quantity/unit price
    df_clean = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[df_clean['Quantity'] > 0]

    # Handle null values by removing rows with missing Description or CustomerID
    df_clean.dropna(subset=['Description', 'CustomerID'], inplace=True)

    # Create the user-item matrix
    user_item_matrix = df_clean.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)

    # Compute the product-to-product similarity matrix
    product_similarity_matrix = pd.DataFrame(cosine_similarity(user_item_matrix.T),
                                             index=user_item_matrix.columns,
                                             columns=user_item_matrix.columns)
    
    return product_similarity_matrix

@st.cache_resource
def load_rfm_assets():
    """
    Attempts to load the trained K-Means model and StandardScaler from pickle files.
    """
    model_path = 'kmeans_model.pkl'
    scaler_path = 'scaler.pkl'
    
    try:
        # Load the K-Means Model
        with open(model_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        
        # Load the StandardScaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return kmeans_model, scaler
        
    except FileNotFoundError:
        # Return None and the specific error for graceful handling in the page function
        return None, f"Model files not found. Please ensure both '{model_path}' and '{scaler_path}' are uploaded and accessible."
    except Exception as e:
        return None, f"An error occurred during model loading: {e}"

# A function to get recommendations based on the similarity matrix
def get_similar_products(product_name, similarity_matrix, top_n=5):
    """Retrieves the top N most similar products based on cosine similarity."""
    if product_name not in similarity_matrix.index:
        return f"Product '{product_name}' not found in the sales history used for modeling."
    
    product_scores = similarity_matrix[product_name].sort_values(ascending=False)
    similar_products = product_scores.drop(product_name, errors='ignore').head(top_n)
    
    return similar_products

# --- Page Functions ---

def home_page():
    st.title("Welcome to the E-commerce Analytics App")
    st.markdown("""
        This application provides an overview of your sales data and offers two key analytical tools:
        
        * **Clustering:** Segment your customers based on their purchasing behavior (RFM Analysis).
        * **Recommendation:** Get product recommendations based on item-based collaborative filtering.
        
        Use the sidebar to navigate between modules.
    """)
    st.image("https://placehold.co/800x200/4CAF50/white?text=E-commerce+Dashboard+Overview", use_column_width=True)

def clustering_page():
    st.title("📊 Customer Segmentation Module")
    st.markdown("---")
    st.markdown("Enter a customer's purchasing metrics to predict their segment (Recency, Frequency, Monetary).")

    # Load the model assets once
    kmeans_model, scaler = load_rfm_assets()

    # Create the three number inputs using st.columns for better alignment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=1, max_value=730, value=30, step=1)
    
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5, step=1)
    
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=10.0, max_value=50000.0, value=500.0, step=10.0)

    st.markdown("---")

    if st.button("Predict Segment", type="primary"):
        with st.spinner("Calculating Customer Segment..."):
            
            # 1. Combine inputs into a DataFrame/Array
            new_customer_data = np.array([[recency, frequency, monetary]])
            
            # 2. Scale the data using the trained scaler
            scaled_data = scaler.transform(new_customer_data)
            
            # 3. Predict the cluster ID
            cluster_id = kmeans_model.predict(scaled_data)[0]
            
            # 4. Map the ID to the label
            segment_label = CLUSTER_LABELS.get(cluster_id, "Unknown Segment")

            st.success("Prediction Complete!")
            
            st.markdown(f"""
                <div style="
                    border: 2px solid #ff4b4b; 
                    padding: 20px; 
                    margin-top: 20px;
                    border-radius: 10px; 
                    background-color: #ffeaea;
                    text-align: center;
                ">
                    <h3 style='margin: 0; color: #ff4b4b;'>Predicted Cluster ID: {cluster_id}</h3>
                    <h2 style='margin: 5px 0 0 0; color: #0e1117;'>Segment: {segment_label}</h2>
                </div>
            """, unsafe_allow_html=True)

def recommendation_page():
    st.title('🛒 Item-Based Product Recommender')
    st.markdown('***')

    # Use st.selectbox for easy selection of known products
    product_name_select = st.selectbox(
        'Select a Product for Recommendations', 
        options=SELECTION_PRODUCTS,
        index=0 # Default to the placeholder option
    )

    if st.button('Get Recommendations', use_container_width=True, type="primary"):
        if product_name_select == SELECTION_PRODUCTS[0]:
            st.warning('Please select a valid product name from the list to get recommendations.')
        else:
            with st.spinner(f'Finding similar products for "{product_name_select}"...'):
                # Load data and similarity matrix
                try:
                    product_similarity_matrix = load_and_prepare_data()
                    recommendations = get_similar_products(product_name_select, product_similarity_matrix)
                except Exception as e:
                    st.error(f"An error occurred during data processing: {e}")
                    recommendations = f"Failed to generate recommendations due to data error."
            
            st.subheader(f'Top 5 Recommendations for: *{product_name_select}*')
            
            # Display the results
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                # Styled card view using HTML/Markdown
                for i, item in enumerate(recommendations.index):
                    st.markdown(
                        f"""
                        <div style="
                            border-left: 5px solid #ff4b4b; 
                            padding: 10px; 
                            margin-bottom: 10px; 
                            border-radius: 5px; 
                            background-color: #f0f2f6;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                        ">
                            <h4 style='margin: 0; color: #0e1117;'>{i+1}. {item}</h4>
                            <p style='margin: 0; font-size: 0.9em; color: #555;'>Similarity Score: <b>{recommendations.iloc[i]:.4f}</b></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
# --- Main App Logic with Sidebar Navigation ---

st.set_page_config(page_title="E-commerce Analytics App", layout="centered", initial_sidebar_state="expanded")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'Clustering' # Set default page to the new Clustering module

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")
    
    # Custom button-based navigation for visual fidelity
    
    if st.button("🖥️ Home", key="nav_home", use_container_width=True, type="secondary" if st.session_state['page'] != 'Home' else "primary"):
        st.session_state['page'] = 'Home'

    if st.button("📅 Clustering", key="nav_cluster", use_container_width=True, type="secondary" if st.session_state['page'] != 'Clustering' else "primary"):
        st.session_state['page'] = 'Clustering'

    if st.button("📕 Recommendation", key="nav_recommendation", use_container_width=True, type="secondary" if st.session_state['page'] != 'Recommendation' else "primary"):
        st.session_state['page'] = 'Recommendation'

# --- Display Content Based on Selected Page ---
if st.session_state['page'] == 'Home':
    home_page()
elif st.session_state['page'] == 'Clustering':
    clustering_page()
elif st.session_state['page'] == 'Recommendation':
    recommendation_page()
