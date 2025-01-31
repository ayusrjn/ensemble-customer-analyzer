import streamlit as st
import pandas as pd
from io import StringIO
import time
import datetime


# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'start_time' not in st.session_state:
    st.session_state.start_time = None


st.set_page_config(
    page_title="SentiScore",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Add progress bar and ETA styling to existing CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .main {
        padding: 0rem;
        max-width: 800px;
        margin: 0 auto;
    }
   
    .section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-image: url('https://wallpaperaccess.com/full/2994791.jpg');
        background-size: cover;
        background-position: center;
    }
   
    .header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
   
    .header-title {
        color: #1e40af;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
   
    .header-subtitle {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.4;
    }
   
    .section-header {
        color: #1f2937;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
   
    .stButton>button {
        background-color: #2563eb;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border: none;
        font-weight: 600;
        width: 100%;
        margin-top: 0.5rem;
    }
   
    .success-message {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #a7f3d0;
        margin: 0.5rem 0;
    }
   
    .error-message {
        background-color: #fef2f2;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #fecaca;
        margin: 0.5rem 0;
    }
   
    .stProgress > div > div > div > div {
        background-color: #2563eb;
    }
   
    .eta-text {
        color: #4b5563;
        text-align: center;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
.footer {
    text-align: center;
    padding: 1rem;
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 2rem;
    border-top: 1px solid #e5e7eb;
}
    .results-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
    }
    
    .sidebar-text {
        text-align: center;
        padding: 0.5rem;
        color: #4b5563;
        font-size: 0.9rem;
    }
   
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
   
    /* Hide Streamlit elements */
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
    </style>
""", unsafe_allow_html=True)


def simulate_analysis():
    """Simulate analysis with progress bar and ETA"""
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
   
    progress_text = "Analysis in progress. Please wait..."
    progress_bar = st.progress(0)
    status = st.empty()
   
    for percent_complete in range(100):
        time.sleep(0.05)  # Simulate processing time
        progress_bar.progress(percent_complete + 1)
       
        # Calculate ETA
        elapsed_time = time.time() - st.session_state.start_time
        estimated_total_time = (elapsed_time / (percent_complete + 1)) * 100
        remaining_time = estimated_total_time - elapsed_time
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
       
        status.markdown(f"""
            <div class="eta-text">
                Progress: {percent_complete + 1}% <br>
                Estimated completion at: {eta.strftime('%I:%M:%S %p')} <br>
                Time remaining: {int(remaining_time)} seconds
            </div>
        """, unsafe_allow_html=True)
   
    progress_bar.empty()
    status.empty()
    return True
def show_results():
        """Redirect to the Results page"""
        st.session_state.page = 'upload'  # Reset the page state
        st.session_state.start_time = None  # Reset the start time
        st.switch_page("pages/1_Results.py")

    # Note: The actual results display code should be moved to pages/1_Results.py
   
    # with col1:
    #     st.markdown("""
    #         <div class="metric-card">
    #             <h3>Positive</h3>
    #             <h2>65%</h2>
    #         </div>
    #     """, unsafe_allow_html=True)
   
    # with col2:
    #     st.markdown("""
    #         <div class="metric-card">
    #             <h3>Negative</h3>
    #             <h2>25%</h2>
    #         </div>
    #     """, unsafe_allow_html=True)
   
    # with col3:
    #     st.markdown("""
    #         <div class="metric-card">
    #             <h3>Neutral</h3>
    #             <h2>10%</h2>
    #         </div>
    #     """, unsafe_allow_html=True)
   
    # # Add additional analysis details here
    # st.markdown('</div>', unsafe_allow_html=True)
   
    # if st.button("⬅️ Start New Analysis"):
    #     st.session_state.page = 'upload'
    #     st.session_state.start_time = None
    #     st.experimental_rerun()


def main():
    with st.sidebar:
        # Add logo
        st.markdown("""
            <div class="sidebar-logo">
                <img src="Targaryns.png" width="150">
            </div>
        """, unsafe_allow_html=True)
        
        # Add divider
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Add sidebar text
        st.markdown("""
            <div class="sidebar-text">
                <h3>Welcome to Sentiment Analysis</h3>
                <p>Analyze your text data with our powerful ML models</p>
            </div>
        """, unsafe_allow_html=True)
    if st.session_state.page == 'results':
        show_results()
        return


    st.markdown("""
        <div class="header">
            <h1 class="header-title">📊 Sentiments 360</h1>
            <p class="header-subtitle">Enterprise-grade sentiment analysis powered by RoBERTa, DistilBERT, and XLNet</p>
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📤 Upload Your Data</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV file (must contain a 'text' column)",
        type=['csv']
    )


    if uploaded_file is not None:
        try:
            df = pd.read_csv(StringIO(uploaded_file.getvalue().decode('cp1252')))
            st.markdown('<h3>Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(3), use_container_width=True, height=120)
           
            if 'Comments' not in df.columns:
                st.markdown("""
                    <div class="error-message">
                        ❌ CSV must contain a 'Comments' column
                    </div>
                """, unsafe_allow_html=True)
                return
           
            st.markdown(f"""
                <div class="success-message">
                    ✅ Loaded {len(df):,} rows successfully
                </div>
            """, unsafe_allow_html=True)
            st.session_state['data'] = df
           
        except Exception as e:
            st.markdown(f"""
                <div class="error-message">
                    ❌ Error: {str(e)}
                </div>
            """, unsafe_allow_html=True)
            return
    st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<h2 class="section-header" style="padding-bottom: 1rem;">⚙️ Model Configuration</h2>', unsafe_allow_html=True)
   
    roberta_path = "roberta-final"
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64],
            value=16
        )
    with col2:
        threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5, 0.1
        )


    if st.button("🚀 Run Analysis"):
        if uploaded_file is None:
            st.markdown("""
                <div class="error-message">
                    Please upload a CSV file first
                </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state.update({
                'model_path': roberta_path,
                'batch_size': batch_size,
                'threshold': threshold
            })
           
            # Run analysis with progress bar and ETA
            if simulate_analysis():
                st.session_state.page = 'results'
                st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)
   
    st.markdown("""
        <div class="footer">
            Powered by Team Targaryns | Enterprise Sentiment Analysis Platform
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
