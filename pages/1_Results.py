import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import pandas as pd
from emsemble1 import predict_from_csv
from absa import predict_from_csv1
from wordcloud import WordCloud
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Analysis Results",
    page_icon="📊",
    layout="wide"
)

# Remove top padding/margin
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    
    .plot-container {
        padding: 0rem;
        background-color: black;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_common_words(texts, n=20):
    words = []
    for text in texts:
        words.extend(clean_text(text).split())
    return Counter(words).most_common(n)

def show_results():
    if 'data' not in st.session_state or 'model_path' not in st.session_state:
        st.error("⚠️ Please upload data and configure model on the Home page first!")
        return

    st.title("📊 Analysis Results")

    # Run analysis
    try:
        with st.spinner("🔄 Running ensemble model analysis..."):
            # Save DataFrame temporarily
            temp_csv_path = "temp_upload.csv"
            st.session_state['data'].to_csv(temp_csv_path, index=False)
            
            results_df = predict_from_csv(
                csv_path=temp_csv_path,
                text_column='Comments',
                roberta_path=st.session_state['model_path'],
                distil_path='distillBert',
                num_labels=3
            )
            # Run ABSA analysis
            results_df1 = predict_from_csv1(
                csv_path=temp_csv_path,
                text_column='Comments'
            )

           

            # Save results to session state
            st.session_state['results_df'] = results_df

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "📈 Predictions Overview",
            "🎯 Confidence Analysis",
            "📝 Text Analysis"
        ])

        with tab1:
            st.subheader("Prediction Distribution")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('''
                    <div class="metric-card" style="border: 2px solid #e0e0e0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                        background: linear-gradient(to bottom, #ffffff, #f8f9fa);">
                ''', unsafe_allow_html=True)
                st.metric("Total Samples", len(results_df))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('''
                    <div class="metric-card" style="border: 2px solid #e0e0e0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        background: linear-gradient(to bottom, #ffffff, #f8f9fa);">
                ''', unsafe_allow_html=True)
                st.metric("Positive Predictions", len(results_df[results_df['prediction'] == 1]))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('''
                    <div class="metric-card" style="border: 2px solid #e0e0e0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        background: linear-gradient(to bottom, #ffffff, #f8f9fa);">
                ''', unsafe_allow_html=True)
                st.metric("Average Confidence", f"{results_df['confidence'].mean():.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('''
                    <div class="metric-card" style="border: 2px solid #e0e0e0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        background: linear-gradient(to bottom, #ffffff, #f8f9fa);">
                ''', unsafe_allow_html=True)
                st.metric("Median Confidence", f"{results_df['confidence'].median():.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Visualization row
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                fig_pie = px.pie(
                    results_df,
                    names='prediction',
                    title='Distribution of Predictions',
                    hole=0.4,
                    color_discrete_sequence=['#66B2FF', '#00FF00']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                pred_counts = results_df['prediction'].value_counts()
                fig_bar = px.bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    title='Prediction Counts',
                    labels={'x': 'Prediction', 'y': 'Count'},
                    color=pred_counts.index,
                    color_discrete_sequence=['#66B2FF', '##00FF00']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.subheader("Confidence Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                fig_hist = px.histogram(
                    results_df,
                    x='confidence',
                    nbins=20,
                    title='Distribution of Confidence Scores',
                    color_discrete_sequence=['#9DC88D']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                fig_box = px.box(
                    results_df,
                    x='prediction',
                    y='confidence',
                    title='Confidence Scores by Prediction',
                    color='prediction',
                    color_discrete_sequence=['#FF9999', '#66B2FF']
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.subheader("Text Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                # Add title for positive sentiments
                st.subheader("Positive Sentiments About")
                
                # Extract aspects and polarities from dictionary
                def extract_aspect_polarity(row):
                    if isinstance(row, dict) and 'span' in row:
                        return row['span']
                    return ''

                # Update the DataFrame columns
                results_df1['aspect'] = results_df1['aspect'].apply(extract_aspect_polarity)
                results_df1['polarity'] = results_df1['polarity'].apply(lambda x: x.get('polarity', '') if isinstance(x, dict) else '')
                results_df1['positive_aspects'] = [aspect if pol == 'positive' else '' 
                                            for aspect, pol in zip(results_df1['aspect'], results_df1['polarity'])]

                # Create wordcloud for positive aspects
                positive_aspects = ' '.join(results_df1['positive_aspects'].dropna().astype(str))

                if positive_aspects.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(positive_aspects)
                    fig_wordcloud = plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig_wordcloud)
                else:
                    st.write("No positive aspects found in the data")

                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)

                st.subheader("Negative Sentiments About")

                # Update the DataFrame for negative aspects
                results_df1['negative_aspects'] = [aspect if pol == 'negative' else '' 
                                            for aspect, pol in zip(results_df1['aspect'], results_df1['polarity'])]

                # Create wordcloud for negative aspects
                negative_aspects = ' '.join(results_df1['negative_aspects'].dropna().astype(str))

                if negative_aspects.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_aspects)
                    fig_wordcloud = plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig_wordcloud)
                else:
                    st.write("No negative aspects found in the data")

                st.markdown('</div>', unsafe_allow_html=True)
                
                

        # Download section
        st.markdown("""
            <div style="padding: 2rem; background-color: #; border-radius: 10px; margin-top: 2rem;">
                <h3>📥 Download Results</h3>
            </div>
        """, unsafe_allow_html=True)
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Results CSV",
            csv,
            "prediction_results.csv",
            "text/csv",
            key='download-csv'
        )

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    show_results()