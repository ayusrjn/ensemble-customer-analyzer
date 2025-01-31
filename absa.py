from setfit import AbsaModel
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
model = AbsaModel.from_pretrained(
    "models\setfit-absa-model-aspect",
    "models\setfit-absa-model-polarity"
)

# Get predictions
# preds = model.predict([
#     "Best pizza outside of Italy and really tasty.",
#     "The food variations are great and the prices are absolutely fair.",
#     "Unfortunately, you have to expect some waiting time and get a note with a waiting number if it should be very full."
# ])

# print(preds)

def predict_from_csv1(csv_path, text_column):
    # Load data
    data = pd.read_csv(csv_path, encoding="cp1252")
    
    # Get predictions
    preds = model.predict(data[text_column].tolist())
    aspects = []
    polarities = []
    
    # Safely extract aspects and polarities
    for pred in preds:
        if isinstance(pred, (list, tuple)) and len(pred) >= 2:
            aspects.append(pred[0])
            polarities.append(pred[1])
        else:
            aspects.append("")
            polarities.append("")
    
    # Add predictions to DataFrame
    # Create separate lists for positive and negative aspects
    data['positive_aspects'] = [aspect if pol == "positive" else "" 
                              for aspect, pol in zip(aspects, polarities)]
    data['negative_aspects'] = [aspect if pol == "negative" else "" 
                              for aspect, pol in zip(aspects, polarities)]
    data['aspect'] = aspects
    data['polarity'] = polarities

    return data

import matplotlib.pyplot as plt

def create_aspect_wordcloud(data):
    # Combine all aspects into a single string
    text = ' '.join(data['aspect'])
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
        # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Get predictions and save to CSV
# Create word clouds for positive aspects
# result = predict_from_csv1("sample_reviews.csv", "text")
# print("Total rows in result:", len(result))

# # Extract actual aspects and polarities from dictionary format
# def extract_aspect_polarity(row):
#     if isinstance(row, dict) and 'span' in row:
#         return row['span']
#     return ''

# # Update the DataFrame columns
# result['aspect'] = result['aspect'].apply(extract_aspect_polarity)
# result['polarity'] = result['polarity'].apply(lambda x: x.get('polarity', '') if isinstance(x, dict) else '')
# result['negative_aspects'] = [aspect if pol == 'negative' else '' 
#                             for aspect, pol in zip(result['aspect'], result['polarity'])]

# print("Sample of aspects and polarities:")
# print(result[['aspect', 'polarity', 'negative_aspects']].head())

# # Create wordcloud for negative aspects
# negative_aspects = ' '.join(result['negative_aspects'].dropna().astype(str))
# print("Negative aspects string:", negative_aspects[:100])
# if negative_aspects.strip():
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_aspects)
#     plt.figure(figsize=(10,5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.savefig('negative_aspects_wordcloud.png')
#     plt.show()
# else:
#     print("No negative aspects found in the data")
