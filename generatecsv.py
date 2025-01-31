import pandas as pd
import numpy as np

# Create sample text data
sample_data = {
    'text': [
        "This product is amazing! I love how well it works.",
        "Terrible experience with customer service. Would not recommend.",
        "The package arrived on time but the item was damaged.",
        "Been using this for 6 months and it's still working great!",
        "Average product, nothing special but gets the job done.",
        "Waste of money, completely useless product.",
        "The setup was straightforward and the interface is user-friendly.",
        "Not what I expected, but it works okay for basic tasks.",
        "Outstanding quality and excellent value for money.",
        "Had some issues initially but support team was helpful.",
        "Perfect for my needs, exactly what I was looking for!",
        "Instructions were confusing and product stopped working after a week.",
        "Good build quality but missing some key features.",
        "Very impressed with the performance and reliability.",
        "Shipping took forever and product barely works."
    ]
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Save to CSV
df.to_csv('sample_reviews.csv', index=False)

# Display first few rows
print(df.head())