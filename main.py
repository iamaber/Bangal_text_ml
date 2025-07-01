import pickle
import pandas as pd
from bangla_news_processor import BanglaNewsProcessor

# Load the saved processor
with open('bangla_news_processor.pkl', 'rb') as f:
    processor = pickle.load(f)

# Load your new dataset
new_data = pd.read_csv('BusinessNewsData.csv')

# Process the new data
processed_new_data = processor.process_data(new_data)

# Use the processor for information retrieval
query = 'ভ্যাটের টাকার' #new query
results = processor.retrieve_information(query, processed_new_data)

# Display results
for result in results:
    print(f"Index: {result['index']}")
    print(f"Relevance Score: {result['score']}")
    print(f"Summary: {result['summary']}")
    print(f"Stemmed Text: {result['Stemmed Text']}")
    print()