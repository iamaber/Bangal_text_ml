import pandas as pd
import numpy as np
import re
import pickle
from collections import defaultdict
from bnlp import BengaliCorpus, CleanText, BasicTokenizer, BengaliNER
from bangla_stemmer.stemmer import stemmer

class BanglaNewsProcessor:
    def __init__(self):
        self.bangla_stopwords = set(BengaliCorpus.stopwords)
        self.clean_text = CleanText(
            fix_unicode=True,
            unicode_norm=True,
            unicode_norm_form="NFKC",
            remove_url=True,
            remove_email=True,
            remove_emoji=True,
            remove_number=False,
            remove_digits=False,
            remove_punct=False,
            replace_with_url="",
            replace_with_email="<EMAIL>",
            replace_with_number="<NUMBER>",
            replace_with_digit="<DIGIT>",
            replace_with_punct=""
        )
        self.tokenizer = BasicTokenizer()
        self.stemmer = stemmer.BanglaStemmer()
        self.bn_ner = BengaliNER()
        self.entity_index = defaultdict(set)

    def preprocess_bangla_text(self, text):
        text = ' '.join(text.split())
        text = ''.join(word for word in text.split() if word not in self.bangla_stopwords)
        bangla_punctuation = r'[!\"#$%&\'()*+,-./:;<=>@\[\\\]^_`{|}~]–?'
        text = re.sub(bangla_punctuation, '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize_and_stem(self, text):
        tokens = self.tokenizer.tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def tag_numbers(self, ner_tags):
        tagged_entities = []
        bangla_digit_pattern = r'[০-৯]+'
        bangla_date_pattern = r'[০-৯]{1,2}/[০-৯]{1,2}/[০-৯]{2,4}'
        for word, tag in ner_tags:
            if re.match(bangla_digit_pattern, word) or re.match(bangla_date_pattern, word):
                tagged_entities.append((word, 'NUMBER'))
            else:
                tagged_entities.append((word, tag))
        return tagged_entities

    def extract_entities(self, ner_tags):
        entities = []
        current_entity = []
        current_tag = None
        
        for word, tag in ner_tags:
            if tag != 'O':
                if tag == current_tag:
                    current_entity.append(word)
                else:
                    if current_entity:
                        entities.append((' '.join(current_entity), current_tag))
                    current_entity = [word]
                    current_tag = tag
            else:
                if current_entity:
                    entities.append((' '.join(current_entity), current_tag))
                    current_entity = []
                    current_tag = None
        
        if current_entity:
            entities.append((' '.join(current_entity), current_tag))
        
        return entities

    def clean_ner(self, text):
        ner_tags = self.bn_ner.tag(text)
        processed_ner_tags = self.tag_numbers(ner_tags)
        entities = self.extract_entities(processed_ner_tags)
        return ' '.join([entity for entity, _ in entities])

    def create_entity_index(self, data):
        for idx, row in data.iterrows():
            entities = row['ner_named_entities'].split()
            for entity in entities:
                self.entity_index[entity].add(idx)

    def expand_query(self, query):
        query_ner = self.bn_ner.tag(query)
        expanded_query = []
        for word, tag in query_ner:
            expanded_query.append(word)
            if tag != 'O':
                expanded_query.append(word)
        return ' '.join(expanded_query)

    def calculate_relevance_score(self, query_entities, document_entities):
        query_set = set(query_entities.split())
        doc_set = set(document_entities.split())
        
        bangla_digit_pattern = r'[০-৯]+'
        bangla_date_pattern = r'[০-৯]{1,2}/[০-৯]{1,2}/[০-৯]{2,4}'
        
        weighted_score = 0
        for i, entity in enumerate(document_entities.split()):
            if entity in query_set:
                if re.match(bangla_digit_pattern, entity) or re.match(bangla_date_pattern, entity):
                    weighted_score += 2 / (i + 1)
                else:
                    weighted_score += 1 / (i + 1)
        
        return weighted_score / len(query_set) if query_set else 0

    def entity_based_summary(self, text, named_entities, max_length=1000):
        words = text.split()
        entities = named_entities.split()
        bangla_digit_pattern = r'[০-৯]+'
        bangla_date_pattern = r'[০-৯]{1,2}/[০-৯]{1,2}/[০-৯]{2,4}'
        number_entities = [e for e in entities if re.match(bangla_digit_pattern, e) or re.match(bangla_date_pattern, e)]
        other_entities = [e for e in entities if e not in number_entities]
        
        summary = []
        context_window = 2
        
        def add_with_context(word_index):
            start = max(0, word_index - context_window)
            end = min(len(words), word_index + context_window + 1)
            summary.extend(words[start:end])
        
        for i, word in enumerate(words):
            if word in number_entities and len(' '.join(summary)) < max_length:
                add_with_context(i)
        
        for i, word in enumerate(words):
            if word in other_entities and len(' '.join(summary)) < max_length and word not in summary:
                add_with_context(i)
            if len(' '.join(summary)) >= max_length:
                break
        
        return ' '.join(summary)

    def process_data(self, data):
        data['preprocess_description'] = data['Description'].apply(self.preprocess_bangla_text)
        data['clean_description'] = data['preprocess_description'].apply(self.clean_text)
        data['tokens'] = data['clean_description'].apply(self.tokenizer.tokenize)
        data['stemmed_description'] = data['clean_description'].apply(self.tokenize_and_stem)
        data['ner_named_entities'] = data['stemmed_description'].apply(self.clean_ner)
        self.create_entity_index(data)
        return data

    def retrieve_information(self, query, data):
        expanded_query = self.expand_query(query)
        query_entities = self.clean_ner(expanded_query)
        
        scores = []
        for idx, row in data.iterrows():
            score = self.calculate_relevance_score(query_entities, row['ner_named_entities'])
            scores.append((idx, score))
        
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:5]
        
        results = []
        for idx, score in top_results:
            summary = self.entity_based_summary(data.loc[idx, 'stemmed_description'], data.loc[idx, 'ner_named_entities'])
            results.append({
                'index': idx,
                'score': score,
                'summary': summary,
                'Stemmed Text': data.loc[idx, 'stemmed_description']
            })
        
        return results

# Usage example:
if __name__ == "__main__":
    processor = BanglaNewsProcessor()
    
    # Load and process data
    file_path = 'BusinessNewsData.csv'
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['Description', 'Keywords-for-related-articles'])
    processed_data = processor.process_data(data)
    
    # Save the processor object
    with open('bangla_news_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    # Example query
    query = "শেয়ারবাজার"
    results = processor.retrieve_information(query, processed_data)
    
    print("Query:", query)
    print("\nTop 5 Results:")
    for result in results:
        print(f"Index: {result['index']}")
        print(f"Relevance Score: {result['score']}")
        print(f"Summary: {result['summary']}")
        print(f"Stemmed Text: {result['Stemmed Text']}")
        print()

    results_df = pd.DataFrame(results)