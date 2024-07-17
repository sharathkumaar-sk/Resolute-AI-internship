import torch
from transformers import (
    BertTokenizer,
    BartForConditionalGeneration,  # Use Bart for summarization
)
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Load pre-trained models (consider fine-tuning on summarization data)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_preprocess_pdf(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    read_pdf = PyPDF2.PdfReader(pdf_file)
    number_of_pages = len(read_pdf.pages)
    text = ''
    for page in read_pdf.pages:
        page_content = page.extract_text()
        text += page_content
    sentences = text.split('. ')
    return sentences

def hierarchical_summarization(sentences):
    sentence_embeddings = sentence_transformer.encode(sentences)
    clustering_model = KMeans(n_clusters=int(np.ceil(len(sentences) / 5)))
    clustering_model.fit(sentence_embeddings)
    clustered_sentences = {}
    for i, label in enumerate(clustering_model.labels_):
        if label not in clustered_sentences:
            clustered_sentences[label] = []
        clustered_sentences[label].append(sentences[i])
    representative_sentences = []
    for cluster in clustered_sentences.values():
        sentence_embeddings = sentence_transformer.encode(cluster)
        centroid = np.mean(sentence_embeddings, axis=0)
        distances = pairwise_distances_argmin([centroid], sentence_embeddings)[0]
        representative_sentences.append(cluster[distances])
    return representative_sentences

def long_range_summarization(sentences):
    input_ids = tokenizer.encode_plus(
        '. '.join(sentences),
        max_length=1024,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    summary_ids = summarization_model.generate(
        input_ids['input_ids'],
        attention_mask=input_ids['attention_mask'],
        max_length=128,
        early_stopping=True
    )
    chunk_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return chunk_summaries

def post_processing(chunk_summaries):
    sentence_embeddings = sentence_transformer.encode(chunk_summaries)
    distances = pairwise_distances_argmin(sentence_embeddings, sentence_embeddings)
    final_summary = [chunk_summaries[i] for i in distances]
    return '. '.join(final_summary)

def main():
    pdf_path = "Operations Management.pdf"
    sentences = load_and_preprocess_pdf(pdf_path)

    representative_sentences = hierarchical_summarization(sentences)
    chunk_summaries = long_range_summarization(representative_sentences)

    final_summary = post_processing(chunk_summaries)
    clean_output = final_summary.replace("[unused", "").replace("]", "")
    print(clean_output)

if __name__ == "__main__":
    main()
