# import gradio as gr
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk import pos_tag, ne_chunk
# from nltk.tree import Tree
# import re
# from collections import Counter
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import pandas as pd
# from textblob import TextBlob
# import textstat
# from yake import KeywordExtractor
# from PyPDF2 import PdfReader
# import json
# import os

# # ====================== ROBUST NLTK SETUP ======================
# def setup_nltk():
#     resources = [
#         'punkt', 'punkt_tab',
#         'wordnet', 'omw-1.4',
#         'averaged_perceptron_tagger_eng',      # POS tagger (English)
#         'maxent_ne_chunker',                   # Legacy chunker
#         'maxent_ne_chunker_tab',               # ← NEW: Required for ne_chunk in recent NLTK
#         'words',                               # Word list for chunker
#         'stopwords'
#     ]

#     # Use a reliable writable path (especially good for Windows)
#     nltk_data_dir = os.path.expanduser("~/nltk_data")
#     os.makedirs(nltk_data_dir, exist_ok=True)
#     if nltk_data_dir not in nltk.data.path:
#         nltk.data.path.append(nltk_data_dir)

#     for res in resources:
#         try:
#             if "tagger" in res:
#                 nltk.data.find(f"taggers/{res}")
#             elif "chunker" in res:
#                 nltk.data.find(f"chunkers/{res}")
#             elif "stopwords" in res or "wordnet" in res or "words" in res:
#                 nltk.data.find(f"corpora/{res}")
#             else:
#                 nltk.data.find(f"tokenizers/{res}")
#         except LookupError:
#             print(f"Downloading NLTK resource: {res}")
#             nltk.download(res, download_dir=nltk_data_dir, quiet=False)

# setup_nltk()

# # ====================== GLOBAL OBJECTS ======================
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # ====================== HELPER FUNCTIONS ======================
# def get_wordnet_pos(tag):
#     if tag.startswith('J'):   return 'a'
#     elif tag.startswith('V'): return 'v'
#     elif tag.startswith('N'): return 'n'
#     elif tag.startswith('R'): return 'r'
#     else:                     return 'n'

# def extract_text_from_file(file):
#     if not file:
#         return ""
#     try:
#         if file.name.lower().endswith('.txt'):
#             with open(file.name, 'r', encoding='utf-8') as f:
#                 return f.read()
#         elif file.name.lower().endswith('.pdf'):
#             reader = PdfReader(file.name)
#             return " ".join(page.extract_text() or "" for page in reader.pages)
#         return "Unsupported file (only .txt and .pdf)"
#     except Exception as e:
#         return f"Error reading file: {str(e)}"

# def extract_entities(text):
#     try:
#         words = word_tokenize(text)
#         pos_tags = pos_tag(words)
#         tree = ne_chunk(pos_tags)
#         entities = []
#         for chunk in tree:
#             if isinstance(chunk, Tree):
#                 entity = " ".join(w for w, t in chunk.leaves())
#                 entities.append([entity, chunk.label()])
#         return entities or [["No named entities found", ""]]
#     except LookupError as e:
#         # Graceful fallback if chunker still missing
#         return [["NER unavailable (missing NLTK resource)", str(e)]]
#     except Exception as e:
#         return [["Error in NER", str(e)]]

# def simple_extractive_summary(text, num_sentences=3):
#     sentences = sent_tokenize(text)
#     if len(sentences) <= num_sentences:
#         return text
#     words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
#     freq = Counter(words)
#     scores = {sent: sum(freq.get(w.lower(), 0) for w in word_tokenize(sent)) for sent in sentences}
#     top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
#     return " ".join(s for s, _ in top)

# def generate_wordcloud_img(text, remove_stop=True):
#     if not text.strip():
#         return None
#     words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
#     if remove_stop:
#         words = [w for w in words if w not in stop_words]
#     if not words:
#         return None
#     wc = WordCloud(width=900, height=500, background_color="white", max_words=150).generate(" ".join(words))
#     return wc.to_array()

# def generate_top_words_plot(freq, top=10):
#     if not freq:
#         return None
#     common = freq.most_common(top)
#     words, counts = zip(*common)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.barh(words[::-1], counts[::-1], color="#4A90E2")
#     ax.set_xlabel("Frequency")
#     ax.set_title("Top 10 Words")
#     plt.tight_layout()
#     return fig

# def generate_pos_pie(pos_tags):
#     if not pos_tags:
#         return None
#     pos_counts = Counter(tag for _, tag in pos_tags)
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.pie(pos_counts.values(), labels=pos_counts.keys(), autopct='%1.1f%%', startangle=140)
#     ax.axis('equal')
#     ax.set_title("POS Distribution")
#     return fig

# # ====================== MAIN PROCESSING ======================
# def process_text(text_input, file, lowercase, remove_stop, num_summary):
#     text = extract_text_from_file(file) if file else text_input
#     if not text or not text.strip():
#         return ["No text provided"] * 14

#     text = re.sub(r'\s+', ' ', text.strip())
#     words = word_tokenize(text)
#     sentences = sent_tokenize(text)

#     if lowercase:
#         words = [w.lower() for w in words]
#         text = text.lower()

#     filtered_words = [w for w in words if w.isalpha() and (not remove_stop or w.lower() not in stop_words)]

#     # POS tagging (used for lemmatization and POS tab)
#     pos_tags = pos_tag(words)

#     # Stemming & POS-aware Lemmatization
#     stemmed = [[w, stemmer.stem(w)] for w in filtered_words]
#     lemmatized = []
#     for word, tag in pos_tags:
#         if word.isalpha() and (not remove_stop or word.lower() not in stop_words):
#             wn_pos = get_wordnet_pos(tag)
#             lemma = lemmatizer.lemmatize(word, wn_pos)
#             lemmatized.append([word, lemma])

#     # Stats
#     stats = f"""
# **Words:** {len(words)}  **Sentences:** {len(sentences)}  **Unique words:** {len(set(filtered_words))}
# **Characters:** {len(text)}  **Avg. sentence length:** {round(len(words)/len(sentences), 1) if sentences else 0} words
# """

#     # POS & Entities
#     pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS"])
#     entities = extract_entities(text)

#     # Sentiment & Readability
#     blob = TextBlob(text)
#     sentiment = f"**Polarity:** {round(blob.sentiment.polarity, 3)}  **Subjectivity:** {round(blob.sentiment.subjectivity, 3)}"
#     readability = f"""
# **Flesch Reading Ease:** {round(textstat.flesch_reading_ease(text), 1)}  
# **Flesch-Kincaid Grade:** {round(textstat.flesch_kincaid_grade(text), 1)}  
# **Gunning Fog Index:** {round(textstat.gunning_fog(text), 1)}
# """

#     # Keywords & Summary
#     kw_extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=15)
#     keywords = kw_extractor.extract_keywords(text)
#     keywords_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])

#     summary = simple_extractive_summary(text, num_summary)

#     # Visualizations
#     wc_img = generate_wordcloud_img(text, remove_stop)
#     freq = Counter(filtered_words)
#     top_plot = generate_top_words_plot(freq)
#     pos_pie = generate_pos_pie(pos_tags)

#     # Export JSON
#     export_data = {
#         "stats": stats,
#         "keywords": keywords,
#         "summary": summary,
#         "sentiment": sentiment,
#         "readability": readability
#     }

#     return (
#         stats,
#         " • ".join([f"`{w}`" for w in filtered_words[:50]]) + (" ..." if len(filtered_words) > 50 else ""),
#         pd.DataFrame(stemmed, columns=["Original", "Stemmed"]),
#         pd.DataFrame(lemmatized, columns=["Original", "Lemma"]),
#         pos_df,
#         pd.DataFrame(entities, columns=["Entity", "Type"]),
#         sentiment,
#         readability,
#         keywords_df,
#         summary,
#         wc_img,
#         top_plot,
#         pos_pie,
#         json.dumps(export_data, indent=2)
#     )

# # ====================== GRADIO UI ======================
# with gr.Blocks(title="Advanced NLP Playground") as demo:
#     gr.Markdown("# 🧠 Advanced NLP Playground")
#     gr.Markdown("**Portfolio Project** • POS • NER • Sentiment • Keywords • Word Cloud • Summarization • File Upload")
#     gr.Markdown("*Note: First run may take 10–30 seconds to download NLTK models automatically*")

#     with gr.Row():
#         with gr.Column(scale=3):
#             text_input = gr.Textbox(
#                 lines=6,
#                 label="✍️ Enter text",
#                 placeholder="Paste your text here...",
#                 value="Artificial Intelligence is transforming the world at an unprecedented pace. Researchers at OpenAI and Google DeepMind are pushing the boundaries of what machines can learn."
#             )
#         with gr.Column(scale=1):
#             file_input = gr.File(label="📁 Upload .txt or .pdf", file_types=[".txt", ".pdf"])

#     with gr.Row():
#         lowercase_cb = gr.Checkbox(label="Convert to lowercase", value=True)
#         stop_cb = gr.Checkbox(label="Remove stopwords", value=True)
#         summary_slider = gr.Slider(1, 8, value=3, step=1, label="Summary sentences")

#     with gr.Row():
#         analyze_btn = gr.Button("🚀 Analyze Text", variant="primary", size="large")
#         clear_btn = gr.ClearButton([text_input, file_input])

#     gr.Examples(
#         examples=[
#             ["Machine learning is revolutionizing healthcare and finance."],
#             ["The quick brown fox jumps over the lazy dog. This is a classic pangram."],
#             ["Climate change is one of the biggest challenges facing humanity today."],
#         ],
#         inputs=text_input
#     )

#     with gr.Tabs():
#         with gr.Tab("📊 Overview"):
#             stats_out = gr.Markdown()
#             tokens_preview = gr.Markdown()

#         with gr.Tab("🔤 Tokenization"):
#             with gr.Row():
#                 stem_df = gr.Dataframe(label="Stemming")
#                 lemma_df = gr.Dataframe(label="POS-aware Lemmatization")

#         with gr.Tab("🏷️ POS & Entities"):
#             with gr.Row():
#                 pos_df_out = gr.Dataframe(label="Part-of-Speech")
#                 entities_df = gr.Dataframe(label="Named Entity Recognition")

#         with gr.Tab("❤️ Sentiment & Readability"):
#             sentiment_out = gr.Markdown()
#             readability_out = gr.Markdown()

#         with gr.Tab("🔑 Keywords & Summary"):
#             keywords_df_out = gr.Dataframe(label="Keywords (YAKE)")
#             summary_out = gr.Markdown(label="Extractive Summary")

#         with gr.Tab("📈 Visualizations"):
#             with gr.Row():
#                 wc_out = gr.Image(label="Word Cloud", height=520)
#                 with gr.Column():
#                     top_words_plot = gr.Plot(label="Top 10 Words")
#                     pos_pie_out = gr.Plot(label="POS Distribution")

#         with gr.Tab("💾 Export"):
#             export_json = gr.JSON(label="Full Results JSON")

#     # ====================== EVENT ======================
#     analyze_btn.click(
#         process_text,
#         inputs=[text_input, file_input, lowercase_cb, stop_cb, summary_slider],
#         outputs=[
#             stats_out, tokens_preview, stem_df, lemma_df,
#             pos_df_out, entities_df, sentiment_out, readability_out,
#             keywords_df_out, summary_out, wc_out, top_words_plot,
#             pos_pie_out, export_json
#         ]
#     )

# # Removed theme from Blocks() to avoid Gradio 6.0+ warning; pass to launch() if needed
# demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


import gradio as gr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import pandas as pd
from textblob import TextBlob
import textstat
from yake import KeywordExtractor
from PyPDF2 import PdfReader
import json
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from collections import Counter

# ────────────────────────────────────────────────
# Hugging Face + sentence-transformers (all added features)
# ────────────────────────────────────────────────
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load models once (with graceful fallback)
abstractive_summarizer = None
qa_pipeline = None
emotion_classifier = None
embedder = None

try:
    abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
except Exception as e:
    print("Abstractive summarizer failed:", e)

try:
    qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2", device=-1)
except Exception as e:
    print("QA pipeline failed:", e)

try:
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1, device=-1)
except Exception as e:
    print("Emotion classifier failed:", e)

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer embedder loaded")
except Exception as e:
    print("Embedder failed:", e)

# Global RAG state
doc_chunks = []
chunk_embeddings = None

# ====================== NLTK SETUP ======================
def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'stopwords']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except:
            pass

setup_nltk()

# ====================== CORE FUNCTIONS ======================
def get_ner_html(text):
    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        chunks = ne_chunk(tags)
        html = []
        colors = {"GPE": "#7aecec", "PERSON": "#aa9cfc", "ORGANIZATION": "#feca74",
                  "LOCATION": "#9cc9cc", "DATE": "#ff9561", "MONEY": "#82dc82", "FACILITY": "#6a0572"}
        for chunk in chunks:
            if isinstance(chunk, Tree):
                ent = " ".join(w for w, _ in chunk.leaves())
                typ = chunk.label()
                c = colors.get(typ, "#e0e0e0")
                html.append(f'<mark style="background:{c};padding:0.2em 0.5em;border-radius:0.5em;">{ent} <small>{typ}</small></mark>')
            else:
                html.append(f" {chunk[0]} ")
        return "".join(html)
    except Exception as e:
        return f'<div style="color:#d32f2f;padding:16px;background:#ffebee;border-radius:8px;">NER unavailable: {str(e)}</div>'

def create_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0,1], 'y': [0,1]},
                                 gauge={'axis': {'range': [-1,1]}, 'bar': {'color': "#2c3e50"},
                                        'steps': [{'range':[-1,-0.2],'color':"#ff4b4b"},
                                                  {'range':[-0.2,0.2],'color':"#ffa500"},
                                                  {'range':[0.2,1],'color':"#00d1b2"}]}))
    fig.update_layout(height=280, margin=dict(l=30,r=30,t=50,b=20), title="Sentiment Polarity")
    return fig

def create_pos_distribution(pos_tags):
    if not pos_tags: return None
    counts = dict(Counter(t for _, t in pos_tags))
    df = pd.DataFrame(list(counts.items()), columns=['Tag','Count']).sort_values('Count', ascending=False).head(10)
    fig = px.bar(df, x='Tag', y='Count', color='Count', color_continuous_scale='Plasma', title="Top POS Tags")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def create_readability_radar(text):
    try:
        metrics = {
            'Reading Ease': textstat.flesch_reading_ease(text),
            'Grade Level': textstat.flesch_kincaid_grade(text) * 5,
            'Complexity': min(textstat.difficult_words(text), 100),
            'Syllables': min(textstat.syllable_count(text)/2, 100)
        }
        fig = go.Figure(go.Scatterpolar(r=list(metrics.values()), theta=list(metrics.keys()),
                                        fill='toself', line_color='#4A90E2'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False,
                          title="Readability Fingerprint", height=350)
        return fig
    except:
        return None

def highlight_emotions(text):
    if emotion_classifier is None:
        return "<p>Emotion view unavailable</p>"
    sents = sent_tokenize(text)
    parts = []
    cmap = {'joy':'#d4f4dd','sadness':'#e3f2fd','anger':'#ffebee','fear':'#fff3e0','love':'#fce4ec','surprise':'#f3e5f5'}
    for s in sents:
        if len(s.strip()) < 15:
            parts.append(s)
            continue
        try:
            res = emotion_classifier(s)[0][0]
            label, score = res['label'], res['score']
            bg = cmap.get(label.lower(), '#f5f5f5')
            parts.append(f'<span style="background:{bg};padding:4px 8px;border-radius:6px;margin:0 4px 8px 0;display:inline-block;" title="{label} {score:.0%}">{s}</span>')
        except:
            parts.append(s)
    return " ".join(parts)

def abstractive_summary(text):
    if abstractive_summarizer is None or len(text.split()) < 60:
        return text if len(text.split()) < 60 else "[Abstractive summary unavailable]"
    try:
        res = abstractive_summarizer(text, max_length=140, min_length=40, do_sample=False, truncation=True)
        return res[0]['summary_text']
    except:
        return "[Summarization error]"

# ─── RAG Chat Preparation & Logic ────────────────────────────────────────
def prepare_rag(text):
    global doc_chunks, chunk_embeddings
    doc_chunks = []
    chunk_embeddings = None
    if not text.strip() or embedder is None:
        return
    chunk_size, overlap = 450, 100
    i = 0
    while i < len(text):
        doc_chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    if doc_chunks:
        chunk_embeddings = embedder.encode(doc_chunks, convert_to_tensor=True, device='cpu')
        print(f"RAG prepared: {len(doc_chunks)} chunks")

def rag_chat(question, history):
    if embedder is None or not doc_chunks or chunk_embeddings is None:
        return "[Analyze a document first]"

    q_emb = embedder.encode(question, convert_to_tensor=True, device='cpu')
    hits = util.semantic_search(q_emb, chunk_embeddings, top_k=3)[0]
    context = "\n\n".join(doc_chunks[h['corpus_id']] for h in hits)

    if qa_pipeline is None:
        return "[QA model unavailable]"

    try:
        prompt = f"Previous: {history[-1][1] if history else ''}\nQuestion: {question}" if history else question
        res = qa_pipeline(question=prompt, context=context[:3800])
        return f"**Answer:** {res['answer']}\nConfidence: {res['score']:.0%}\n\n**Source context snippet:**\n{context[:500]}..."
    except:
        return "[Could not answer]"

# ─── Main Processing ──────────────────────────────────────────────────────
def process_everything(text_input, file_input, num_sentences):
    num_sentences = int(num_sentences)
    text = ""
    if file_input is not None:
        try:
            fp = file_input.name if hasattr(file_input, 'name') else file_input
            if str(fp).lower().endswith('.pdf'):
                reader = PdfReader(fp)
                text = " ".join(p.extract_text() or "" for p in reader.pages if p.extract_text())
            else:
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
        except Exception as e:
            text = f"[File error: {str(e)}]"
    else:
        text = text_input.strip()

    if not text:
        return ["No text"] * 12

    blob = TextBlob(text)
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    stats = f"""
    <div style='display:flex;justify-content:space-around;text-align:center;background:#f8f9fa;padding:20px;border-radius:10px;border:1px solid #ddd;'>
        <div><small>SENTENCES</small><br><b style='font-size:1.6em;'>{len(sentences)}</b></div>
        <div><small>WORDS</small><br><b style='font-size:1.6em;'>{len(tokens)}</b></div>
        <div><small>COMPLEX WORDS</small><br><b style='font-size:1.6em;'>{textstat.difficult_words(text) if 'textstat' in globals() else 'N/A'}</b></div>
    </div>"""

    wc_array = None
    try:
        stop = set(stopwords.words('english'))
        clean = " ".join(w for w in tokens if w.lower() not in stop and w.isalpha())
        wc = WordCloud(width=800,height=400,background_color="white",colormap='winter').generate(clean)
        wc_array = wc.to_array()
    except:
        pass

    kw_df = pd.DataFrame(columns=["Keyword","Relevance"])
    try:
        kw = KeywordExtractor(lan="en", n=1, top=10).extract_keywords(text)
        kw_df = pd.DataFrame(kw, columns=["Keyword","Relevance"])
    except:
        pass

    sentiment_json = json.dumps({
        "polarity": round(blob.sentiment.polarity,4),
        "subjectivity": round(blob.sentiment.subjectivity,4)
    }, indent=2)

    prepare_rag(text)  # prepare for chat

    return (
        stats,
        get_ner_html(text),
        create_sentiment_gauge(blob.sentiment.polarity),
        create_pos_distribution(pos_tags),
        create_readability_radar(text),
        wc_array,
        " ".join(sentences[:num_sentences]) if sentences else "[No sentences]",
        abstractive_summary(text),
        kw_df,
        pd.DataFrame(pos_tags, columns=["Token","POS Tag"]) if pos_tags else pd.DataFrame(),
        sentiment_json,
        highlight_emotions(text)
    )

# ====================== GRADIO UI ======================
with gr.Blocks(title="NLP Nexus Pro") as demo:
    gr.HTML("<h1 style='text-align:center; color:#2c3e50;'>🌐 NLP Nexus Pro</h1>")

    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(label="Input Text", lines=10, value="Elon Musk's SpaceX launched...")
            file_in = gr.File(label="Upload .txt or .pdf")
        with gr.Column(scale=2):
            gr.Markdown("### Controls")
            sum_slider = gr.Slider(1, 15, 4, step=1, label="Extractive Summary Sentences")
            analyze_btn = gr.Button("🚀 Analyze", variant="primary")
            stats_out = gr.HTML()

    with gr.Tabs():
        gr.Tab("🏷️ Named Entities").add(ner_out := gr.HTML())
        with gr.Tab("📊 Sentiment & Emotions"):
            with gr.Row():
                sentiment_out = gr.Plot()
                emotion_out = gr.HTML()
            gr.Markdown("**Legend:** +1.0 = Very Positive • 0 = Neutral • -1.0 = Very Negative")
        with gr.Tab("📈 Readability & Grammar"):
            with gr.Row():
                radar_out = gr.Plot()
                pos_out = gr.Plot()
        gr.Tab("☁️ Word Cloud").add(wc_out := gr.Image())
        with gr.Tab("🔑 Summaries & Keywords"):
            with gr.Row():
                extract_sum = gr.Textbox(label="Extractive Summary", lines=8)
                abstract_sum = gr.Textbox(label="Abstractive Summary", lines=8)
            kw_out = gr.Dataframe(label="Keywords")
        with gr.Tab("❓ Ask Questions (single-turn)"):
            q_in = gr.Textbox(label="Question", placeholder="What is the main goal?")
            q_out = gr.Textbox(label="Answer", lines=5, interactive=False)
            ask_btn = gr.Button("Ask")
        with gr.Tab("💬 Chat with Document (multi-turn RAG)"):
            gr.Markdown("Ask multiple questions — remembers context loosely")
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(placeholder="Ask anything about the document...", container=False)
            clear = gr.Button("Clear Chat")

        with gr.Tab("💾 Export"):
            raw_df = gr.Dataframe(label="Tokens + POS")
            json_out = gr.JSON(label="Sentiment Data")

    # ─── Events ───────────────────────────────────────────────────────────────
    analyze_btn.click(
        process_everything,
        inputs=[text_in, file_in, sum_slider],
        outputs=[stats_out, ner_out, sentiment_out, pos_out, radar_out, wc_out,
                 extract_sum, abstract_sum, kw_out, raw_df, json_out, emotion_out]
    )

    def single_qa(q, ctx):
        return answer_question(q, ctx) if qa_pipeline else "[QA unavailable]"

    ask_btn.click(single_qa, [q_in, text_in], q_out)

    def chat_respond(message, history):
        reply = rag_chat(message, history)
        history.append((message, reply))
        return "", history

    msg.submit(chat_respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())