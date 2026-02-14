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
from nltk.stem import WordNetLemmatizer
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
import os
import tempfile

# ====================== NLTK SETUP ======================
def setup_nltk():
    resources = [
        'punkt', 'punkt_tab',
        'wordnet', 'omw-1.4',
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker', 'maxent_ne_chunker_tab',
        'words',
        'stopwords'
    ]
    missing = []
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            missing.append((res, str(e)))
    if missing:
        print("Warning: Some NLTK resources could not be downloaded automatically:")
        for r, err in missing:
            print(f"  - {r}: {err}")
        print("Please run manually:")
        print("import nltk; nltk.download('maxent_ne_chunker_tab')  # and others if needed")

setup_nltk()

# ====================== ANALYTICS & VISUALS ======================
def get_ner_html(text):
    """Wraps entities in HTML/CSS for visual highlighting."""
    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        chunks = ne_chunk(tags)
        
        html_output = []
        colors = {
            "GPE": "#7aecec", "PERSON": "#aa9cfc", "ORGANIZATION": "#feca74",
            "LOCATION": "#9cc9cc", "DATE": "#ff9561", "MONEY": "#82dc82",
            "FACILITY": "#6a0572"
        }

        for chunk in chunks:
            if isinstance(chunk, Tree):
                ent_text = " ".join([word for word, tag in chunk.leaves()])
                ent_type = chunk.label()
                color = colors.get(ent_type, "#e0e0e0")
                html_output.append(
                    f'<mark style="background: {color}; padding: 0.2em 0.5em; margin: 0 0.2em; '
                    f'border-radius: 0.5em; line-height: 2.5; font-weight: bold; font-family: sans-serif;">'
                    f'{ent_text} <span style="font-size: 0.7em; text-transform: uppercase; opacity: 0.7;">{ent_type}</span></mark>'
                )
            else:
                html_output.append(f" {chunk[0]} ")
        
        return "".join(html_output)
    except LookupError as e:
        return (
            '<div style="color: #d32f2f; padding: 16px; background: #ffebee; border-radius: 8px; border: 1px solid #ef9a9a;">'
            f'<strong>NER unavailable:</strong> {str(e)}<br>'
            'Run in terminal/Python: <code>import nltk; nltk.download("maxent_ne_chunker_tab")</code>'
            '</div>'
        )
    except Exception as e:
        return f'<div style="color: #f57c00;">NER processing error: {str(e)}</div>'

def create_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [-1, -0.2], 'color': "#ff4b4b"},
                {'range': [-0.2, 0.2], 'color': "#ffa500"},
                {'range': [0.2, 1], 'color': "#00d1b2"}
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20), title="Emotional Polarity")
    return fig

def create_pos_distribution(pos_tags):
    if not pos_tags:
        return None
    counts = dict(Counter(tag for _, tag in pos_tags))
    df = pd.DataFrame(list(counts.items()), columns=['Tag', 'Count']).sort_values('Count', ascending=False).head(10)
    fig = px.bar(df, x='Tag', y='Count', color='Count', color_continuous_scale='Plasma', title="Grammar Structure (Top 10)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def create_readability_radar(text):
    try:
        if textstat is None:
            return None
        metrics = {
            'Reading Ease': textstat.flesch_reading_ease(text),
            'Grade Level': textstat.flesch_kincaid_grade(text) * 5,
            'Complexity': min(textstat.difficult_words(text), 100),
            'Syllables': min(textstat.syllable_count(text) / 2, 100)
        }
        fig = go.Figure(data=go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            line_color='#4A90E2'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          showlegend=False, title="Readability Fingerprint", height=350)
        return fig
    except Exception:
        return None

# ====================== MAIN ENGINE ======================
def process_everything(text_input, file_input, num_sentences):
    num_sentences = int(num_sentences)

    text = ""
    if file_input is not None:
        try:
            file_path = file_input.name if hasattr(file_input, 'name') else file_input
            if str(file_path).lower().endswith('.pdf'):
                reader = PdfReader(file_path)
                text = " ".join(page.extract_text() or "" for page in reader.pages if page.extract_text())
            else:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
        except Exception as e:
            text = f"[Error reading file: {str(e)}]"
    else:
        text = text_input.strip()

    if not text:
        return ["No text provided."] * 10

    blob = TextBlob(text)
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Stats Header
    stats_html = f"""
    <div style='display: flex; justify-content: space-around; text-align: center; background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #ddd;'>
        <div><small style='color: #666;'>SENTENCES</small><br><b style='font-size: 1.5em; color: #2c3e50;'>{len(sentences)}</b></div>
        <div><small style='color: #666;'>WORDS</small><br><b style='font-size: 1.5em; color: #2c3e50;'>{len(tokens)}</b></div>
        <div><small style='color: #666;'>COMPLEX WORDS</small><br><b style='font-size: 1.5em; color: #2c3e50;'>{textstat.difficult_words(text) if textstat else "N/A"}</b></div>
    </div>
    """

    # Word Cloud
    wc_array = None
    try:
        if WordCloud is not None:
            stop = set(stopwords.words('english'))
            clean_text = " ".join([w for w in tokens if w.lower() not in stop and w.isalpha()])
            wc = WordCloud(width=800, height=400, background_color="white", colormap='winter').generate(clean_text)
            wc_array = wc.to_array()
    except Exception:
        pass

    # Keywords
    kw_df = pd.DataFrame(columns=["Keyword", "Relevance"])
    try:
        if KeywordExtractor is not None:
            kw_extractor = KeywordExtractor(lan="en", n=1, top=10)
            keywords = kw_extractor.extract_keywords(text)
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Relevance"])
    except Exception:
        pass

    # Safer JSON export
    sentiment_json = json.dumps({
        "polarity": round(blob.sentiment.polarity, 4),
        "subjectivity": round(blob.sentiment.subjectivity, 4)
    }, indent=2)

    return (
        stats_html,
        get_ner_html(text),
        create_sentiment_gauge(blob.sentiment.polarity),
        create_pos_distribution(pos_tags),
        create_readability_radar(text),
        wc_array,
        " ".join(sentences[:num_sentences]) if sentences else "[No sentences found]",
        kw_df,
        pd.DataFrame(pos_tags, columns=["Token", "POS Tag"]) if pos_tags else pd.DataFrame(),
        sentiment_json
    )

# ====================== GRADIO INTERFACE ======================
with gr.Blocks(title="NLP PlayGround") as demo:
    gr.HTML("<h1 style='text-align: center; font-family: sans-serif; color: #2c3e50;'>🌐 NLP Nexus Pro</h1>")
    
    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(label="Analysis Input", lines=10, 
                                value="Elon Musk's SpaceX launched a Falcon 9 rocket from Cape Canaveral in Florida. The mission aims to deploy Starlink satellites into orbit to provide global internet access.")
            file_in = gr.File(label="Upload Source File (txt or pdf)")
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Settings")
            sum_slider = gr.Slider(1, 10, value=3, step=1, label="Summary Depth (sentences)")
            analyze_btn = gr.Button("🚀 Run Full Diagnostic", variant="primary")
            stats_out = gr.HTML()

    with gr.Tabs():
        with gr.Tab("🏷️ Named Entity Map"):
            ner_out = gr.HTML()
            
        with gr.Tab("📊 Sentiment & Tone"):
            with gr.Row():
                sentiment_out = gr.Plot()
                gr.Markdown("### Polarity Legend\n- **+1.0**: Highly Positive\n- **0.0**: Neutral\n- **-1.0**: Highly Negative")

        with gr.Tab("📈 Linguistic Fingerprint"):
            with gr.Row():
                radar_out = gr.Plot()
                pos_out = gr.Plot()
        
        with gr.Tab("☁️ Word Cloud"):
            wc_out = gr.Image()
        
        with gr.Tab("🔑 Summary & Keywords"):
            with gr.Row():
                sum_out = gr.Textbox(label="Extractive Summary", lines=10)
                kw_out = gr.Dataframe(label="Key Terms")
                    
        with gr.Tab("💾 Export"):
            with gr.Row():
                raw_df = gr.Dataframe(label="Full Token Log")
                json_out = gr.JSON(label="Metadata")

    analyze_btn.click(
        process_everything,
        inputs=[text_in, file_in, sum_slider],
        outputs=[stats_out, ner_out, sentiment_out, pos_out, radar_out, wc_out, sum_out, kw_out, raw_df, json_out]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())