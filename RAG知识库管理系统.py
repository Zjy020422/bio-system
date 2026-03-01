#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""api_upgrade.py: 基因信息检索脚本，结合 RAG 与 Chat API，通过 Gradio 提供网页界面。"""
import gradio as gr
from openai import OpenAI
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import logging
import sys
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import io
from PIL import Image
import base64

# 加载 .env 文件
load_dotenv()

# 日志基础配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s [%(levelname)s] %(message)s')

# 从环境变量加载配置
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logging.error("请设置环境变量 OPENAI_API_KEY")
    sys.exit(1)
API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.chatanywhere.tech/v1")
CORPUS_DIR = os.getenv("CORPUS_DIR", "docs")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RAG_K = int(os.getenv("RAG_K", "3"))
print(API_KEY)
# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# 简单的 RAG 检索器类
class SimpleRAGRetriever:
    def __init__(self, documents, k=3):
        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.doc_vectors = self.vectorizer.fit_transform([doc.page_content for doc in documents])
    
    def get_relevant_documents(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-self.k:][::-1]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

    def get_relevant_documents_with_scores(self, query):
        """返回相关文档及其相似度得分"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-self.k:][::-1]
        results = []
        for i in top_indices:
            if similarities[i] > 0:
                results.append({
                    'document': self.documents[i],
                    'score': float(similarities[i]),
                    'source': self.documents[i].metadata.get('source', 'unknown')
                })
        return results

# RAG: 初始化检索器，加载 CORPUS_DIR 目录中的文本，并构建向量索引
def init_rag(corpus_dir=CORPUS_DIR):
    """初始化 RAG 检索器，加载语料目录并构建向量索引"""
    logging.info(f"Initializing RAG retriever with corpus: {corpus_dir}")
    if not os.path.isdir(corpus_dir):
        logging.error(f"语料目录 {corpus_dir} 不存在，请创建并添加 .txt 文本文件")
        sys.exit(1)
    docs = []
    for fname in os.listdir(corpus_dir):
        fpath = os.path.join(corpus_dir, fname)
        if os.path.isfile(fpath) and fpath.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": fname}))
    
    if not docs:
        logging.error(f"在目录 {corpus_dir} 中未找到任何 .txt 文件")
        sys.exit(1)
    
    logging.info(f"Found {len(docs)} documents, creating text chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    logging.info(f"Created {len(chunks)} text chunks, building TF-IDF index...")
    retriever = SimpleRAGRetriever(chunks, k=RAG_K)
    logging.info("RAG retriever initialized successfully!")
    return retriever

# 全局初始化 RAG 检索器
rag_retriever = init_rag(CORPUS_DIR)
def get_gene_info(gene_name):
    """调用API获取基因信息（结合检索上下文）"""
    gene_name = gene_name.strip()
    logging.info(f"Retrieving RAG documents for gene: {gene_name}")
    retrieved_docs = rag_retriever.get_relevant_documents(gene_name)
    if not retrieved_docs:
        logging.warning("No relevant documents found in corpus.")
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    system_prompt = """You are a biomedical expert. Given a gene name, provide:
1. Gene function (in English)
2. Associated diseases (in English)
3. Current treatment approaches (in English)
Format the response in clear bullet points. If gene is invalid, state 'Gene not found'."""
    if context:
        system_prompt += f"\n\nBackground Context:\n{context}"

    logging.debug("System prompt constructed, sending to chat completion API")
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Gene: {gene_name}"}
        ],
        temperature=0.8,
    )
    logging.info("Received completion response")
    return completion.choices[0].message.content

def create_similarity_chart(retrieved_results):
    """创建相似度得分条形图"""
    if not retrieved_results:
        # 如果没有结果，返回空图表
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
        ax.text(0.5, 0.5, 'No relevant documents found',
                ha='center', va='center', fontsize=14, color='#666')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    else:
        # 创建条形图
        sources = [r['source'] for r in retrieved_results]
        scores = [r['score'] for r in retrieved_results]

        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')

        # 使用浅蓝色系配色
        colors = ['#4FC3F7', '#81D4FA', '#B3E5FC'][:len(sources)]
        bars = ax.barh(sources, scores, color=colors, edgecolor='#01579B', linewidth=1.5)

        ax.set_xlabel('Similarity Score', fontsize=13, fontweight='bold', color='#01579B')
        ax.set_title('📊 Document Relevance Scores', fontsize=15, fontweight='bold',
                     color='#01579B', pad=15)
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1)

        # 设置网格
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 在条形上添加数值标签
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + max(scores) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', ha='left', va='center', fontsize=11,
                   fontweight='bold', color='#01579B')

        # 设置y轴标签样式
        ax.tick_params(axis='y', labelsize=11, colors='#01579B')
        ax.tick_params(axis='x', labelsize=10, colors='#666')

        # 设置边框
        for spine in ax.spines.values():
            spine.set_edgecolor('#B3E5FC')
            spine.set_linewidth(2)

    # 将图表转换为PIL Image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def process_gene_input(gene_name):
    """处理基因输入并返回格式化结果、摘要和可视化图表"""
    gene_name = gene_name.strip()
    if not gene_name:
        return "❌ 请输入有效的基因名称", "No query", None

    logging.info(f"Processing gene input: {gene_name}")
    try:
        # 获取带有得分的检索结果
        retrieved_results = rag_retriever.get_relevant_documents_with_scores(gene_name)

        # 生成摘要信息
        summary = f"""📊 Query Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Query Gene: {gene_name.upper()}
📚 Documents Retrieved: {len(retrieved_results)}
🎯 Model Used: {CHAT_MODEL}
📖 Knowledge Base: {len(rag_retriever.documents)} total chunks"""

        if retrieved_results:
            avg_score = np.mean([r['score'] for r in retrieved_results])
            summary += f"\n📈 Average Relevance Score: {avg_score:.4f}"
            summary += "\n\n📄 Retrieved Sources:"
            for i, r in enumerate(retrieved_results, 1):
                summary += f"\n  {i}. {r['source']} (Score: {r['score']:.4f})"

        # 创建相似度图表
        chart = create_similarity_chart(retrieved_results)

        # 调用API获取基因信息
        response = get_gene_info(gene_name)
        if "not found" in response.lower():
            return f"⚠️ Error: {response}", summary, chart

        result_text = f"""🧬 Gene: {gene_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{response}"""

        return result_text, summary, chart

    except Exception as e:
        logging.error(f"Error processing gene input: {e}")
        return f"❌ API Error: {str(e)}", f"Error occurred for query: {gene_name}", None

# 将背景图片转换为base64编码
def get_background_image_base64():
    """读取本地背景图片并转换为base64编码"""
    try:
        bg_path = os.path.join(os.path.dirname(__file__), "background.png")
        with open(bg_path, "rb") as f:
            img_data = f.read()
        b64_data = base64.b64encode(img_data).decode()
        return f"data:image/png;base64,{b64_data}"
    except Exception as e:
        logging.warning(f"Failed to load background image: {e}")
        return ""

# 获取背景图片的base64编码
bg_image_b64 = get_background_image_base64()

# 自定义CSS样式
custom_css = f"""
/* 为整个页面添加背景 */
body, #root, .gradio-container {{
    background-image: url('{bg_image_b64}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

/* 在容器前添加一个背景层 */
.gradio-container::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url('{bg_image_b64}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    z-index: -1;
}}

.gradio-container {{
    background-color: transparent !important;
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}}

/* 为内容区域添加半透明背景 */
.contain {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
}}

.input-box {{
    border-radius: 10px;
    border: 2px solid #4FC3F7 !important;
}}
.output-box {{
    border-radius: 10px;
    background-color: #f5f5f5;
}}
.summary-box {{
    border-radius: 10px;
    background-color: #E3F2FD;
    border: 2px solid #4FC3F7;
    padding: 15px;
}}
h1 {{
    color: #2c3e50;
    text-align: center;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}}
/* 按钮样式 */
.primary-button {{
    background-color: #B3E5FC !important;
    border-color: #81D4FA !important;
    color: #01579B !important;
    font-weight: bold !important;
}}
.primary-button:hover {{
    background-color: #81D4FA !important;
    border-color: #4FC3F7 !important;
}}
"""

# 创建Gradio Blocks界面
with gr.Blocks(title="RAG Gene Knowledge Base") as iface:
    # 添加自定义CSS
    gr.HTML(f"<style>{custom_css}</style>")
    # 添加固定背景层
    gr.HTML(f"""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: url('{bg_image_b64}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        z-index: -1;
        pointer-events: none;
    "></div>
    """)

    # 标题
    gr.Markdown("""
    # 🧬 Gene Information Query System
    ### RAG-Powered Knowledge Base with Multi-Agent Support
    Enter a gene name to get detailed functional information, associated diseases, and treatment approaches.
    """)

    # 项目摘要文字框
    with gr.Row():
        gr.Textbox(
            label="📋 Project Summary",
            value="""This is a RAG (Retrieval-Augmented Generation) based Gene Knowledge Base Management System.
It combines web interface, RAG technology, and multi-agent architecture to provide intelligent gene information retrieval.

Key Features:
• RAG-powered document retrieval with TF-IDF vectorization
• OpenAI GPT-4o-mini integration for intelligent responses
• Real-time similarity scoring and visualization
• Multi-source knowledge base (cancer genes, genetic diseases, gene database)
• Interactive query summary and relevance analysis

The system retrieves relevant documents from the knowledge base and uses AI to generate comprehensive gene information including functions, associated diseases, and treatment approaches.""",
            lines=8,
            interactive=False,
            elem_classes="summary-box"
        )

    with gr.Row():
        # 左侧输入区域
        with gr.Column(scale=1):
            gene_input = gr.Textbox(
                label="🔍 Enter Gene Name",
                placeholder="e.g. Etv6, Smyd3, Hspa8...",
                max_lines=1,
                elem_classes="input-box"
            )
            submit_btn = gr.Button("🚀 Query", variant="primary", size="lg", elem_classes="primary-button")

            gr.Markdown("### 📝 Quick Examples")
            gr.Examples(
                examples=[['Etv6'], ['Smyd3'], ['Hspa8']],
                inputs=gene_input,
                label="Click to try"
            )

        # 右侧摘要区域
        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                label="📊 Query Summary",
                lines=12,
                interactive=False,
                elem_classes="output-box"
            )

    # 中间主要结果区域
    with gr.Row():
        result_output = gr.Textbox(
            label="🧬 Gene Information Details",
            lines=15,
            interactive=False,
            elem_classes="output-box"
        )

    # 底部结果图区域
    with gr.Row():
        chart_output = gr.Image(
            label="📈 Document Relevance Visualization",
            type="pil",
            height=300
        )

    # 连接按钮点击事件
    submit_btn.click(
        fn=process_gene_input,
        inputs=gene_input,
        outputs=[result_output, summary_output, chart_output]
    )

    # 也支持按Enter键提交
    gene_input.submit(
        fn=process_gene_input,
        inputs=gene_input,
        outputs=[result_output, summary_output, chart_output]
    )

    # 添加结果图片展示区域
    gr.Markdown("""
    ---
    ## 📊 Analysis Results Gallery
    ### Research Findings and Visualizations
    """)

    with gr.Row():
        # 获取results文件夹中的所有图片
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        if os.path.exists(results_dir):
            image_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()  # 按文件名排序

            # 使用Gallery组件展示图片
            gr.Gallery(
                value=image_files,
                label="Research Analysis Results",
                columns=3
            )

    # 添加页脚信息
    gr.Markdown("""
    ---
    💡 **Powered by**: RAG (Retrieval-Augmented Generation) + OpenAI GPT-4o-mini + Multi-Agent System
    📚 **Knowledge Base**: Cancer genes, genetic diseases, and gene database
    """)

if __name__ == "__main__":
    # 设置share=False只在本地运行，share=True可以生成公网链接
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))