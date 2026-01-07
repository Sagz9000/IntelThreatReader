from flask import Flask, render_template, jsonify, request
import feedparser
import sqlite3
import json
import os
import logging
import sys
import threading
import time
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
import ollama
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_chroma import Chroma
import chromadb
import requests # Added global import for requests
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Jinja2 Filter for JSON
@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return json.loads(s)
    except:
        return []

# Constants
DB_FILE = 'analysis_db.sqlite'
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
LLM_MODEL = "gemma3:4b"
VISION_MODEL = "llama3.2-vision"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe" # Using better embedding model
RSS_FEED_URL = "https://www.bleepingcomputer.com/feed/"

# Globals
llm = None
embeddings = None
vectorstore = None

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            link TEXT,
            published TEXT,
            summary TEXT,
            content TEXT,
            media TEXT, -- JSON list of image URLs
            ai_category TEXT,
            ai_summary TEXT,
            ai_risk_level TEXT,
            analyzed BOOLEAN DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Simple migration: check if media column exists
    try:
        c.execute("SELECT media FROM articles LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating DB: Adding media column")
        c.execute("ALTER TABLE articles ADD COLUMN media TEXT")
        
    try:
        c.execute("SELECT user_tags FROM articles LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating DB: Adding user_tags column")
        c.execute("ALTER TABLE articles ADD COLUMN user_tags TEXT")

    try:
        c.execute("SELECT source FROM articles LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating DB: Adding source column")
        c.execute("ALTER TABLE articles ADD COLUMN source TEXT")
        
    # Create Feeds table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            active BOOLEAN DEFAULT 1,
            last_fetched DATETIME
        )
    ''')
    
    # Check if feeds empty, seed if so
    c.execute("SELECT count(*) FROM feeds")
    if c.fetchone()[0] == 0:
        seed_feeds(c)
    else:
        # Cleanup: Remove USOM feeds if they exist from previous sessions
        c.execute("DELETE FROM feeds WHERE url LIKE '%usom.gov.tr%'")
        
    conn.commit()
    conn.close()

def seed_feeds(cursor):
    initial_feeds = [
        ("BleepingComputer", "https://www.bleepingcomputer.com/feed/"),
        ("Darknet Diaries", "https://podcast.darknetdiaries.com/"),
        ("Graham Cluley", "https://grahamcluley.com/feed/"),
        ("Krebs on Security", "https://krebsonsecurity.com/feed/"),
        ("SANS Internet Storm Center", "https://isc.sans.edu/rssfeed_full.xml"),
        ("Schneier on Security", "https://www.schneier.com/feed/atom/"),
        ("Securelist", "https://securelist.com/feed/"),
        ("Sophos Security Operations", "https://news.sophos.com/en-us/category/security-operations/feed/"),
        ("The Hacker News", "https://feeds.feedburner.com/TheHackersNews?format=xml"),
        ("Sophos Threat Research", "https://news.sophos.com/en-us/category/threat-research/feed/"),
        ("Troy Hunt", "https://www.troyhunt.com/rss/"),
        ("WeLiveSecurity", "https://feeds.feedburner.com/eset/blog")
    ]
    logger.info("Seeding initial security feeds...")
    for name, url in initial_feeds:
        cursor.execute("INSERT INTO feeds (name, url) VALUES (?, ?)", (name, url))

def initialize_ai():
    global llm, embeddings, vectorstore
    try:
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # Initialize ChromaDB
        chroma_host = os.environ.get('CHROMA_HOST', 'chromadb')
        chroma_port = os.environ.get('CHROMA_PORT', '8000')
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        
        vectorstore = Chroma(
            client=client,
            collection_name="intel_articles",
            embedding_function=embeddings
        )
        
        logger.info(f"AI: Initialized Ollama with model {LLM_MODEL} and ChromaDB RAG")
    except Exception as e:
        logger.error(f"AI: Failed to initialize AI components: {e}")

def scrape_article_content(url):
    """Scrapes the main article body and images from BleepingComputer."""
    try:
        logger.info(f"Scraping content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # BleepingComputer specific selector
        article_body = soup.find('div', class_='articleBody')
        
        # The Hacker News specific selector
        if not article_body:
            article_body = soup.find('div', id='articlebody')
            
        if not article_body:
            article_body = soup.find('article') or soup.find('main')
            
        text_content = ""
        media_urls = []
        
        if article_body:
            # Extract images BEFORE decomposing noise
            images = article_body.find_all('img')
            for img in images:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http'):
                    media_urls.append(src)
                    
            # Remove scripts, styles, etc.
            for script in article_body(["script", "style", "aside", "nav", "noscript", "iframe"]):
                script.decompose()
            
            # Get text
            text = article_body.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text_content = '\n'.join(lines)
            
        # Fallback to meta description if content is empty or very short
        if not text_content or len(text_content) < 200:
            logger.info("Content short/empty, falling back to meta description")
            meta_desc = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                desc_text = meta_desc.get('content', '').strip()
                if desc_text:
                    text_content = f"SUMMARY (Content behind JS/Login): {desc_text}\n\n" + text_content
            
        return text_content, json.dumps(media_urls)
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return "", "[]"

def fetch_single_feed(feed_url, feed_name):
    try:
        logger.info(f"Fetching RSS feed from {feed_url}")
        feed = feedparser.parse(feed_url)
        conn = get_db_connection()
        c = conn.cursor()
        
        new_count = 0
        for entry in feed.entries:
            # Generate a unique stable ID (MD5 hash of link)
            article_id = hashlib.md5(entry.link.encode()).hexdigest()
            
            # Check if exists
            c.execute("SELECT published FROM articles WHERE id = ?", (article_id,))
            existing = c.fetchone()
            
            # Simple deduplication: skip if ID exists and published date matches
            # If date is different, it might be an update (though RSS links usually stable)
            published = entry.published if 'published' in entry else datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
            
            if existing:
                if existing['published'] == published:
                    continue # Identical, skip
                else:
                    logger.info(f"Article update detected for {article_id}: {entry.title}")
                    # Remove old one to re-ingest (or update)
                    c.execute("DELETE FROM articles WHERE id = ?", (article_id,))
                    # Also remove from Chroma if exists (simplified update)
                    if vectorstore:
                        try: vectorstore.delete([article_id])
                        except: pass
                
            # Clean RSS summary
            rss_summary = BeautifulSoup(entry.summary, 'lxml').get_text(strip=True) if 'summary' in entry else ""
            
            # --- FAST INGESTION ---
            # We skip AI analysis here to make fetching fast.
            # We scrape content because we need it for analysis later.
            title = entry.title
            full_content, media_json = scrape_article_content(entry.link)
            
            c.execute('''
                INSERT INTO articles (id, title, link, published, summary, content, media, source, analyzed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ''', (article_id, title, entry.link, published, rss_summary, full_content, media_json, feed_name))
            new_count += 1
            
        conn.commit()
        conn.close()
        logger.info(f"Fetched {new_count} new articles from {feed_name}")
        return new_count
    except Exception as e:
        logger.error(f"Error fetching feed {feed_url}: {e}")
        return 0

def fetch_all_feeds():
    logger.info("Starting scheduled feed fetch...")
    conn = get_db_connection()
    feeds = conn.execute("SELECT * FROM feeds WHERE active = 1").fetchall()
    conn.close()
    
    total_new = 0
    for feed in feeds:
        total_new += fetch_single_feed(feed['url'], feed['name'])
        
    # Trigger analysis batch if we found new stuff
    if total_new > 0:
        logger.info("Triggering AI analysis for new articles...")
        analyze_pending_articles() # Process immediate batch
        
    return total_new

# Background Job: Analyze Pending Articles
def analyze_pending_articles():
    """Processes unanalyzed articles in the background, staggered."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Process batch of 2 to avoid overloading the AI service
    c.execute("SELECT id, title, content, link, media FROM articles WHERE analyzed = 0 ORDER BY RANDOM() LIMIT 2")
    articles = c.fetchall()
    
    if not articles:
        conn.close()
        return

    # Initialize Client with correct host
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
    except Exception as e:
        logger.error(f"Failed to init Ollama Client: {e}")
        conn.close()
        return

    for article in articles:
        logger.info(f"Background Analysis: Processing {article['title']}")
        
        article_id = article['id']
        full_content = article['content']
        media_json = article['media']
        
        # --- AI Analysis Logic ---
        analysis = "Analysis failed."
        risk_level = "Unknown"
        category = "Unclassified"
        
        try:
            # Prepare Context
            images = json.loads(media_json) if media_json else []
            analysis_text = full_content[:6000] if full_content else "No content."
            
            # Format Prompt
            prompt = f"""You are a Cyber Threat Intelligence Analyst. 
            Analyze this article and return a STRICT JSON object with these keys:
            - summary: 2-sentence executive summary
            - risk_level: (Critical, High, Medium, Low, Info)
            - category: (Ransomware, Phishing, Vulnerability, Malware, Data Breach, Policy, Other)
            
            Article:
            {analysis_text} 
            """
            
            # Call Multimodal LLM (limit to 1 image)
            images_b64 = []
            if images:
                try:
                    import base64
                    # Download first image
                    img_resp = requests.get(images[0], timeout=5)
                    if img_resp.status_code == 200:
                        # Encode to base64
                        b64_str = base64.b64encode(img_resp.content).decode('utf-8')
                        images_b64.append(b64_str)
                except Exception as e:
                    logger.warning(f"Failed to process image for {article_id}: {e}")

            if images_b64:
                 response = client.chat(model=VISION_MODEL, messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [images_b64[0]] 
                }], format='json')
            else:
                 response = client.chat(model=VISION_MODEL, messages=[{
                    'role': 'user',
                    'content': prompt
                }], format='json')
                
            result = json.loads(response['message']['content'])
            analysis = result.get('summary', 'No summary provided.')
            risk_level = result.get('risk_level', 'Info')
            category = result.get('category', 'Other')
            
            # Embed and Store in Vector DB
            if vectorstore:
                vectorstore.add_texts(
                    texts=[f"Title: {article['title']}\nSummary: {analysis}\nContent: {full_content}"],
                    metadatas=[{"source": article['link'], "risk": risk_level, "id": article_id, "images": str(images)}],
                    ids=[article_id]
                )
                
        except Exception as e:
            logger.error(f"AI Analysis failed for {article_id}: {e}")
            analysis = "AI analysis currently unavailable."

        # Update DB
        c.execute('''
            UPDATE articles 
            SET ai_summary = ?, ai_risk_level = ?, ai_category = ?, analyzed = 1 
            WHERE id = ?
        ''', (analysis, risk_level, category, article_id))
        conn.commit()
    
    conn.close()
    logger.info(f"Completed analysis batch of {len(articles)}")

# Scheduler
scheduler = BackgroundScheduler()
# Fetch feeds every 30 minutes
scheduler.add_job(func=fetch_all_feeds, trigger="interval", minutes=30)
# Process analysis queue every 10 seconds (staggered)
scheduler.add_job(func=analyze_pending_articles, trigger="interval", seconds=10)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

def analyze_article_ai(title, summary, full_content):
    if not llm:
        initialize_ai()
    if not llm:
        return "Unknown", "AI Service Unavailable", "Unknown"

    # Perfer full content, but truncate to avoid context limit if necessary
    analysis_text = full_content if full_content and len(full_content) > 100 else summary
    # Limit to ~2000 words/tokens roughly to be safe with smaller models
    analysis_text = analysis_text[:8000]

    try:
        prompt = ChatPromptTemplate.from_template(
            "Analyze the following cybersecurity news article.\n"
            "Title: {title}\n"
            "Content: {content}\n\n"
            "Provide the output in valid JSON format with the following keys:\n"
            "- category: (e.g., Ransomware, Phishing, Vulnerability, Data Breach, General)\n"
            "- risk_level: (Critical, High, Medium, Low, Info)\n"
            "- analysis: (A one sentence executive summary of the threat based on the full details)\n"
            "Do not include markdown formatting like ```json."
        )
        chain = prompt | llm | StrOutputParser()
        result_str = chain.invoke({"title": title, "content": analysis_text})
        
        # basic cleanup if model adds markdown
        result_str = result_str.replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(result_str)
            return result.get('category', 'General'), result.get('analysis', 'Analysis failed'), result.get('risk_level', 'Medium')
        except json.JSONDecodeError:
            logger.error("Failed to parse AI response as JSON")
            return "General", result_str[:100], "Medium"

    except Exception as e:
        logger.error(f"AI Analysis failed: {e}")
        return "Error", f"Error: {str(e)}", "Unknown"

    
    conn.close()
    return count

# Routes

@app.route('/')
def index():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Group by Category
    c.execute("SELECT DISTINCT ai_category FROM articles WHERE analyzed = 1")
    categories = [row[0] for row in c.fetchall() if row[0]]
    
    articles_by_cat = {}
    for cat in categories:
        c.execute("SELECT * FROM articles WHERE ai_category = ? ORDER BY published DESC LIMIT 5", (cat,))
        articles_by_cat[cat] = [dict(row) for row in c.fetchall()]
        
    # Also get "Latest"
    c.execute("SELECT * FROM articles ORDER BY published DESC LIMIT 10")
    latest = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return render_template('index.html', categories=articles_by_cat, latest=latest)

@app.route('/refresh')
def refresh():
    # Sync call to fetch all feeds
    try:
        new = fetch_all_feeds()
        return jsonify({"message": f"Fetched {new} new articles from all enabled feeds."})
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return jsonify({"error": str(e)}), 500

# Feed Management API
@app.route('/api/feeds', methods=['GET', 'POST'])
def handle_feeds():
    conn = get_db_connection()
    if request.method == 'GET':
        feeds = conn.execute("SELECT * FROM feeds").fetchall()
        conn.close()
        return jsonify([dict(f) for f in feeds])
        
    if request.method == 'POST':
        data = request.json
        name = data.get('name')
        url = data.get('url')
        if not name or not url:
            return jsonify({"error": "Name and URL required"}), 400
        
        try:
            conn.execute("INSERT INTO feeds (name, url) VALUES (?, ?)", (name, url))
            conn.commit()
            conn.close()
            return jsonify({"success": True}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/feeds/<int:feed_id>', methods=['DELETE', 'PUT'])
def feed_ops(feed_id):
    conn = get_db_connection()
    if request.method == 'DELETE':
        conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    
    if request.method == 'PUT':
        # Toggle active state or update
        data = request.json
        if 'active' in data:
            conn.execute("UPDATE feeds SET active = ? WHERE id = ?", (data['active'], feed_id))
            conn.commit()
            conn.close()
            return jsonify({"success": True})
        return jsonify({"error": "Invalid update data"}), 400

@app.route('/api/articles/<article_id>/tags', methods=['POST', 'DELETE'])
def manage_tags(article_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT user_tags FROM articles WHERE id = ?", (article_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Article not found"}), 404
    
    current_tags = json.loads(row['user_tags']) if row['user_tags'] else []
    
    data = request.json
    tag = data.get('tag')
    
    if request.method == 'POST':
        if tag and tag not in current_tags:
            current_tags.append(tag)
    
    if request.method == 'DELETE':
        if tag and tag in current_tags:
            current_tags.remove(tag)
            
    c.execute("UPDATE articles SET user_tags = ? WHERE id = ?", (json.dumps(current_tags), article_id))
    conn.commit()
    conn.close()
    return jsonify({"tags": current_tags})

@app.route('/details/<article_id>')
def details(article_id):
    conn = get_db_connection()
    article = conn.execute('SELECT * FROM articles WHERE id = ?', (article_id,)).fetchone()
    conn.close()
    if not article:
        return "Article not found", 404
    return render_template('details.html', article=dict(article))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not llm or not vectorstore:
        initialize_ai()
        
    context_str = ""
    retrieved_media = []
    
    # Process Frontend Context
    frontend_context = data.get('context', {})
    page_awareness = ""
    if frontend_context:
        page_type = frontend_context.get('pageType')
        if page_type == 'article_details':
            art = frontend_context.get('currentArticle', {})
            page_awareness = f"\n[USER IS CURRENTLY VIEWING THIS ARTICLE]:\nTitle: {art.get('title')}\nRisk: {art.get('risk')}\nSummary: {art.get('summary')}\nLink: {art.get('sourceLink')}\n"
        elif page_type in ['dashboard', 'listing']:
            arts = frontend_context.get('articles', [])
            titles = ", ".join([a.get('title') for a in arts[:10]])
            page_awareness = f"\n[USER IS VIEWING A LIST OF ARTICLES INCLUDING]: {titles}\n"

    if vectorstore:
        try:
            # Perform similarity search
            docs = vectorstore.similarity_search(query.lower(), k=3)
            context_pieces = []
            for d in docs:
                source_url = d.metadata.get('source', 'Unknown')
                context_pieces.append(f"--- ARTICLE: {d.metadata.get('title')} ---\nURL: {source_url}\n{d.page_content}")
                # Retrieve media for this article from DB
                aid = d.metadata.get('id')
                if aid:
                    conn_m = get_db_connection()
                    res = conn_m.execute("SELECT media FROM articles WHERE id = ?", (aid,)).fetchone()
                    if res and res['media']:
                        urls = json.loads(res['media'])
                        retrieved_media.extend(urls[:1]) 
                    conn_m.close()
                    
            context_str = "\n\n".join(context_pieces)
            logger.info(f"Chat: Retrieved {len(docs)} articles and {len(retrieved_media)} images for RAG")
        except Exception as e:
            logger.error(f"Chat: RAG search failed: {e}")
            
    # Fallback/Default context
    if not context_str:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT title, ai_summary FROM articles WHERE analyzed = 1 ORDER BY published DESC LIMIT 5")
        articles = c.fetchall()
        conn.close()
        context_str = "\n".join([f"- {a['title']}: {a['ai_summary']}" for a in articles])
    
    # Use Multimodal Model if images are available
    model_to_use = VISION_MODEL if retrieved_media else LLM_MODEL
    llm_vision = Ollama(model=model_to_use, base_url=OLLAMA_BASE_URL)
    
    try:
        prompt_text = (
            f"You are a Senior Cyber Threat Intelligence Analyst. You are professional, precise, and data-driven.\n"
            f"Current Page Context: {page_awareness}\n\n"
            f"Knowledge Base Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Acknowledge the user's current view if relevant.\n"
            f"2. ANALYTICAL RIGOR: For specific technical or threat-related queries, base all responses strictly on the provided 'Knowledge Base Context'. If the data contains statistics (e.g., volume, frequency, percentages), lead with these facts. For general conversation or greetings, respond naturally in your professional persona.\n"
            f"3. STRUCTURE: Use Markdown headers (##) for 'Executive Summary', 'Statistical Analysis', and 'Recommendations' when providing intelligence reports.\n"
            f"4. CITATION: Use bullet points for facts. Every claim must be supported by a data point from the context. Provide clickable Markdown links [Title](URL) for all referenced sources.\n"
            f"5. NO HALLUCINATION: If a specific statistic or data point is missing, do not estimate. State that the data is unavailable.\n"
            f"6. FORMATTING: Do NOT defang URLs from reputable news sources.\n"
            f"7. IOC EXTRACTION: If IOCs (IPs, hashes, domains) are present, append a JSON block:\n"
            f"```json\n{{\"iocs\": [\"ioc1\", \"ioc2\"]}}\n```\n"
            f"Assistant:"
        )
        
        if retrieved_media:
            # Try to fetch and encode images for the vision model
            import base64
            images_b64 = []
            for m_url in retrieved_media[:1]: # Limit to 1 image as model only supports one
                try:
                    img_resp = requests.get(m_url, timeout=5)
                    if img_resp.status_code == 200:
                        images_b64.append(base64.b64encode(img_resp.content).decode('utf-8'))
                except:
                    continue
            
            if images_b64:
                response = llm_vision.invoke(prompt_text, images=images_b64)
            else:
                response = llm_vision.invoke(prompt_text)
        else:
            response = llm_vision.invoke(prompt_text)
            
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return jsonify({"error": "AI processing timeout or error. Please try again."}), 500

# Init
try:
    init_db()
except Exception as e:
    logger.error(f"DB Init failed: {e}")

@app.route('/api/articles/<article_id>/rescrape', methods=['POST'])
def rescrape_article(article_id):
    conn = get_db_connection()
    c = conn.cursor()
    row = c.execute("SELECT link FROM articles WHERE id = ?", (article_id,)).fetchone()
    
    if not row:
        conn.close()
        return jsonify({"error": "Article not found"}), 404
        
    link = row['link']
    logger.info(f"Re-scraping article {article_id} from {link}")
    
    try:
        # Re-use scrape_article_content
        full_content, media_json = scrape_article_content(link)
        
        # Update DB and reset analysis
        c.execute("""
            UPDATE articles 
            SET content = ?, media = ?, analyzed = 0, ai_summary = 'Pending Re-Analysis...', ai_risk_level = 'Unknown' 
            WHERE id = ?
        """, (full_content, media_json, article_id))
        
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        conn.close()
        logger.error(f"Re-scrape failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/articles/<article_id>/analyze', methods=['POST'])
def reanalyze_article(article_id):
    conn = get_db_connection()
    c = conn.cursor()
    # Reset analyzed status to 0 so the scheduler picks it up
    c.execute("UPDATE articles SET analyzed = 0, ai_summary = 'Pending Re-Analysis...', ai_risk_level = 'Unknown' WHERE id = ?", (article_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/api/status/queue')
def queue_status():
    conn = get_db_connection()
    c = conn.cursor()
    # Count pending articles
    pending_count = c.execute("SELECT COUNT(*) FROM articles WHERE analyzed = 0").fetchone()[0]
    conn.close()
    
    # Estimate: assume 20s per article
    est_seconds = pending_count * 20
    if est_seconds > 60:
        est_str = f"{est_seconds // 60} mins"
    else:
        est_str = f"{est_seconds} sec"
        
    return jsonify({"count": pending_count, "estimate": est_str})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/story-feed')
def story_feed():
    conn = get_db_connection()
    c = conn.cursor()
    # Fetch all analyzed articles for the feed
    c.execute("SELECT * FROM articles WHERE analyzed = 1 ORDER BY published DESC LIMIT 200")
    articles = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('story_feed.html', articles=articles)

@app.route('/visuals')
def visuals_page():
    return render_template('visuals.html')

@app.route('/api/generate_chart', methods=['POST'])
def generate_chart():
    data = request.json
    query = data.get('query', 'Bar chart of categories')
    
    conn = get_db_connection()
    c = conn.cursor()
    # Get a broad set of data for the AI to analyze
    c.execute("SELECT ai_category, ai_risk_level, source, published, ai_summary FROM articles WHERE analyzed = 1 LIMIT 200")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    
    if not rows:
        return jsonify({"error": "Not enough data"}), 400

    # Simplify data for context window
    simple_data = []
    for r in rows:
        simple_data.append({
            "category": r['ai_category'],
            "risk": r['ai_risk_level'],
            "source": r['source']
        })
    data_summary = json.dumps(simple_data)
    
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        prompt = f"""
        You are a Data Logic Expert for ApexCharts.
        User Request: "{query}"
        
        Dataset (JSON):
        {data_summary}
        
        Task:
        1. Analyze the dataset to calculate the counts/metrics requested.
        2. Decide the best ApexCharts type: 'bar', 'pie', 'donut', 'line', 'area', 'heatmap'.
        3. Construct the JSON configuration for ApexCharts options.
        
        Rules:
        - For 'pie'/'donut': "labels" are categories, "series" is array of numbers.
        - For 'bar'/'line': "xaxis": {{ "categories": [...] }}, "series": [ {{ "name": "Metric", "data": [...] }} ]
        - For 'heatmap': "series": [ {{ "name": "Y-Label", "data": [ {{ "x": "X-Label", "y": value }} ] }} ]
        
        Return JSON ONLY:
        {{
            "message": "Brief comment on what is shown",
            "chart_options": {{ ...valid ApexCharts options object... }}
        }}
        """
        
        response = client.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], format='json')
        result = json.loads(response['message']['content'])
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chart Gen Failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
