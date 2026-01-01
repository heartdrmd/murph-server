"""
Murph Server - The Cloud Brain
Flask API with Pinecone vectors, PostgreSQL structured data, Claude reasoning, Gemini big-picture queries
"""

import os
import json
import hashlib
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import anthropic
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///murph_dev.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Handle Render's postgres:// vs postgresql:// issue
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

db = SQLAlchemy(app)

# Initialize AI clients
anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'murph-brain')
PINECONE_HOST = os.environ.get('PINECONE_HOST', 'https://murph-brain-or2vyhb.svc.aped-4627-b74a.pinecone.io')

def get_pinecone_index():
    """Get Pinecone index with integrated embeddings."""
    try:
        return pc.Index(host=PINECONE_HOST)
    except Exception as e:
        app.logger.error(f"Pinecone error: {e}")
        return None


# ============ Database Models ============

class Feature(db.Model):
    """Available features/capabilities Murph can do."""
    __tablename__ = 'features'
    
    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    enabled = db.Column(db.Boolean, default=True)
    version = db.Column(db.String(20), default='1.0')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    methods = db.relationship('FeatureMethod', backref='feature', lazy=True, cascade='all, delete-orphan')


class FeatureMethod(db.Model):
    """Specific methods within a feature."""
    __tablename__ = 'feature_methods'
    
    id = db.Column(db.Integer, primary_key=True)
    feature_id = db.Column(db.String(50), db.ForeignKey('features.id'), nullable=False)
    method_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    code_template = db.Column(db.Text)
    code_type = db.Column(db.String(20))  # applescript, shell, shortcut
    parameters = db.Column(db.JSON)  # list of parameter names
    examples = db.Column(db.JSON)  # example phrases that trigger this
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ApprovedRecipe(db.Model):
    """User-approved recipes synced from local."""
    __tablename__ = 'approved_recipes'
    
    id = db.Column(db.Integer, primary_key=True)
    intent_hash = db.Column(db.String(64), unique=True)  # hash of normalized intent
    intent_examples = db.Column(db.JSON)  # list of phrases that triggered this
    code = db.Column(db.Text, nullable=False)
    code_type = db.Column(db.String(20), nullable=False)
    feature_id = db.Column(db.String(50), db.ForeignKey('features.id'))
    times_used = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime, default=datetime.utcnow)


class UserSettings(db.Model):
    """User preferences."""
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), unique=True, nullable=False)
    chattiness = db.Column(db.Integer, default=50)  # 1-100
    voice = db.Column(db.String(50), default='prof_josh')
    preferences = db.Column(db.JSON)  # arbitrary prefs
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeatureChangelog(db.Model):
    """Version history for features."""
    __tablename__ = 'feature_changelog'
    
    id = db.Column(db.Integer, primary_key=True)
    feature_id = db.Column(db.String(50), db.ForeignKey('features.id'), nullable=False)
    version = db.Column(db.String(20), nullable=False)
    changes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============ Embedding & Vector Functions ============
# Using Pinecone integrated embeddings (llama-text-embed-v2)
# No need to call OpenAI - just send text directly

def store_in_pinecone(id, text, metadata):
    """Store text in Pinecone with integrated embeddings."""
    index = get_pinecone_index()
    if not index:
        return False
    
    # Add the text field for integrated embedding
    metadata['text'] = text
    
    index.upsert_records(
        namespace="",
        records=[{
            '_id': id,
            'text': text,  # Pinecone embeds this automatically
            **metadata
        }]
    )
    return True


def search_pinecone(query, top_k=5, filter_dict=None):
    """Semantic search in Pinecone with integrated embeddings."""
    index = get_pinecone_index()
    if not index:
        return []
    
    # Search using text - Pinecone embeds query automatically
    results = index.search(
        namespace="",
        query={
            "top_k": top_k,
            "inputs": {"text": query},
            "filter": filter_dict
        },
        include_metadata=True
    )
    return results.get('matches', [])


# ============ AI Functions ============

def ask_claude(prompt, system_prompt=None, context=None):
    """Ask Claude for reasoning/code generation."""
    messages = []
    
    if context:
        messages.append({"role": "user", "content": f"Context from knowledge base:\n{context}\n\n---\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})
    
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": messages
    }
    
    if system_prompt:
        kwargs["system"] = system_prompt
    
    response = anthropic_client.messages.create(**kwargs)
    return response.content[0].text


def ask_gemini(prompt, include_all_knowledge=False):
    """Ask Gemini, optionally with entire knowledge base."""
    if include_all_knowledge:
        # Gather all knowledge
        features = Feature.query.all()
        recipes = ApprovedRecipe.query.all()
        
        knowledge = "# Murph's Complete Knowledge Base\n\n"
        knowledge += "## Features\n"
        for f in features:
            knowledge += f"- {f.name}: {f.description}\n"
            for m in f.methods:
                knowledge += f"  - {m.method_name}: {m.description}\n"
        
        knowledge += "\n## Approved Recipes\n"
        for r in recipes:
            knowledge += f"- Intent: {r.intent_examples}\n  Code: {r.code[:200]}...\n"
        
        full_prompt = f"{knowledge}\n\n---\n\nQuestion: {prompt}"
    else:
        full_prompt = prompt
    
    response = gemini_model.generate_content(full_prompt)
    return response.text


# ============ Core System Prompt ============

MURPH_SYSTEM_PROMPT = """You are Murph, a Mac assistant that can control the user's computer through AppleScript and shell commands.

Your personality:
- Helpful and capable
- Adjust verbosity based on chattiness level (1=terse, 100=chatty)
- You learn from mistakes and remember what works

When asked to do something:
1. Check if there's a known recipe/method for this
2. If yes, use it (possibly adapted)
3. If no, figure out how to do it - write AppleScript or shell commands
4. Always return structured JSON with your plan

Response format (JSON):
{
    "understanding": "what you think user wants",
    "approach": "how you'll do it",
    "code": "the actual code to execute",
    "code_type": "applescript|shell|shortcut",
    "spoken_response": "what to say back (adjust length to chattiness)",
    "confidence": 0.0-1.0
}

Available features and methods will be provided in context.
"""


# ============ API Routes ============

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({"status": "ok", "service": "murph-server"})


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint - receives user input, returns action plan."""
    data = request.json
    user_input = data.get('input', '')
    chattiness = data.get('chattiness', 50)
    user_id = data.get('user_id', 'default')
    
    # Search for relevant knowledge
    relevant = search_pinecone(user_input, top_k=5)
    
    # Build context from search results
    context_parts = []
    for match in relevant:
        meta = match.metadata
        context_parts.append(f"[{meta.get('type', 'unknown')}] {meta.get('content', '')}")
    
    context = "\n".join(context_parts) if context_parts else None
    
    # Get features for context
    features = Feature.query.filter_by(enabled=True).all()
    features_context = "\n".join([
        f"Feature: {f.name} ({f.id})\nMethods: {', '.join([m.method_name for m in f.methods])}"
        for f in features
    ])
    
    # Build prompt
    prompt = f"""Chattiness level: {chattiness}/100

Available features:
{features_context}

User request: {user_input}

Respond with JSON only."""
    
    # Ask Claude
    response = ask_claude(prompt, system_prompt=MURPH_SYSTEM_PROMPT, context=context)
    
    # Try to parse JSON from response
    try:
        # Handle potential markdown code blocks
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        result = json.loads(response.strip())
    except json.JSONDecodeError:
        result = {
            "understanding": user_input,
            "approach": "direct response",
            "code": None,
            "code_type": None,
            "spoken_response": response,
            "confidence": 0.5
        }
    
    return jsonify(result)


@app.route('/report_result', methods=['POST'])
def report_result():
    """Mac app reports execution result - for learning."""
    data = request.json
    success = data.get('success', False)
    user_input = data.get('input', '')
    code = data.get('code', '')
    code_type = data.get('code_type', '')
    error = data.get('error', None)
    
    # If failed, we might want to store this for analysis
    if not success and error:
        # Could store failures for later analysis
        app.logger.warning(f"Execution failed: {error}")
    
    return jsonify({"status": "recorded"})


@app.route('/sync_recipe', methods=['POST'])
def sync_recipe():
    """Sync an approved recipe from local to cloud."""
    data = request.json
    intent_examples = data.get('intent_examples', [])
    code = data.get('code', '')
    code_type = data.get('code_type', '')
    feature_id = data.get('feature_id')
    
    # Create hash for deduplication
    intent_hash = hashlib.sha256(code.encode()).hexdigest()
    
    # Check if exists
    existing = ApprovedRecipe.query.filter_by(intent_hash=intent_hash).first()
    
    if existing:
        # Update usage
        existing.times_used += 1
        existing.last_used = datetime.utcnow()
        # Merge intent examples
        existing_examples = existing.intent_examples or []
        existing.intent_examples = list(set(existing_examples + intent_examples))
    else:
        # Create new
        recipe = ApprovedRecipe(
            intent_hash=intent_hash,
            intent_examples=intent_examples,
            code=code,
            code_type=code_type,
            feature_id=feature_id
        )
        db.session.add(recipe)
        
        # Store in Pinecone
        for example in intent_examples:
            store_in_pinecone(
                id=f"recipe_{intent_hash}_{hash(example)}",
                text=example,
                metadata={
                    'type': 'recipe',
                    'content': code[:500],
                    'code_type': code_type,
                    'feature_id': feature_id
                }
            )
    
    db.session.commit()
    return jsonify({"status": "synced"})


@app.route('/features', methods=['GET'])
def get_features():
    """Get all features for Mac app sync."""
    features = Feature.query.all()
    return jsonify([{
        'id': f.id,
        'name': f.name,
        'description': f.description,
        'enabled': f.enabled,
        'version': f.version,
        'methods': [{
            'method_name': m.method_name,
            'description': m.description,
            'code_template': m.code_template,
            'code_type': m.code_type,
            'parameters': m.parameters,
            'examples': m.examples
        } for m in f.methods]
    } for f in features])


@app.route('/features', methods=['POST'])
def create_feature():
    """Create a new feature."""
    data = request.json
    
    feature = Feature(
        id=data['id'],
        name=data['name'],
        description=data.get('description', ''),
        enabled=data.get('enabled', True),
        version=data.get('version', '1.0')
    )
    db.session.add(feature)
    
    # Add methods
    for method_data in data.get('methods', []):
        method = FeatureMethod(
            feature_id=feature.id,
            method_name=method_data['method_name'],
            description=method_data.get('description', ''),
            code_template=method_data.get('code_template', ''),
            code_type=method_data.get('code_type', 'applescript'),
            parameters=method_data.get('parameters', []),
            examples=method_data.get('examples', [])
        )
        db.session.add(method)
        
        # Store in Pinecone for semantic search
        for example in method_data.get('examples', []):
            store_in_pinecone(
                id=f"method_{feature.id}_{method.method_name}_{hash(example)}",
                text=example,
                metadata={
                    'type': 'method',
                    'feature_id': feature.id,
                    'method_name': method.method_name,
                    'content': method_data.get('description', '')
                }
            )
    
    db.session.commit()
    return jsonify({"status": "created", "id": feature.id})


@app.route('/settings/<user_id>', methods=['GET', 'PUT'])
def user_settings(user_id):
    """Get or update user settings."""
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    
    if request.method == 'GET':
        if not settings:
            return jsonify({
                'user_id': user_id,
                'chattiness': 50,
                'voice': 'prof_josh',
                'preferences': {}
            })
        return jsonify({
            'user_id': settings.user_id,
            'chattiness': settings.chattiness,
            'voice': settings.voice,
            'preferences': settings.preferences
        })
    
    else:  # PUT
        data = request.json
        if not settings:
            settings = UserSettings(user_id=user_id)
            db.session.add(settings)
        
        if 'chattiness' in data:
            settings.chattiness = data['chattiness']
        if 'voice' in data:
            settings.voice = data['voice']
        if 'preferences' in data:
            settings.preferences = data['preferences']
        
        db.session.commit()
        return jsonify({"status": "updated"})


@app.route('/ask_gemini', methods=['POST'])
def ask_gemini_endpoint():
    """Ask Gemini with optional full knowledge base."""
    data = request.json
    query = data.get('query', '')
    include_all = data.get('include_all_knowledge', False)
    
    response = ask_gemini(query, include_all_knowledge=include_all)
    return jsonify({"response": response})


@app.route('/search', methods=['POST'])
def search_knowledge():
    """Search the knowledge base."""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    filter_type = data.get('type')  # optional: 'recipe', 'method', etc
    
    filter_dict = {'type': filter_type} if filter_type else None
    results = search_pinecone(query, top_k=top_k, filter_dict=filter_dict)
    
    return jsonify([{
        'id': r.id,
        'score': r.score,
        'metadata': r.metadata
    } for r in results])


# ============ Initialize with base features ============

def init_base_features():
    """Initialize with Chrome and File Management features."""
    
    # Check if already initialized
    if Feature.query.first():
        return
    
    # Chrome feature
    chrome = Feature(
        id='chrome',
        name='Chrome Control',
        description='Control Google Chrome browser - tabs, windows, navigation'
    )
    db.session.add(chrome)
    
    chrome_methods = [
        FeatureMethod(
            feature_id='chrome',
            method_name='close_all_tabs_except_current',
            description='Close all Chrome tabs except the currently active one',
            code_template='''tell application "Google Chrome"
    set currentTab to active tab of front window
    set currentURL to URL of currentTab
    repeat with t in (tabs of front window)
        if URL of t is not currentURL then
            close t
        end if
    end repeat
end tell''',
            code_type='applescript',
            parameters=[],
            examples=['close all tabs except this one', 'close other tabs', 'keep only this tab']
        ),
        FeatureMethod(
            feature_id='chrome',
            method_name='open_url',
            description='Open a URL in Chrome',
            code_template='''tell application "Google Chrome"
    open location "{url}"
    activate
end tell''',
            code_type='applescript',
            parameters=['url'],
            examples=['open google.com', 'go to {url}', 'open {url} in chrome']
        ),
        FeatureMethod(
            feature_id='chrome',
            method_name='get_current_url',
            description='Get the URL of the current tab',
            code_template='''tell application "Google Chrome"
    return URL of active tab of front window
end tell''',
            code_type='applescript',
            parameters=[],
            examples=['what page am I on', 'current url', 'what site is this']
        ),
        FeatureMethod(
            feature_id='chrome',
            method_name='list_tabs',
            description='List all open tabs',
            code_template='''tell application "Google Chrome"
    set tabList to ""
    repeat with w in windows
        repeat with t in tabs of w
            set tabList to tabList & title of t & " | " & URL of t & "\n"
        end repeat
    end repeat
    return tabList
end tell''',
            code_type='applescript',
            parameters=[],
            examples=['list my tabs', 'what tabs are open', 'show all tabs']
        ),
        FeatureMethod(
            feature_id='chrome',
            method_name='close_tab_by_title',
            description='Close a tab by its title',
            code_template='''tell application "Google Chrome"
    repeat with w in windows
        repeat with t in tabs of w
            if title of t contains "{search}" then
                close t
                return "Closed tab: " & title of t
            end if
        end repeat
    end repeat
    return "No tab found matching: {search}"
end tell''',
            code_type='applescript',
            parameters=['search'],
            examples=['close the youtube tab', 'close tab with {search}', 'close {search} tab']
        )
    ]
    
    for method in chrome_methods:
        db.session.add(method)
    
    # File Management feature
    files = Feature(
        id='files',
        name='File Management',
        description='Manage files and folders in Finder'
    )
    db.session.add(files)
    
    file_methods = [
        FeatureMethod(
            feature_id='files',
            method_name='open_folder',
            description='Open a folder in Finder',
            code_template='''tell application "Finder"
    open folder "{path}"
    activate
end tell''',
            code_type='applescript',
            parameters=['path'],
            examples=['open downloads folder', 'open {folder}', 'show me {folder}']
        ),
        FeatureMethod(
            feature_id='files',
            method_name='list_files',
            description='List files in a folder',
            code_template='ls -la "{path}"',
            code_type='shell',
            parameters=['path'],
            examples=['what files are in {folder}', 'list files in {path}', 'show contents of {folder}']
        ),
        FeatureMethod(
            feature_id='files',
            method_name='create_folder',
            description='Create a new folder',
            code_template='mkdir -p "{path}"',
            code_type='shell',
            parameters=['path'],
            examples=['create folder {name}', 'make directory {path}', 'new folder {name}']
        ),
        FeatureMethod(
            feature_id='files',
            method_name='move_file',
            description='Move a file to another location',
            code_template='mv "{source}" "{destination}"',
            code_type='shell',
            parameters=['source', 'destination'],
            examples=['move {file} to {folder}', 'put {file} in {folder}']
        ),
        FeatureMethod(
            feature_id='files',
            method_name='find_file',
            description='Find files by name',
            code_template='find ~ -name "*{search}*" -type f 2>/dev/null | head -20',
            code_type='shell',
            parameters=['search'],
            examples=['find {filename}', 'where is {file}', 'search for {name}']
        ),
        FeatureMethod(
            feature_id='files',
            method_name='open_file',
            description='Open a file with default application',
            code_template='open "{path}"',
            code_type='shell',
            parameters=['path'],
            examples=['open {file}', 'launch {file}']
        )
    ]
    
    for method in file_methods:
        db.session.add(method)
    
    db.session.commit()
    
    # Store in Pinecone
    for feature in [chrome, files]:
        for method in feature.methods:
            for example in (method.examples or []):
                try:
                    store_in_pinecone(
                        id=f"method_{feature.id}_{method.method_name}_{hash(example)}",
                        text=example,
                        metadata={
                            'type': 'method',
                            'feature_id': feature.id,
                            'method_name': method.method_name,
                            'content': method.description
                        }
                    )
                except Exception as e:
                    app.logger.warning(f"Could not store in Pinecone: {e}")


# ============ Entry Point ============

with app.app_context():
    db.create_all()
    init_base_features()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
