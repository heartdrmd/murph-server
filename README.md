# Murph Server

The cloud brain for Murph - your Mac AI assistant.

## Architecture

- **Pinecone**: Semantic vector search across all knowledge
- **PostgreSQL**: Structured data (features, methods, settings)
- **Claude**: Reasoning and code generation
- **Gemini 1.5 Pro**: Big-picture queries across entire knowledge base
- **OpenAI**: Embeddings for vector search

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy env file and add your keys
cp .env.example .env

# Run server
python app.py
```

## Pinecone Setup

1. Go to [pinecone.io](https://pinecone.io) and create account
2. Create index named `murph-brain`:
   - Dimensions: 3072 (for text-embedding-3-large)
   - Metric: cosine
3. Copy API key to .env

## Deploy to Render

1. Push to GitHub
2. In Render, create new Blueprint
3. Connect repo and select `render.yaml`
4. Add environment variables in Render dashboard
5. Deploy

## API Endpoints

### Core
- `POST /chat` - Main conversation endpoint
- `POST /report_result` - Report execution success/failure
- `POST /sync_recipe` - Sync approved recipe from local

### Features
- `GET /features` - Get all features for Mac app sync
- `POST /features` - Create new feature

### Settings
- `GET /settings/<user_id>` - Get user settings
- `PUT /settings/<user_id>` - Update user settings

### Search & AI
- `POST /search` - Search knowledge base
- `POST /ask_gemini` - Query Gemini with optional full knowledge base

## Built-in Features

### Chrome Control
- Close all tabs except current
- Open URL
- Get current URL
- List all tabs
- Close tab by title

### File Management
- Open folder
- List files
- Create folder
- Move file
- Find file
- Open file
