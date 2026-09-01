# OpenRouter setup

This project uses [OpenRouter](https://openrouter.ai) instead of a single vendor's API, so you can pick
any underlying model (OpenAI, Anthropic, Google, open-weight models, etc.) through one key.

## 1. Create an API key

1. Sign up / log in at https://openrouter.ai
2. Go to https://openrouter.ai/keys and create a new key (starts with `sk-or-...`)
3. Add a small amount of credit, or use free-tier models if your account qualifies

## 2. Configure the app

Copy `.env.example` to `.env` and set:

```
OPENROUTER_API_KEY=sk-or-your-real-key-here
```

Optional overrides:

```
# Pin a specific model instead of the built-in fallback list
OPENROUTER_MODEL=openai/gpt-4o-mini

# Sent as attribution headers to OpenRouter (not required, but recommended)
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=resume-rag-assistant
```

## 3. Run

```
streamlit run app.py
```

If no key is configured, the app still works using a local, regex/keyword-based fallback for
common resume questions (CGPA, skills, graduation year, etc.) — answers will just be less
flexible and the summary/strengths/questions sections will use generic heuristics instead of
being grounded in the actual resume content.

## Model fallback order

`rag_pipeline.py` tries models in this order when `OPENROUTER_MODEL` isn't set:

1. `openai/gpt-4o-mini`
2. `anthropic/claude-3.5-haiku`
3. `google/gemini-2.0-flash-001`
4. `meta-llama/llama-3.1-8b-instruct`

If one model is unavailable or returns an error, it automatically tries the next.