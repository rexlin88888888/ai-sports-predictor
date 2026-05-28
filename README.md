# AI Sports Predictor

Streamlit app for NBA and Football World Cup predictions. The project is organized as one deployable app with shared core modules and sport-specific model modules.

## Project Structure

```text
ai_sports_predictor/
|-- app.py
|-- requirements.txt
|-- README.md
|-- Dockerfile
|-- .streamlit/
|-- core/
|-- sports/
|-- models/
|-- utils/
|-- ui/
|-- data/
|-- outputs/
`-- cache/
```

## Install

```powershell
py -m pip install -r requirements.txt
```

Optional environment variables can be copied from `.env.example`:

```powershell
Copy-Item .env.example .env
```

The app works without API keys by using local cache and demo fallback data.

## Run

From inside this folder:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Streamlit Cloud

Use this folder as the GitHub project root and set the main file path to:

```text
app.py
```

API keys are optional. If no secrets are configured, the app uses fallback data.

## Docker

```powershell
docker build -t ai-sports-predictor .
docker run --rm -p 8501:8501 ai-sports-predictor
```
