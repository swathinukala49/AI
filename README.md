# AI Projects

## PDF question-answering app

The Streamlit RAG application is in `rag/`.

1. Install the app dependencies with `pip install -r rag/requirements.txt`.
2. Set the `GROQ_API_KEY` environment variable.
3. Run `streamlit run rag/app.py`.

The app accepts a PDF upload, indexes its contents with FAISS, and answers questions using Groq.

## Model experiments

The root-level Python scripts contain salary prediction and sentiment-analysis experiments. They log runs to an MLflow server at `http://127.0.0.1:5000`. The experiments require their respective Python packages in addition to the RAG app dependencies.
