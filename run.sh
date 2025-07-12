source ~/.virtualenvs/dld/bin/activate

uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0