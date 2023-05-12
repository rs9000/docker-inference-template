cd /app
uvicorn app:app --host localhost --port 8000 &
streamlit run zoo.py --server.port 8001 --server.address=localhost