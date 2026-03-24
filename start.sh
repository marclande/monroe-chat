#!/bin/bash
echo "Injecting OG meta tags..."
python inject_og.py
echo "Starting Streamlit..."
streamlit run app.py --server.port $PORT --server.headless true --server.address 0.0.0.0
