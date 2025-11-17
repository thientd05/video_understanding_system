#!/bin/bash

# Script chạy Video RAG Web Application (Gradio)

cd "$(dirname "$0")"

echo "🚀 Starting Video RAG Web Interface..."
echo "📖 Open your browser and go to: http://localhost:7860"
echo ""

PYTHONPATH="$(pwd)" ./env/bin/python src/app/web_app.py
