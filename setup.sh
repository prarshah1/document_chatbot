deactivate
source ../nvidia_document_bot/venv/bin/activate

pip install -r requirements.txt

streamlit run ./src/apps/nvdiassist_app
