# Cardic_image_model_1.1
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_multi_disease_prediction.py
git add requirements.txt
git commit -m "Update requirements.txt with TensorFlow 2.16.0+ for Python 3.12 compatibility"
git push origin main
