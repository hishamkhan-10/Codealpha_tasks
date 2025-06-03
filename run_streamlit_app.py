
import os
from pyngrok import ngrok
import threading

def run_streamlit():
    os.system('streamlit run streamlit_app.py --server.port 8501')

threading.Thread(target=run_streamlit).start()

public_url = ngrok.connect(8501)
print(f"Your app is live at: {public_url}")
