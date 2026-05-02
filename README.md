# Maalde
Design Demand Prediction Engine

Grisa Desai
Mobile No. - 9638223863
Github Repo - https://github.com/GrishaDesai/Maalde.git

SETUP Instructions - 

# 1. Clone & install dependencies
git clone <repo-url> && cd demand-oracle
pip install -r requirements.txt

# 2. Add environment variables
    One Gemini key
    GEMINI_API_KEY=your_key
    
    # OR multiple fallback keys
    GEMINI_API_KEY_1=key_one
    GEMINI_API_KEY_2=key_two
    GEMINI_API_KEY_3=key_three
    
    # OpenRouter (backup when all Gemini exhausted)
    OPENROUTER_KEY=your_openrouter_key
    
    # Google Drive for product images
    GOOGLE_API_KEY=your_google_key
    DRIVE_FOLDER_ID=your_folder_id


# 3. Place model files in models_clean/
#    model.pkl | ohe.pkl | scaler.pkl | config.json | resnet_features.pkl


# 4. Run the app
streamlit run app.py

Download from PyTorch model hub.
ResNet50 weights file resnet50-11ad3fa6.pth