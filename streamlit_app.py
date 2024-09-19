import joblib
import pandas as pd
import numpy as np
import json
import streamlit as st
from xgboost import XGBRegressor
import datetime 
import base64
import os
import sklearn
import xgboost as xgb
import requests
from io import StringIO
from dotenv import load_dotenv

# GitHub repository details

load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Replace with your GitHub personal access token
if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
REPO_OWNER = 'RayyanPP'                      # Replace with your GitHub username
REPO_NAME = 'project'                        # Replace with your repository name
FILE_PATH = 'file_path.csv'                  # Path to the CSV file in the repository

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)


# Load models
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_info["xgb_model_path"])

gb_model = joblib.load(model_info["gb_model_path"])
# Extract weights
weight3 = model_info["weights"]["weight3"]
weight4 = model_info["weights"]["weight4"]

def add_bg_from_local(image_path):
    # Check if the file exists before proceeding
    if not os.path.exists(image_path):
        st.error(f"File not found: {image_path}")
        return

    try:
        # Read the image file and encode it as base64
        with open(image_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode()
        
        # CSS to set the background image
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64_image}");
                background-size: cover;
                background-position: center; /* Center the image */
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An error occurred while reading the image: {e}")

def remove_bg():
    # CSS to remove the background image
    st.markdown(
        """
        <style>
        .stApp {{
            background-image: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def save_to_file(name, email, message):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # Get the current file content from GitHub
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()
        file_sha = content['sha']  # Current SHA of the file
        
        # Decode the current content and load it into a DataFrame
        csv_content = base64.b64decode(content['content']).decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Append the new data
        new_entry = pd.DataFrame({
            'Name': [name],
            'Email': [email],
            'Message': [message],
            'Timestamp': [datetime.datetime.now()]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        
        # Convert the updated DataFrame back to CSV
        updated_csv = df.to_csv(index=False)
        
        # Encode updated CSV content to base64
        updated_content = base64.b64encode(updated_csv.encode()).decode('utf-8')
        
        # Prepare the payload for GitHub API
        data = {
            "message": "Updated contact details",
            "content": updated_content,
            "sha": file_sha
        }
        
        # Commit the updated file to GitHub
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        
        return True
    
    except Exception as e:
        st.error(f"Failed to save data: {e}")
        return False
        
def show_home_page():
    add_bg_from_local(r'seccar.jpg')  # Ensure the path is correct
    st.markdown('<h1 class="main-title">Car Price Prediction Using Hybrid ML Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Welcome to the Car Price Prediction App. Fill in the details to predict the car price</p>', unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
    .main-title {
        color: #FF7F50; /* Coral */
        text-align: center; /* Center alignment */
        font-family: 'Roboto', sans-serif; /* Custom font */
        font-size: 2.8em; /* Large font size */
        font-weight: bold; /* Bold text */
        text-transform: uppercase; /* Uppercase text */
        background: linear-gradient(90deg, rgba(255,99,71,1) 100%, rgba(255,255,255,1) 100%); /* Gradient background */
        -webkit-background-clip: text; /* Clip background to text */
        -webkit-text-fill-color: transparent; /* Transparent text fill */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); /* Text shadow */
        margin: 20px 0; /* Margin above and below */
        }
    .sub-text {
        color: #0A0A0A; /* Greyish blue */
        text-align: center; /* Center alignment */
        font-family: 'Pacifico', cursive; /* Custom font */
        font-size: 1.1em; /* Medium font size */
        margin-top: 10px; /* Margin above */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    # Car Brands and Models selection
    car_brands_and_models = {
    'Maruti Suzuki': ['Swift Deca Limited Edition VDi','Swift Windsong Limited edition VXI','Swift Windsong Limited edition VDI','Swift VXi Glory Edition','Swift VXi ABS [2014-2017]','Swift VXi ABS','Swift VXi [2014-2017]','Swift VXi','Swift VDi Glory Edition','Swift VDi ABS [2014-2017]','Swift VDi ABS','Swift VDi [2014-2017]','Swift ZDi','Swift VDi','Swift Lxi ABS(O)','Swift Lxi (O) [2014-2017]','Swift Lxi (O)','Swift Lxi',
                      'Swift Limited Edition Petrol','Swift Limited Edition Diesel','Swift LDi ABS [2014-2017]','Swift LDi ABS','Swift LDi [2014-2017]','Swift LDi','Swift Deca Limited Edition','Swift Lxi ABS [2014-2017]','Swift ZXi','Alto LX','Alto LX (Airbag) [2016-2019]','Alto LX [2014-2019]', 'Alto LXi','Alto LXi (Airbag) [2016-2019]','Alto LXi [2014-2019]','Alto LXi CNG','Alto LXi CNG (Airbag) [2014-2019]','Alto LXi CNG [2014-2018]',
                      'Alto VXi','Alto VXi (O)','Alto VXi (O) [2014-2019]','Alto VXi [2014-2019]','Alto VXi AMT','Alto 800 LX (O) [2016-2019]','Alto 800 LX [2016-2019]','Alto 800 LXi', 'Alto 800 LXi (O)','Alto 800 LXi CNG','Alto 800 LXi CNG (O)','Alto 800 STD (O)','Alto 800 STD','Alto 800 VXi','Alto 800 VXi (O)','Baleno Alpha 1.2','Baleno Alpha 1.2 AT','Baleno Alpha 1.3', 'Baleno Delta 1.2','Baleno Delta 1.2 AT','Baleno Delta 1.3',
                      'Baleno RS 1.0','Baleno Sigma 1.2','Baleno Sigma 1.3','Baleno Zeta 1.2','Baleno Zeta 1.2 AT','Baleno Zeta 1.3','Celerio LXi','Celerio VXi AMT [2019-2020]','Celerio VXi CNG', 'Celerio VXi CNG [2017-2019]','Celerio ZXi','Celerio ZXi (O) AMT','Celerio VXi AMT [2017-2019]','Celerio ZXi (O) AMT [2017-2019]','Celerio ZXi (O)','Celerio ZXi (O) [2017-2019]','Celerio ZXi (O) [2019-2020]','Celerio ZXi [2017-2019]',
                      'Celerio ZXi [2019-2020]','Celerio ZXi AMT','Celerio ZXi (O) AMT [2019-2020]','Celerio VXi AMT','Celerio VXi [2019-2020]','Celerio VXi [2017-2019]','Celerio LXi (O)','Celerio LXi (O) [2017-2019]','Celerio LXi (O) [2019-2020]','Celerio LXi [2017-2019]','Celerio LXi [2019-2020]','Celerio VXi','Celerio VXi (O)','Celerio VXi (O) [2017-2019]','Celerio VXi (O) [2019-2020]','Celerio VXi (O) AMT','Celerio VXi (O) AMT [2017-2019]',
                      'Celerio VXi (O) AMT [2019-2020]','Celerio VXi (O) CNG','Celerio VXi (O) CNG [2017-2019]','Celerio VXi (O) CNG [2019-2020]','Celerio ZXi AMT [2017-2019]','Celerio ZXi AMT [2019-2020]','Celerio X VXi','Celerio X ZXi AMT','Celerio X ZXi [2019-2020]', 'Celerio X ZXi (O) [2017-2019]','Celerio X ZXi (O)','Celerio X ZXi (O) AMT [2019-2020]','Celerio X ZXi (O) AMT [2017-2019]','Celerio X ZXi (O) AMT','Celerio X ZXi','Celerio X VXi AMT [2019-2020]',
                      'Celerio X VXi AMT [2017-2019]','Celerio X VXi AMT','Celerio X VXi [2019-2020]','Celerio X VXi [2017-2019]','Celerio X VXi (O) AMT [2019-2020]','Celerio X VXi (O) AMT [2017-2019]','Celerio X VXi (O) AMT','Celerio X VXi (O) [2019-2020]','Celerio X VXi (O) [2017-2019]','Celerio X VXi (O)','Celerio X ZXi AMT [2017-2019]','Celerio X ZXi AMT [2019-2020]','Ciaz Sigma 1.5','Ciaz Delta 1.5','Ciaz Zeta 1.5', 'Ciaz Delta 1.5 AT','Ciaz Alpha 1.5 Dual Tone',
                      'Ciaz Alpha 1.5','Ciaz Zeta 1.5 AT','Ciaz Alpha 1.5 AT','Ciaz Alpha 1.5 AT Dual Tone','Ciaz Sigma 1.3 Diesel','Ciaz Sigma 1.5 [2020-2023]','Ciaz Alpha 1.3 Diesel','Ciaz Zeta 1.3 Diesel','Ciaz S 1.5 MT [2020-2023]','Ciaz Zeta 1.5 [2020-2023]','Ciaz Zeta 1.5 AT [2020-2023]','Ciaz Zeta 1.5 Diesel','Ciaz Sigma Hybrid 1.5 [2018-2020]','Ciaz S 1.5 MT','Ciaz Delta 1.5 AT [2020-2023]','Ciaz Delta Hybrid 1.5 [2018-2020]','Ciaz Delta 1.5 Diesel',
                      'Ciaz Zeta Hybrid 1.5 [2018-2020]','Ciaz Delta 1.5 [2020-2023]','Ciaz Delta 1.3 Diesel','Ciaz Alpha Hybrid 1.5 AT [2018-2020]','Ciaz Alpha Hybrid 1.5 [2018-2020]','Ciaz Alpha 1.5 Dual Tone [2023]','Ciaz Alpha 1.5 Diesel','Ciaz Alpha 1.5 AT Dual Tone [2023]','Ciaz Alpha 1.5 AT [2020-2023]','Ciaz Alpha 1.5 [2020-2023]','Ciaz Delta Hybrid 1.5 AT [2018-2020]','Ciaz Zeta Hybrid 1.5 AT [2018-2020]',
                      'Dzire LDi','Dzire LDi Special Edition','Dzire LXi', 'Dzire LXi Special Edition','Dzire VDi','Dzire VDi AMT','Dzire VXi','Dzire VXi AMT','Dzire ZDi','Dzire ZDi AMT','Dzire ZDi Plus','Dzire ZDi Plus AMT','Dzire ZXi','Dzire ZXi AMT','Dzire ZXi Plus','Dzire ZXi Plus AMT''Eeco 5 STR','Eeco 7 STR [2019-2020]','Eeco 7 STR [2014-2019]', 'Eeco 7 STR','Eeco 5 STR WITH HTR CNG [2018-2019]','Eeco 5 STR WITH HTR CNG','Eeco 5 STR WITH A/C+HTR CNG [2019]',
                      'Eeco 5 STR WITH A/C+HTR CNG [2019-2020]','Eeco 5 STR WITH A/C+HTR CNG [2017-2019]','Eeco 5 STR WITH A/C+HTR CNG','Eeco 5 STR WITH A/C+HTR[2019]','Eeco 5 STR WITH A/C+HTR [2019-2020]','Eeco 5 STR WITH A/C+HTR[2014-2019]','Eeco 5 STR WITH A/C+HTR','Eeco 5 STR STD (O)','Eeco 5 STR AC (O) CNG','Eeco 5 STR AC (O)','Eeco 5 STR [2019]','Eeco 5 STR [2019-2020]',
                      'Eeco 5 STR [2014-2019]','Eeco 7 STR [2019]','Eeco 7 STR STD (O)','Ertiga LDI (O) SHVS','Ertiga LDI SHVS','Ertiga LXI', 'Ertiga LXI (O)','Ertiga VDI Limited Edition [2017]','Ertiga VDI SHVS','Ertiga VXI','Ertiga VXI AT','Ertiga VXI CNG','Ertiga VXI Limited Edition [2017]','Ertiga ZDI+SHVS','Ertiga ZDI SHVS','Ertiga ZXI','Ertiga ZXI+',
                      'Omni 5 STR BS-II','Omni 5 STR BS-III','Omni 5 STR BS-IV', 'Omni 5 STR','Omni 8 STR BS-II','Omni 8 STR BS-III','Omni 8 STR','Omni Ambulance','Omni Cargo','Omni Cargo BS-III','Omni Cargo BS-IV','Omni Cargo LPG BS-III','Omni CNG','Omni E 8 STR BS-IV','Omni LPG BS-III','Omni LPG BS-IV',
                      'Vitara Brezza LDi','Vitara Brezza LDi (O) [2016-2018]','Vitara Brezza VDi', 'Vitara Brezza VDi (O) [2016-2018]','Vitara Brezza VDi AGS','Vitara Brezza ZDi','Vitara Brezza ZDi AGS','Vitara Brezza ZDi Plus','Vitara Brezza ZDi Plus Dual Tone','Vitara Brezza ZDi Plus AGS','Vitara Brezza ZDi Plus Dual Tone AGS','Vitara Brezza ZDi Plus Dual Tone [2017-2018]','Gypsy King HT BS-IV','Gypsy King ST BS-IV',
                      'Wagon R LX','Wagon R VXI+ (O)','Wagon R VXI+', 'Wagon R VXI AMT (O)','Wagon R VXI AMT','Wagon R VXI ABS','Wagon R VXI ABS (Airbag)','Wagon R VXI+ AMT','Wagon R VXI','Wagon R LXI CNG Avance LE','Wagon R LXI CNG (O)','Wagon R LXI CNG','Wagon R LXI Avance LE','Wagon R LXI ABS','Wagon R LXI','Wagon R LXI LPG','Wagon R VXI+ AMT (O)',
                      'S-Cross Alpha 1.3','S-Cross Delta 1.3','S-Cross Sigma 1.3','S-Cross Zeta 1.3','Ignis Alpha 1.2 AMT','Ignis Alpha 1.2 MT','Ignis Alpha 1.3 AMT Diesel [2017-2018]', 'Ignis Alpha 1.3 Diesel [2017-2018]','Ignis Delta 1.2 AMT','Ignis Delta 1.2 MT','Ignis Delta 1.3 AMT Diesel [2017-2018]','Ignis Delta 1.3 Diesel [2017-2018]','Ignis Sigma 1.2 MT',
                      'Ignis Zeta 1.2 AMT','Ignis Zeta 1.2 MT','Ignis Zeta 1.3 AMT Diesel [2017-2018]','Ignis Zeta 1.3 Diesel [2017-2018]'
                     ],
    'Hyundai': ['Grand i10 Asta Kappa VTVT','Grand i10 Sportz AT 1.2 Kappa VTVT','Grand i10 Sportz 1.2 Kappa VTVT Dual Tone', 'Grand i10 Sportz 1.2 Kappa VTVT [2017-2020]','Grand i10 Sportz 1.2 Kappa VTVT','Grand i10 Sportz (O) U2 1.2 CRDi [2017-2018]','Grand i10 Sportz (O) AT 1.2 Kappa VTVT [2017-2018]','Grand i10 Sportz (O) 1.2 Kappa VTVT [2017-2018]','Grand i10 Sportz U2 1.2 CRDi','Grand i10 Magna U2 1.2 CRDi',
                'Grand i10 Magna 1.2 Kappa VTVT CNG [2019-2020]','Grand i10 Magna 1.2 Kappa VTVT CNG','Grand i10 Magna 1.2 Kappa VTVT [2017-2020]','Grand i10 Magna 1.2 Kappa VTVT','Grand i10 Era U2 1.2 CRDi','Grand i10 Era 1.2 Kappa VTVT','Grand i10 Asta U2 1.2 CRDi','Grand i10 Magna AT 1.2 Kappa VTVT','Grand i10 Sportz U2 1.2 CRDi Dual Tone',
               'Creta E 1.6 Petrol','Creta E Plus 1.4 CRDi','Creta E Plus 1.6 Petrol', 'Creta S 1.4 CRDi','Creta S 1.6 AT CRDi','Creta SX 1.6 (O) Petrol','Creta SX 1.6 AT CRDi','Creta SX 1.6 AT Petrol','Creta SX 1.6 CRDi','Creta SX 1.6 CRDi (O)','Creta SX 1.6 CRDi Dual Tone','Creta SX 1.6 Dual Tone Petrol','Creta SX 1.6 Petrol',
               'Elantra 1.6 S MT','Elantra 1.6 SX (O)','Elantra 1.6 SX (O) AT', 'Elantra 1.6 SX MT','Elantra 2.0 S MT','Elantra 2.0 SX (O)','Elantra 2.0 SX (O) AT','Elantra 2.0 SX AT','Elantra 2.0 SX MT',
               'Elite i20 Asta 1.2','Elite i20 Asta 1.2 (O)','Elite i20 Asta 1.2 Dual Tone', 'Elite i20 Asta 1.4 CRDi','Elite i20 Asta 1.4 CRDi (O)','Elite i20 Asta 1.4 CRDi Dual Tone','Elite i20 Era 1.2','Elite i20 Era 1.4 CRDi','Elite i20 Magna 1.4 AT','Elite i20 Magna Executive 1.2','Elite i20 Magna Executive 1.4 CRDi','Elite i20 Sportz 1.2','Elite i20 Sportz 1.4 CRDi',
               'Eon 1.0 Kappa Era+','Eon Magna+ SE','Eon Magna+ LPG [2012-2016]', 'Eon Magna+ Airbag','Eon Magna+','Eon Magna [2011-2012]','Eon Era+ SE','Eon Era+ LPG Airbag','Eon Era+ LPG','Eon Era+ Airbag','Eon Magna O [2011-2012]','Eon Era+','Eon D-Lite O [2011-2012]','Eon D-Lite+ LPG [2012-2015]','Eon D-Lite+ Airbag','Eon D-Lite+','Eon D-Lite','Eon 1.0 Kappa Magna Airbag','Eon 1.0 Kappa Magna+ (O) [2014-2016]',
                'Eon Kappa Magna+ [2014-2016]','Eon 1.0 Kappa Magna (O) Airbag','Eon Era [2011-2012]','Eon Sportz',
               'Santro Asta','Santro Sportz CNG [2018-2020]','Santro Sportz CNG', 'Santro Sportz AMT SE [2019-2020]','Santro Sportz AMT SE','Santro Sportz AMT [2018-2020]','Santro Sportz AMT','Santro Sportz [2018-2020]','Santro Sportz','Santro Magna Corporate Edition','Santro Magna CNG [2018-2020]','Santro Sportz SE','Santro Magna CNG','Santro Magna AMT [2018-2020]','Santro Magna AMT',
                'Santro Magna [2018-2020]','Santro Magna','Santro Era Executive [2019-2020]','Santro Era Executive','Santro Era','Santro Dlite','Santro Asta AMT','Santro Asta [2018-2020]','Santro Magna AMT Corporate Edition','Santro Sportz SE [2019-2020]',
               'Xcent E','Xcent E CRDi','Xcent E Plus', 'Xcent E Plus CRDi','Xcent S','Xcent S AT','Xcent S CRDi','Xcent SX','Xcent SX (O)','Xcent SX (O) CRDi','Xcent SX CRDi',
               'i20 Active 1.2 [2015-2016]','i20 Active 1.2 Base','i20 Active 1.2 S', 'i20 Active 1.2 SX','i20 Active 1.2 SX [2015-2016]','i20 Active SX Dual Tone','i20 Active 1.4 [2015-2016]','i20 Active 1.4 [2016-2017]','i20 Active 1.4 S','i20 Active 1.4 SX','i20 Active 1.4 SX [2015-2016]','i20 Active SX Dual Tone','i20 Active 1.4L SX (O) [2015-2016]',
               'Verna E 1.4 CRDi','Verna SX 1.6 VTVT','Verna SX 1.6 CRDi', 'Verna SX (O) 1.6 VTVT','Verna SX (O) Anniversary Edition 1.6 VTVT','Verna SX (O) Anniversary Edition 1.6 CRDi','Verna SX (O) 1.6 VTVT AT','Verna SX (O) 1.6 CRDi AT','Verna SX (O) 1.6 CRDi','Verna EX 1.6 VTVT [2017-2018]','Verna EX 1.6 CRDi AT [2017-2018]','Verna EX 1.6 CRDi [2017-2018]','Verna EX 1.4 VTVT',
                'Verna EX 1.4 CRDi','Verna E 1.6 VTVT [2017-2018]','Verna E 1.6 CRDi [2017-2018]','Verna E 1.4 VTVT','Verna EX 1.6 VTVT AT [2017-2018]','Verna SX Plus 1.6 VTVT AT',
               'Santa Fe 2WD AT [2014-2017]','Santa Fe 2WD MT [2014-2017]','Santa Fe 4WD AT [2014-2017]',
               'Tucson 2WD AT GLS Diesel','Tucson 2WD MT Diesel','Tucson 2WD MT Petrol', 'Tucson GL (O) 2WD AT Diesel','Tucson GL (O) 2WD AT Petrol','Tucson GL 2WD AT Diesel','Tucson GL 2WD AT Petrol','Tucson GLS 2WD AT Petrol','Tucson GLS 4WD AT Diesel',
               'Seltos','Carens','Sonet','EV6'],
    'Tata': ['Aria Pleasure 4x2','Aria Pride 4x4','Aria Pure LX 4x2',
            'Bolt XE Diesel','Bolt XE Petrol','Bolt XM Diesel', 'Bolt XM Petrol','Bolt XMS Petrol','Bolt XMS Diesel','Bolt XT Petrol','Bolt XT Diesel',
            'Estate Std','Grande CX 7 STR','Grande CX 8 STR','Grande CX 9 STR', 'Grande LX 7 STR','Grande LX 8 STR','Grande LX 9 STR',
            'Hexa XE 4x2 7 STR','Hexa XM 4x2 7 STR','Hexa XM Plus 4x2 7 STR', 'Hexa XMA 4x2 7 STR','Hexa XT 4x2 6 STR','Hexa XT 4x2 7 STR','Hexa XT 4x4 6 STR','Hexa XT 4x4 7 STR','Hexa XTA 4x2 6 STR','Hexa XTA 4x2 7 STR',
            'Indica GLS eMAX','Indica GLX eMAX','Indica LS', 'Indica LX','Indigo GLS','Indigo GLS eMAX','Indigo GLX','Indigo GLX eMAX', 'Indigo GVX','Indigo LS CR4 BS-IV','Indigo LS TDI BS-III','Indigo LS CR4 BS-IV','Indigo LX TDI BS-III','Indigo VX CR4 BS-IV',
            'Nano XE','Nano XM','Nano XMA','Nano XT','Nano XTA','Nexon KRAZ Diesel','Nexon XZA Plus Diesel Dual Tone','Nexon XZA Plus Diesel','Nexon XZ Plus Dual Tone', 'Nexon XZ Plus Diesel Dual Tone','Nexon XZ Plus DIesel','Nexon XZ Plus','Nexon XZ Diesel',
             'Nexon XZ','Nexon XT Plus Diesel','Nexon XT Plus','Nexon XT Diesel [2017-2019]','Nexon XT Diesel','Nexon XT [2017-2019]','Nexon XT','Nexon XMA Petrol','Nexon XMA Diesel','Nexon XM Diesel','Nexon XM','Nexon XE Diesel','Nexon XE','Nexon KRAZ Plus Petrol','Nexon KRAZ Plus Diesel',
             'Nexon KRAZ Plus AMT Petrol','Nexon KRAZ Plus AMT Diesel','Nexon KRAZ Petrol','Nexon KRAZ MT Petrol','Nexon KRAZ MT Diesel','Nexon XZA Plus Petrol','Nexon XZA Plus Petrol Dual Tone',
            'Safari 4x2 EX DICOR 2.2 VTT','Safari 4x4 GX DICOR BS-IV','Safari 4x4 GX DICOR 2.2 VTT','Safari 4x4 EXi BS-III', 'Safari 4x4 EX DICOR BS-IV','Safari 4x4 EX DICOR 2.2 VTT','Safari 4x2 VX DICOR BS-IV','Safari 4x2 VX DICOR 2.2 VTT','Safari 4x2 LX TCIC','Safari 4x2 LX DICOR BS-III','Safari 4x2 LX DICOR BS-IV','Safari 4x2 LX DICOR 2.2 VTT',
             'Safari 4x2 GX DICOR BS-IV','Safari 4x2 GX DICOR BS-III','Safari 4x2 GX DICOR 2.2 VTT','Safari 4x2 EXi BS-III','Safari 4x2 EX DICOR BS-III','Safari 4x2 EX DICOR BS-IV','Safari 4x4 VX DICOR 2.2 VTT','Safari 4x4 VX DICOR BS-IV',
            'Sierra Std','Sierra Turbo','Sumo CX BS-III','Sumo CX BS-IV','Sumo CX PS BS-III','Sumo CX PS BS-IV', 'Sumo EX BS-III','Sumo EX BS-IV','Sumo FX BS-IV','Sumo GX BS-IV','Sumo LX BS-III','Sumo LX BS-IV','Sumo SA Gold','Tiago Revotorq XB','Tiago Revotron XM','Tiago Revotron XM (O)[2016-2019]','Tiago Revotron XM [2016-2019]',
             'Tiago Revotron XT (O) [2016-2019]','Tiago Revotron XT [2016-2019]','Tiago Revotron XTA [2017-2019]','Tiago Revotron XZ','Tiago Revotron XZ Plus','Tiago Revotron XZ Plus Dual Tone','Tiago Revotron XZ Plus Dual Tone [2018-2019]','Tiago Revotron XZ w/o Alloy [2018-2019]','Tiago Revotron XZ (O)','Tiago Revotron XZA',
             'Tiago Revotron XZA [2017-2019]','Tiago Revotron XZA Plus','Tiago Revotron XZA Plus Dual Tone','Tiago Wizz Edition Diesel','Tiago Revotron XE [2016-2019]','Tiago Wizz Edition Petrol','Tiago Revotron XE (O) [2016-2019]','Tiago Revotron XB [2016-2018]','Tiago Revotorq XB [2016-2018]','Tiago Revotorq XE','Tiago Revotorq XE (O) [2016-2019]',
             'Tiago Revotorq XE [2016-2019]','Tiago Revotorq XM','Tiago Revotorq XM (O) [2016-2019]','Tiago Revotorq XM [2016-2019]','Tiago Revotorq XT (O) [2016-2019]','Tiago Revotorq XT [2016-2019]','Tiago Revotorq XZ','Tiago Revotorq XZ [2016-2019]','Tiago Revotorq XZ Plus','Tiago Revotorq XZ Plus Dual Tone','Tiago Revotorq XZ Plus Dual Tone [2018-2019]',
             'Tiago Revotorq XZ w/o Alloy [2018-2019]','Tiago Revotorq XZ (O)','Tiago Revotron XB','Tiago Revotron XE','Tiago Wizz Edition Petrol [2017-2018]','Zest Premio','Zest XE 75 PS Diesel','Zest XE Petrol','Zest XM 75 PS Diesel', 'Zest XM Diesel','Zest XM Petrol','Zest XMA Diesel','Zest XMS 75 PS Diesel','Zest XMS Diesel',
             'Zest XMS Diesel Anniversary LE','Zest XMS Petrol','Zest XMS Petrol Anniversary LE','Zest XT Diesel','Zest XT Petrol','Zest XTA Diesel','Tiago NRG AMT','Tiago NRG Diesel','Tiago NRG Petrol','Venture CX','Venture EX 7 STR','Venture EX 8 STR','Venture GX 8 STR', 'Venture GX 7 STR','Venture LX 7 STR','Venture LX 8 STR',
             'Tigor Buzz Diesel','Tigor Buzz Petrol','Tigor Revotorq XE','Tigor Revotorq XM', 'Tigor Revotorq XT','Tigor Revotorq XZ','Tigor Revotorq XZ (O)','Tigor Revotron XE','Tigor Revotron XM','Tigor Revotron XT','Tigor Revotron XTA','Tigor Revotron XZ','Tigor Revotron XZ (O)','Tigor Revotron XZA',
             'Tiago JTP 1.2','Tiago JTP 1.2 [2018-2019]','Tigor JTP 1.2','Tigor JTP 1.2 [2018-2019]','Vista LS BS-III','Vista LS BS-IV','Vista LX BS-III','Vista VX BS-IV','TL 4x4'
            ],
    'Mahindra': ['Alturas G4 2WD AT','Alturas G4 2WD AT [2018-2020]','Alturas G4 2WD High AT','Alturas G4 4WD AT', 'Alturas G4 4WD AT [2018-2020]',
                'Armada AC ','Armada Grand 2WD','Armada Grand 4WD','Armada Std',
                'Bolero Camper','Bolero SLX BS-IV','Bolero SLX BS-III','Bolero SLE BS-IV', 'Bolero SLE BS-III','Bolero Power Plus ZLX [2016-2019]','Bolero Power Plus ZLX','Bolero Power Plus SLX [2016-2019]','Bolero Poer Plus SLX','Bolero Power Plus SLE [2016-2019]','Bolero Power Plus SLE','Bolero Power Plus LX [2017]','Bolero ZLX BS-III',
                 'Bolero Power Plus LX','Bolero Plus BS-III','Bolero Plus AC BS-IV','Bolero Plus AC BS-III','Bolero EX BS-IV','Bolero EX AC BS-IV','Bolero DI BS-III','Bolero DI AC BS-III','Bolero DI 4WD BS-III','Bolero Plus BS-IV','Bolero ZLX BS-IV','e2o Plus P4','e2o Plus P6','e2o Plus P8',
                'Jeep CJ 340','Jeep MM 550 XDB','Jeep MM 550 PE','Jeep MM 550 DP', 'Jeep MM 540 XDB','Jeep MM 540 DP','Jeep MM 540','Jeep Commander 750 ST','Jeep MM 775 XDB','Jeep Commander 750 DP','Jeep Commander 650 DI','Jeep Classic','Jeep CL 550 MDI','Jeep CL 500 MDI','Jeep CJ 500 DI','Jeep CJ 500 D','Jeep CJ 340 DP',
                 'Jeep Commander 750 DI','Jeep MM ISZ Petrol',
                'KUV100 K2 6 STR','KUV100 K8 D 6 STR','KUV100 K8 D 5 STR','KUV100 K8 6 STR Dual Tone [2017-2020]', 'KUV100 K8 6 STR Dual Tone','KUV100 K8 6 STR','KUV100 K8 5 STR','KUV100 K6 Plus D 6 STR','KUV100 K6 Plus D 5 STR','KUV100 K6 Plus 6 STR [2017-2020]','KUV100 K6 Plus 6 STR','KUV100 K6 Plus 5 STR','KUV100 K4 Plus D 6 STR',
                 'KUV100 K4 Plus D 5 STR','KUV100 K4 Plus 6 STR [2017-2020]','KUV100 K4 Plus 6 STR','KUV100 K4 Plus 5 STR','KUV100 K2 Plus D 6 STR','KUV100 K2 Plus 6 STR [2017-2020]','KUV100 K2 Plus 6 STR','KUV100 K2 D 6 STR','KUV100 K8 D 6 STR Dual Tone','KUV100 Trip 6S CNG','Logan/Verito 1.5 D2','Logan/Verito 1.5 D4','Logan/Verito 1.5 D6',
                'Marazzo M2 7 STR','Marazzo M2 8 STR','Marazzo M4 7 STR','Marazzo M4 8 STR', 'Marazzo M6 7 STR','Marazzo M6 8 STR','Marazzo M8 7 STR','Marazzo M8 8 STR',
                'Marshal DI DX','Marshal DI','Marshal DX Royale','Marshal Std','NuvoSport N4','NuvoSport N4 Plus','NuvoSport N6','NuvoSport N6 AMT', 'NuvoSport N8','NuvoSport N8 AMT',
                'Scorpio 2WD BS-III','Scorpio 2WD BS-IV','Scorpio 4WD BS-III','Scorpio 4WD BS-IV','Thar 700 Special Edition','Thar CRDe 4x4 ABS','Thar CRDe 4x4 AC','Thar CRDe 4x4 AC1', 
                 'Thar CRDe 4x4 Non AC','Thar DI 2WD BS-IV','Thar DI 2WD PS BS-III','Thar DI 4WD BS-IV','Thar DI 4WD BS-III','Thar DI 4WD PS BS-IV','TUV300 P4 Plus','TUV300 T10','TUV300 T10 AMT','TUV300 T10 AMT Dual Tone', 'TUV300 T10 Dual Tone','TUV300 T4','TUV300 T4 Plus',
                 'TUV300 T6','TUV300 T6 Plus','TUV300 T6 Plus AMT','TUV300 T8','TUV300 T8 AMT','TUV300 T8 AMT mHAWK100','TUV300 T8 mHAWK100','TUV300 T8 mHAWK100 Dual Tone','Voyager AC','Voyager Std','XUV500 G AT','XUV500 W9 1.99','XUV500 W9','XUV500 W8 AWD [2015-2017]', 'XUV500 W8 AT 1.99 [2016-2017]',
                 'XUV500 W8 AT [2015-2017]','XUV500 W8 1.99 [2016-2017]','XUV500 W8 [2015-2017]','XUV500 W6 AT 1.99','XUV500 W6 AT','XUV500 W6 1.99','XUV500 W6','XUV500 W4 1.99','XUV500 W9 AT','XUV500 W4 [2015-2016]','XUV500 W10 Black Interiors [2017]','XUV500 W10 AWD Black Interiors [2017]',
                 'XUV500 W10 AWD AT Black Interiors [2017]','XUV500 W10 AWD AT','XUV500 W10 AWD','XUV500 W10 AT Black Interiors [2017]','XUV500 W10 AT 1.99','XUV500 W10 AT','XUV500 W10 1.99','XUV500 W10','XUV500 Sportz Edition MT','XUV500 Sportz Edition AT','XUV500 W4','XUV500 W9 AT 1,99',
                'Xylo D2 BS-III','Xylo D2 BS-IV','Xylo D4 BS-III','Xylo D4 BS-IV', 'Xylo H4 ABS Airbag BS-IV','Xylo H4 BS-IV','Xylo H8 ABS Airbag BS-IV','Xylo H8 ABS BS-IV','Xylo H9 BS-IV'],
    'Kia': ['Seltos','Carens','Sonet','EV6'],
    'Toyota': ['Innova Crysta 2.4 G 7 STR','Innova Crysta 2.7 GX 8 STR [2016-2020]','Innova Crysta 2.7 GX AT 7 STR','Innova Crysta 2.7 GX AT 7 STR [2016-2020]','Innova Crysta 2.7 GX AT 8 STR','Innova Crysta 2.7 GX AT 8 STR [2016-2020]','Innova Crysta 2.7 VX 7 STR','Innova Crysta 2.7 VX 7 STR [2016-2020]','Innova Crysta 2.7 ZX AT 7 STR',
               'Innova Crysta 2.7 ZX AT 7 STR [2016-2020]','Innova Crysta 2.8 GX AT 7 STR [2016-2020]','Innova Crysta 2.8 GX AT 8 STR [2016-2020]','Innova Crysta 2.8 ZX AT 7 STR [2016-2020]','Innova Crysta Leadership Edition','Innova Crysta Touring Sport Diesel AT','Innova Crysta Touring Sport Diesel AT [2017-2020]','Innova Crysta Touring Sport Diesel MT',
               'Innova Crysta Touring Sport Diesel MT [2017-2020]','Innova Crysta Touring Sport Petrol AT','Innova Crysta Touring Sport Petrol AT [2017-2020]','Innova Crysta 2.7 GX 8 STR','Innova Crysta 2.7 GX 7 STR [2016-2020]','Innova Crysta 2.7 GX 7 STR','Innova Crysta 2.4 ZX AT 7 STR','Innova Crysta 2.4 G 7 STR [2016-2017]','Innova Crysta 2.4 G 8 STR',
               'Innova Crysta 2.4 G 8 STR [2016-2017]','Innova Crysta 2.4 G Plus 7 STR','Innova Crysta 2.4 G Plus 7 STR [2019-2020]','Innova Crysta 2.4 G Plus 8 STR','Innova Crysta 2.4 G Plus 8 STR [2019-2020]','Innova Crysta 2.4 GX 7 STR','Innova Crysta 2.4 GX 7 STR [2016-2020]','Innova Crysta Touring Sport Petrol MT','Innova Crysta 2.4 GX 8 STR',
               'Innova Crysta 2.4 GX AT 7 STR','Innova Crysta 2.4 GX AT 8 STR','Innova Crysta 2.4 V Diesel','Innova Crysta 2.4 VX 7 STR','Innova Crysta 2.4 VX 7 STR [2016-2020]','Innova Crysta 2.4 VX 8 STR','Innova Crysta 2.4 VX 8 STR [2016-2020]','Innova Crysta 2.4 ZX 7 STR','Innova Crysta 2.4 ZX 7 STR [2016-2020]','Innova Crysta 2.4 GX 8 STR [2016-2020]',
               'Innova Crysta Touring Sport Petrol MT [2017-2020]',
              'Fortuner 2.7 4x2 AT','Fortuner TRD Celebratory Edition','Fortuner 2.8 4x4 MT [2016-2020]','Fortuner 2.8 4x4 MT','Fortuner 2.8 4x4 TRD Limited Edition','Fortuner 2.8 4x4 AT [2016-2020]','Fortuner 2.8 4x4 AT','Fortuner TRD Celebratory Edition [2019-2020]','Fortuner 2.8 4x2 MT [2016-2020]','Fortuner 2.8 4x2 AT TRD Limited Edition','Fortuner 2.8 4x2 AT [2016-2020]','Fortuner 2.8 4x2 AT','Fortuner 2.7 4x2 MT [2016-2020]','Fortuner 2.7 4x2 MT','Fortuner 2.7 4x2 AT [2016-2020]','Fortuner 2.8 4x2 MT','Fortuner TRD Sportivo',
              'Yaris G CVT','Yaris VX CVT [2018-2020]','Yaris VX CVT','Yaris V MT OPT Dual Tone','Yaris V MT [2018-2020]','Yaris V MT','Yaris V CVT OPT Dual Tone [2019-2020]','Yaris V CVT OPT Dual Tone','Yaris V CVT','Yaris J MT OPT [2019-2020]','Yaris J MT OPT','Yaris J MT [2018-2020]','Yaris J MT','Yaris J CVT OPT [2019-2020]','Yaris J CVT OPT','Yaris J CVT [2018-2020]','Yaris J CVT','Yaris G MT OPT [2019-2020]','Yaris G MT OPT','Yaris G MT [2018-2020]','Yaris G MT','Yaris G CVT OPT [2019-2020]','Yaris G CVT OPT','Yaris G CVT [2018-2020]','Yaris VX MT','Yaris VX MT [2018-2020]',
              'Etios 1.2 G','Etios 1.2 Limited Edition','Etios 1.4 GD','Etios 1.4 Limited Edition','Etios 1.4 VD','Etios 1.5 V','Etios X-Edition Diesel','Etios X-Edition Petrol',
              'Land Cruiser V8','Etios Liva GX','Etios Liva GXD','Etios Liva V','Etios Liva V Dual Tone','Etios Liva VD','Etios Liva VD Dual Tone','Etios Liva VX','Etios Liva VX Dual Tone','Etios Liva VX Dual Tone LE','Etios Liva VXD','Etios Liva VXD Dual Tone','Etios Liva VXD Dual Tone LE',
              'Corolla Altis G CVT Petrol','Corolla Altis G Diesel','Corolla Altis G Petrol','Corolla Altis GL Diesel','Corolla Altis GL Petrol','Corolla Altis VL CVT Petrol',
              'Prado TX','Prado TX Petrol','Prado VX L','RAV4 Diesel','RAV4 Petrol','Prius 1.8 Z8 CVT','Ventury 2.7 Petrol','Tercel AT','Sera Coupe','MR2 AW11 (4A-GE)',
              'MasterAce Diesel','Mark Grande','Majesta 3.0 V6 AT','Majesta 4.0 V8 AT','Hiace Commuter','Estima Emina 2.0 TDI AT','Estima GL','Estima Lucida',
              'Cynos 1.5 Coupe','Crown Royal Saloon 3.0 V6 AT','Crown Super Saloon 2.4 AT','Cresta Suffire 2.4 Diesel AT','Cressida 2.4 MT Petrol','Cressida XL 2.4 AT','Corona Diesel','Corona Petrol','Corolla Fz','Celica Coupe','Camry 2.5L AT','Camry Hybrid','Camry Hybrid [2015-2017]','Alphard 3.5 VX Petrol AT'],
    'Honda': ['Accord EX Sunroof','Amaze 1.2 E i-VTEC','Amaze Pride Edition Petrol','Amaze Pride Edition Diesel','Amaze 1.5 VX i-DTEC','Amaze 1.5 SX i-DTEC','Amaze 1.5 S i-DTEC Opt','Amaze 1.5 S i-DTEC','Amaze 1.5 E i-DTEC Opt','Amaze Privilege Edition Diesel','Amaze 1.5 E i-DTEC','Amaze 1.2 VX AT i-VTEC','Amaze 1.2 SX i-VTEC','Amaze 1.2 S i-VTEC Opt','Amaze 1.2 S i-VTEC','Amaze 1.2 S AT i-VTEC Opt','Amaze 1.2 S AT i-VTEC','Amaze 1.2 E i-VTEC Opt','Amaze 1.2 VX i-VTEC','Amaze Privilege Edition Petrol',
             'BR-V E Diesel','BR-V VX Petrol','BR-V VX Diesel Style Edition','BR-V VX Diesel [2016-2017]','BR-V VX Diesel','BR-V V Petrol Style Edition','BR-V V Petrol','BR-V V Diesel Style Edition','BR-V V Diesel','BR-V V CVT Petrol Style Edition','BR-V V CVT Petrol','BR-V S Petrol Style Edition','BR-V S Petrol','BR-V S Diesel Style Edition','BR-V S Diesel','BR-V E Petrol','BR-V VX Petrol [2016-2017]','BR-V VX Petrol Style Edition',
             'Brio E MT','Brio S (O) MT','Brio S MT','Brio VX AT','Brio VX MT','City Anniversary Edition Diesel','City ZX Diesel','City ZX CVT Petrol [2017-2019]','City ZX CVT Petrol','City VX Petrol [2017-2019]','City VX Petrol','City VX Diesel','City VX CVT Petrol [2017-2019]','City VX CVT Petrol','City V Petrol [2019-2020]','City V Petrol [2017-2019]','City ZX Petrol','City V Petrol','City V CVT Petrol [2017-2019]','City V CVT Petrol','City SV Petrol Edge Edition','City SV Petrol [2019-2020]','City SV Petrol [2017-2019]','City SV Petrol','City SV Diesel Edge Edition','City SV Diesel','City S Petrol','City Anniversary Edition Petrol','City V Diesel','City ZX Petrol [2019-2019]',
             'Concerto EX 1.6 MT','CR-V 2.0L 2WD AT','CR-V 2.0L 2WD MT','CR-V 2.4L 2WD','CR-V 2.4L 4WD AVN','CR-X Petrol','Jazz Exclusive Edition CVT','Jazz S Diesel','Jazz V CVT Petrol','Jazz V Diesel','Jazz V Petrol','Jazz VX CVT Petrol','Jazz VX Diesel','Jazz VX Petrol',
             'Mobilio E Diesel','Mobilio E Petrol','Mobilio RS Diesel','Mobilio RS (O) Diesel','Mobilio S Diesel','Mobilio S Petrol','Mobilio V (O) Diesel','Mobilio V (O) Petrol','Mobilio V Petrol','Mobilio V Diesel',
             'Step Wagon','WR-V Edge Edition Diesel [2018-2019]','WR-V Edge Edition Petrol [2018-2019]','WR-V Edge Edition Plus Diesel','WR-V Edge Edition Plus Petrol','WR-V Exclusive Edition Diesel','WR-V Exclusive Edition Petrol','WR-V S Diesel Alive Edition','WR-V S MT Diesel','WR-V S MT Petrol','WR-V S Petrol Alive Edition','WR-V V MT Diesel','WR-V VX MT Diesel','WR-V VX MT Petrol'],
    'Renault': ['Captur Platine Diesel Dual Tone','Captur Platine Mono Diesel','Captur RXE Diesel','Captur RXE Petrol','Captur RXL Diesel','Captur RXL Petrol','Captur RXT Diesel Dual Tone','Captur RXT Mono Diesel','Captur RXT Mono Petrol','Captur RXT Petrol Dual Tone',
               'Duster 110 PS RXL 4x2 AMT [2016-2017]','Duster RXL Petrol [2016-2017]','Duster RXL Petrol','Duster RxE Petrol [2016-2017]','Duster RxE Petrol','Duster Adventure Edition 85 PS RXL 4X2 MT','Duster Adventure Edition 85 PS RxE 4X2 MT','Duster Adventure Edition 110 PS RXZ 4X4 MT','Duster 85 PS Sandstorm Edition Diesel','Duster 85 PS RXZ MT Diesel (Opt)','Duster 85 PS RXS 4X2 MT Diesel','Duster 85 PS RXL 4X2 MT [2016-2017]','Duster 85 PS RxE 4X2 MT Diesel','Duster 85 PS Base 4X2 MT Diesel','Duster 110 PS Sandstorm Edition Diesel','Duster 110 PS RXZ 4X4 MT Diesel','Duster 110 PS RXZ 4X2 MT Diesel','Duster 110 PS RXZ AMT Diesel','Duster 110 PS RXS 4X2 AMT Diesel','Duster 110 PS RXL 4X2 MT','Duster RXS CVT','Duster RXS Petrol',
               'Flence Diesel E2 [2014-2017]','Fluence Diesel E4 [2014-2017]','Koleos 4x2 MT [2014-2017]','Koleos 4x4 AT [2014-2017]','Koleos 4x4 MT [2014-2017]',
               'Kwid Marvel Captain America Edition','Kwid RXT Edition','Kwid RXT 1.0 SCE Edition','Kwid RXT [2015-2019]','Kwid RXT (O) 1.0 SCE Eition','Kwid RXL Edition','Kwid RXL [2015-2019]','Kwid RXE Opt [2015-2019]','Kwid RXE [2015-2019]','Kwid Climber 1.0 AMT [2017-2019]','Kwid RXT Opt [2015-2019]','Kwid Climber 1.0 [2017-2019]','Kwid 1.0 RXT Edition','Kwid 1.0 RXT AMT Opt [2016-2019]','Kwid 1.0 RXT [2016-2019]','Kwid 1.0 RXL Edition','Kwid 1.0 RXL AMT [2017-2019]','Kwid 1.0 RXL [2017-2019]','Kwid 1.0 Marvel Iron Man Edition AMT','Kwid 1.0 Marvel Iron Man Edition','Kwid 1.0 Marvel Captain America Edition','Kwid 1.0 RXT OPT [2016-2019]','Kwid STD [2015-2019]',
               'Lodgy 110 PS RxL [2015-2016]','Lodgy 85 PS RXZ Stepway 8 STR','Lodgy 85 PS RXZ [2015-2016]','Lodgy 85 PS RXL 8 STR','Lodgy 85 PS RXL [2015-2016]','Lodgy 85 PS RxE 8 STR','Lodgy 85 PS RxE 7 STR','Lodgy 110 PS World Edition 8 STR','Lodgy 110 PS RXZ Stepway 8 STR','Lodgy 110 PS RXZ Stepway 7 STR','Lodgy 110 PS RXZ Stepway [2015-2016]','Lodgy 110 PS RXZ 7 STR Stepway [2015-2016]','Lodgy 110 PS RXZ 7 STR [2015-2016]','Lodgy 110 PS RXZ [2015-2016]','Lodgy 110 PS RXL Stepway 8 STR','Lodgy 110 PS RXL 7 STR [2016]','Lodgy 85 PS STD 8 STR','Lodgy 85 PS World Edition 8 STR',
               'Megane 2.0 MT Petrol','Scala RXE Diesel','Scala RXE Petrol','Scala RXL Diesel','Scala RXL Diesel Travelogue','Scala RXL Petrol','Scala RXL Petrol AT','Scala RXl Petrol AT Travelogue','Scala RXL Petrol Travelogue','Scala RXZ Diesel','Scala RXZ Diesel Travelogue','Scala RXZ Petrol AT','Scala RXZ Petrol AT Travelogue','Scala Travelogue Edition',
               'Scenic 2'],
    'Volkswagen': ['1600 L Fastback','Ameo 1.0 Pace','Ameo Trendline 1.0L (P)','Ameo Highline 1.5L (D) [2016-2018]','Ameo Highline 1.2L Plus (P) 16 Alloy [2017-2018]','Ameo Highline 1.2L (P) [2016-2018]','Ameo Highline 1.0L (P)','Ameo Highline Plus 1.5L AT (D) 16 Alloy','Ameo Highline Plus 1.5L (D) 16 Alloy','Ameo Highline Plus 1.5L (D) Connect Edition','Ameo Highline Plus 1.0L (P) Connect Edition','Ameo Highline Plus 1.0L 16 Alloy','Ameo GT Line (D)','Ameo CUP Edition Petrol','Ameo CUP Edition Diesel','Ameo Comfortline Plus 1.5L AT (D)','Ameo Comfortline Plus 1.5 (D)','Ameo Comfortline Plus 1.2L (P)','Ameo Comfortline 1.5L (D)','Ameo Comfortline 1.2L (P)','Ameo Comfortline 1.0L (P)','Ameo Trendline 1.2L (P)','Ameo Trendline 1.5L (D)',
                  'Caravelle Highline','Caravelle T3','GTi 1.8 TSI','Passat Comfortline','Passet Comfortline Connect','Passet Highline','Passet Highline Connect',
                  'Polo 1.0 Pace','Polo Trendline 1.0L (P)','Polo Highline 1.5L (D)','Polo Highline 1.2L (P)','Polo Highline 1.0L (P)','Polo Highline Plus 1.5 (D) Connect Edition','Polo','Ameo Highline Plus 1.5L (D) 16 Alloy','Polo Highline Plus 1.2 (P) 16 Alloy [2017-2018]','Polo Highline Plus 1.0 (P) Connect Edition','Polo Highline Plus 1.0 (P) 16 Alloy','Polo GT TSI Sport','Polo GT TSI','Polo GT TDI Sport','Polo GT TDI [2016-2017]','Polo GT TDI','Polo CUP Edition Petrol','Polo CUP Edition Diesel','Polo Comfortline 1.5L (D)','Polo Comfortline 1.2L (P)','Polo Comfortline 1.0L (P)','Polo Allstar 1.5 (D)','Polo Allstar 1.2 (P)','Polo Trendline 1.2L (P)','Polo Trendline 1.5L (D)',
                  'Tiguan Comfortline TDI','Tiguan Highline TDI','Transporter T5','Vento Allstar 1.5 (D)','Vento Highline Petrol [2015-2016]','Vento Highline Petrol AT [2015-2016]','Vento Highline Plus 1.2 (P) AT 16 Alloy','Vento Highline Plus 1.5 (D) 16 Alloy','Vento Highline Plus 1.5 AT (D) 16 Alloy','Vento Highline Plus 1.6 (P) 16 Alloy','Vento Highline Diesel AT [2015-2016]','Vento Highline Plus Diesel [2015-2016]','Vento Preferred Edition Diesel [2016-2017]','Vento Preferred Edition Petrol AT [2016-2017]','Vento Trendline 1.5 (D)','Vento Trendline 1.6 (P)','Vento Highline Plus Petrol [2015-2016]','Vento Highline Diesel [2015-2016]','Vento Highline 1.6 (P) Connect Edition','Vento Highline 1.6 (P)','Vento Allstar 1.6 (P)','Vento Comfortline 1.2 (P) AT',
                   'Vento Comfortline 1.5 (D)','Vento Comfortline 1.5 (D) AT','Vento 1.6 (P)','Vento Comfortline Diesel [2015-2016]','Vento Comfortline Diesel AT [2015-2016]','Vento Comfortline Petrol [2015-2016]','Vento Comfortline Petrol AT [2015-2016]','Vento CUP Edition Diesel','Vento CUP Edition Petrol','Vento Highline 1.2 (P) AT','Vento Highline 1.5 (D)','Vento Highline 1.5 (D) AT','Vento Highline 1.5 (D) Connect Edition','Vento Trendline Diesel [2015-2016]','Vento Trendline Petrol [2015-2016]'],
    'MG': ['Hector','Astor','Comet EV','Gloster','ZS EV','Hector Plus'], 
    'Skoda': ['Octavia 1.4 TSI Ambition','Octavia 1.4 TSI Style','Octavia 1.8 TSI L&K','Octavia 1.8 TSI Style AT','Octavia 1.8 TSI Style Plus AT [2017]','Octavia 2.0 TDI CR Ambition','Octavia 2.0 TDI CR Style','Octavia 2.0 TDI CR Style AT','Octavia 2.0 TDI CR Style Plus AT [2017]','Octavia 2.0 TDI L&K','Octavia ONYX 1.8 TSI AT','Octavia ONYX 2.0 TDI AT','Octavia RS','Octavia RS 245',
             'Rapid Active 1.5 TDI','Rapid Style 1.5 TDI AT','Rapid Style 1.5 TDI','Rapid Rider Limited Edition','Rapid ONYX 1.6 MPI LE','Rapid ONYX 1.6 MPI AT LE','Rapid ONYX 1.5 TDI LE','Rapid ONYX 1.5 TDI AT LE','Rapid Monte Carlo 1.6 MPI MT [2017]','Rapid Monte Carlo 1.6 MPI MT','Rapid Monte Carlo 1.6 MPI AT [2017]','Rapid Monte Carlo 1.6 MPI AT','Rapid Style 1.6 MPI','Rapid Monte Carlo 1.5 TDI MT [2017]','Rapid Monte Carlo 1.5 TDI AT [2017]','Rapid Monte Carlo 1.5 TDI AT','Rapid Edition 1.6 MPI AT','Rapid Edition 1.6 MPI','Rapid Edition 1.5 TDI AT','Rapid Edition 1.5 TDI','Rapid Ambition 1.6 MPI AT','Rapid Ambition 1.6 MPI','Rapid Ambition 1.5 TDI AT','Rapid Ambition 1.5 TDI','Rapid Active 1.6 MPI','Rapid Monte Carlo 1.5 TDI MT','Rapid Style 1.6 MPI AT',
             'Superb Corporate Edition','Superb L&K TDI AT','Superb L&K TSI AT','Superb Sportline TDI AT','Superb Sportline TSI AT','Superb Style TDI AT','Superb Style TSI AT','Superb Style TSI MT',
             'Kodiaq L&K 2.0 TDI 4x4 AT','Kodiaq L&K 2.0 TSI 4X4','Kodiaq Scout','Kodiaq Style 2.0 TDI 4x4 AT'],
    'Nissan': ['300Z 3','350Z Convertible','350Z Coupe','Cedric 2.0 AT','Cima V8 Ltd','Gloria V6 Sedan','Granroad 4WD Diesel','GT-R Premium','GT-R Sport [2016-2021]','Jonga 4.0 Petrol','Lurel Diesel','Micra Fashion Edition','Micra X Shift [2015-2016]','Micra XE Diesel [2013-2016]','Micra XL (O) [2013-2016]','Micra XL (O) Diesel','Micra XL (O) Diesel [2013-2017]','Micra XL [2013-2016]','Micra XL CVT','Micra XL CVT [2015-2017]','Micra XL Diesel','Micra XL Diesel [2013-2017]','Micra XV CVT','Micra XV CVT [2016-2017]','Micra XV Diesel [2013-2016]','Micra XV P Diesel [2013-2016]','Micra XV P Diesel [2016]',
              'Murano V^ Petrol','Serena 2.0 L Diesel','Sunny Special Edition','Sunny XE','Sunny XE D','Sunny XL','Sunny XL CVT AT','Sunny XL D','Sunny XV CVT','Sunny XV D','Sunny XV Premium Pack (Leather)','Sunny XV Premium Pack (Safety)',
              'Terrano Sport Edition','Terrano XE (D)','Terrano XL (P)','Terrano XL O (D)','Terrano XV Premium AMT','Terrano XV Premium D'],
    'Ford': ['Aspire Ambiente 1.2 Ti-VCT','Aspire Trend 1.2 Ti-VCT [2014-2016]','Aspire Trend 1.2 Ti-VCT','Aspire Titanium 1.5 TDCi','Aspire Titanium Plus 1.5 TDCi','Aspire Titanium 1.5 Ti-VCT AT','Aspire Trend 1.5 TDCi','Aspire Titanium 1.5 TDCi Sports Edition','Aspire Titanium 1.2 Ti-VCT Sports Edition','Aspire Titanium 1.2 Ti-VCT Opt','Aspire Titanium 1.2 Ti-VCT','Aspire Ambiente 1.5 TDCi-ABS','Aspire Ambiente 1.5 TDCi','Aspire Ambiente 1.2 Ti-VCT ABS','Aspire Titanium 1.5 TDCi Opt','Aspire Trend 1.5 TDCi [2015-2016]',
            'Capri 1.6 Petrol','Ecosport Ambiente 1.5L TDCi','Ecosport Ambiente 1.5L Ti-VCT','Ecosport S Diesel','Ecosport S Petrol','Ecosport Signature Edition Diesel','Ecosport Signature Edition Petrol','Ecosport Titanium+ 1.5L TDCi','Ecosport Titanium+ 1.5L Ti-VCT','Ecosport Titanium+ 1.5L Ti-VCT AT','Ecosport Titanium 1.5L TDCi','Ecosport Titanium 1.5L Ti-VCT','Ecosport Trend+ 1.5L TDCi','Ecosport Trend+ 1.5 TDCi','EcoSport Trend+ 1.5L Ti-VCT AT','Ecosport Trend 1.5L TDCi','Ecosport Trend 1.5L Ti-VCT',
            'Endeavour Titanium 2.2 4x2 AT','Endeavour Titanium 2.2 4x2 AT [2016-2018]','Endeavour Titanium 3.2 4x4 AT','Endeavour Trend 2.2 4x2 AT','Endeavour Trend 2.2 4x2 MT','Endeavour Trend 2.2 4x4 MT','Endeavour Trend 3.2 4x4 AT',
            'Figo Ambiente 1.2 Ti-VCT','Figo Trend 1.5L TDCi [2015-2016]','Figo Trend 1.5L TDCi','Figo Trend 1.2 Ti-VCT [2015-2016]','Figo Trend 1.2 Ti-VCT','Figo Titanium 1.5 TDCi','Figo Titanium Plus 1.5 TDCi','Figo Titanium+ 1.2 Ti-VCT','Figo Titanium+ 1.5 Ti-VCT AT','Figo Trend+ 1.2 Ti-VCT','Figo Titanium 1.5 TDCi Sports Edition','Figo Titanium 1.2 Ti-VCT Sports Edition','Figo Titanium 1.2 Ti-VCT Opt','Figo Titanium 1.2 Ti-VCT','Figo Base 1.5 TDCi','Figo Base 1.2 Ti-VCT','Figo Ambiente 1.5 TDCi ABS','Figo Ambiente 1.5 TDCi','Figo Ambiente 1.2 Ti-VCT ABS','Figo Titanium 1.5 TDCi Opt','Figo Trend+ 1.5 TDCi',
            'Freestyle Ambiente 1.2 Ti-VCT','Freestyle Trend 1.5L TDCi [2018-2019]','Freestyle Trend 1.5 TDCi','Freestyle Trend 1.2 Ti-VCT [2018-2019]','Freestyle Trend 1.2 Ti-VCT','Freestyle Titanium+ 1.5 TDCi [2018-2019]','Freestyle Titanium Plus 1.5 TDCi','Freestyle Titanium+ 1.2 Ti-VCT [2018-2019]','Freestyle Trend+ 1.2 Ti-VCT [2019-2020]','Freestyle Titanium+ 1.2 Ti-VCT','Freestyle Titanium 1.5 TDCi','Freestyle Titanium 1.2 Ti-VCT [2018-2020]','Freestyle Titanium 1.2 Ti-VCT','Freestyle Flair Edition 1.5 TDCi','Freestyle Flair Edition 1.2 Ti-VCT','Freestyle Ambiente 1.5 TDCi','Freestyle Ambiente 1.2 Ti-VCT [2018-2020]','Freestyle Titanium 1.5 TDCi [2018-2020]','Freestyle Titanium 1.5 TDCi [2018-2020]','Freestyle Trend+ 1.5 TDCi [2019-2020]',
            'Model T Petrol','Mondeo Duratec HE','Mondeo Duratorq DI','Mondeo Ghia Duratec','Mustang GT Fastback 5.0L v8','Raptor F-150 4WD 5.0 Ti-VCT','Zephyr Petrol'],
    'Jeep': ['Compass','Meridian','Wrangler','Grand Cherokee','Avenger'],
    'Mercedes-Benz': ['G-Class 300','G-Class 500','G-Class 500 Grand Edition','G-Class 55 AMG','G-Class 55 AMG Grand Edition','G-Class G 63 AMG Crazy Colour Edition','G-Class G 63 AMG','G-Class G 63 Edition 463',
                     'C-Class C 200 Avantgarde','C-Class C 200 Avantgarde Edition','C-Class C 220 CDI Avantgarde','C-Class C 220 CDI Style','C-Class C 250 d','C-Class C 43 AMG','C-Class C 63 S AMG',
                     'E-Class 200 Avantgarde','E-Class E 200 Exclusive','E-Class E 200 Exclusive [2019-2019]','E-Class E 200 Expression','E-Class E 200 Expression [2019-2019]','E-Class E 200 Expression [2019-2020]','E-Class E 220d Avantgarde','E-Class E 220d Exclusive','E-Class E 220d Exclusive [2019-2019]','E-Class E 220d Expression','E-Class E 220d Expression [2019-2019]','E-Class E 350d [2019-2020]','E-Class E 350d Exclusive [2017-2019]','E-Class E 350d Elite','E-Class E 63 AMG S 4MATIC+','E-Class E 63 AMG S 4MATIC+ [2018-2019]',
                     'C-Class Cabriolet C 300','S-Class Maybach S 500','S-Class Maybach S 560','S-Class Maybach S 600','S-Class Maybach S 600 Guard','S-Class Maybach S 650','S-Class S 350 CDI','S-Class S 350 CDI Connoisseurs Edition','S-Class S 400','S-Class S 400 Connoisseurs Edition','S-Class S 500','S-Class S 63 AMG','S-Class S Guard',
                     'GLE Coupe 43 4MATIC [2017-2019]','GLE Coupe 43 AMG 4MATIC','GLE Coupe 43 AMG 4MATIC 2016','GLE Coupe 450 AMG','GLE Coupe 53 AMG 4MATIC Plus','AMG GT Coupe',
                     'CLS 250 CDI','CLS 350','S-Coupe S 500','S-Coupe S 63 AMG','S-Coupe 63 AMG [2015-2018]','S-Coupe S 63 AMG [2018-2019]','S-Coupe S 63 AMG 4MATIC',
                     'CLA 200 CDI Sport','CLA 200 CDI Style','CLA 200 D Urban Sport','CLA 200 Petrol Sport','CLA 200 Urban Sport','CLA 45 Aero Edition','CLA 45 AMG 4MATIC [2017-2017]','CLA 45 AMG 4MATIC [2017-2019]',
                     'GLA 200d Sport','GLA 200d Style','GLA 200 Sport','GLA 200 Urban Edition','GLA 200d Urban Edition','GLA 220d 4MATIC','GLA 220d Urban Edition','GLA 45 Aero Edition','GLA 45 AMG 4MATIC',
                     'E-Class All-Terrain E 220d [2018-2019]','E-Class All-Terrain E 220d 4MATIC','GLS 350d','GLS 400 4MATIC','GLS 63 AMG','GLS Grand Edition Diesel','GLS Grand Edition Petrol',
                     'A-Class A 180','A-Class A 180 Night Edition','A-Class A 200d','A-Class A 200d Night Edition','M-Class M Guard','M-Class ML 250 CDI','M-Class ML 350 CDI','M-Class ML 63 AMG',
                     'GLE 250d','GLE 350d','GLE 400 4MATIC','B-Class B 180 Sport','B-Class B 180 Night Edition','B-Class B 200 Sport CDI','B-Class B 200 Night Edition',
                     'GLC Coupe 43 AMG','GLC Coupe 43 AMG [2017-2019]','GLC 220d CBU','GLC 220d Prime','GLC 20d Progressive','GLC 220d Sport','GLC 300 CBU','GLC 300 Progressive','GLC Celebration Edition Diesel','GLC Celebration Edition Petrol',
                     'E-Class Cabriolet E 400 Cabriolet','R-Class R350 4MATIC','R-Class R350 CDI 4MATIC','SLC 43 AMG','S-Class Cabriolet S 500 Cabriolet','Viano AT','Viano MT',
                     'CLK 320 Cabriolet','CLK 320 Coupe','CLK 350 AMG','CLK 350 Cabriolet','CLK 500 Cabriolet','CLK 500 Coupe','CLK 55 AMG Cabriolet','CLK 55 AMG Coupe',
                     'W123 300 D','MB-Class 2.9 D','190 D','190 W110','SDL 3.5','W110 Diesel','W110 Petrol'],
    'BMW': ['5-Series 523i Sedan','5-Series 523i Touring','5-Series 525 tds','5-Series 525d Touring','5-Series 525i','5-Series 528i Sedan','5-Series 530d Touring','5-Series 530i Touring','5-Series 535d Sedan','5-Series 535d Touring','5-Series 540i Sedan','5-Series 550i Sedan','5-Series 550i Touring',
           'X3 SAV 2.0i','X3 SAV 2.5i','6-Series GT 620d Luxury Line','6-Series GT 620d Luxury Line [2019-2019]','6-Series GT 630d Luxury Line','6-Series GT 630d Luxury Line [2018-2019]','6-Series GT 630d M Sport','6-Series GT 630d M Sport [2018-2019]','6-Series GT 630i Luxury Line','6-Series GT 630i Luxury Line [2018-2019]','6-Series GT 630i Sport Line',
           '3-Series GT 320d Luxury Line','3-Series GT 320d Sport','3-Series GT 320d Sport Line','3-Series GT 330i Luxury Line','3-Series GT 330i M Sport','3-Series GT 330i M Sport [2017-2019]','3-Series GT 330i M Sport Shadow Edition',
           'M5 4.4 V8','M2 Competition','M2 Competition [2018-2019]','M4 Coupe','M3 Sedan','i8 1.5 Hybrid','X1 sDrive20d Expedition','X1 sDrive20d M Sport','X1 sDrive20d xLine','X1 sDrive20i xLine','X1 sDrive20i xLine [2018-2019]','X1 sDrive20i xLine [2017-2018]','X1 xDrive20d M Sport','X1 xDrive20d xLine',
           '3-Series 316Ci Coupe','3-Series 330d Convertible','3-Series 330Ci Coupe','3-Series 330Ci Convertible','3-Series 330Cd Coupe','3-Series 330Cd Convertible','3-Series 325 tds','3-Series 325i Touring','3-Series 325Ci Coupe','3-Series 325Ci Convertible','3-Series 320si Sedan','3-Series 320i Touring','3-Series 320d Touring','3-Series 320Ci Coupe','3-Series 320Ci Convertible','3-Series 320Cd Coupe','3-Series 320Cd Convertible','3-Series 318i Touring','3-Series 318i Sedan','3-Series 318d Touring','3-Series 318d Sedan','3-Series 318Ci Coupe','3-Series 318Ci Convertible','3-Series 316i Sedan','3-Series 330d Touring','3-Series 330i Touring',
           '6-Series 645i','M6 Convertible','M6 Coupe','X5 SAV 3.0i','X5 SAV 4.4i','X5 SAV 4.8is','Z4 M 40i','Z4 sDrive 35i','Z5 sDrive 35i DPT',
           '7-Series 728i','7-Series 730d Sedan','7-Series 730i Sedan','7-Series 730Li Sedan','7-Series 735Li','7-Series 740i Sedan','7-Series 745d Sedan','7-Series 750i Sedan','7-Series 750Li','7-Series 760i Sedan',
           'X6 35i M Sport','X6 M Coupe','X6 xDrive40d M Sport','X6 xDrive40i M Sport','Z3 Roadster'], 
    'Audi': ['Q3 30 TDI Premium FWD','Q3 30 TFSI Premium','Q3 35 TDI quattro Premium Plus','Q3 35 TDI quattro Technology','Q7 40 TFSI Premium Plus','Q7 45 TDI Black Styling','Q7 45 TDI Premium Plus','Q7 45 TDI Technology Pack', 'Q7 45 TFSI Black Styling','Q7 45 TFSI Premium Plus','Q7 45 TFSI Technology Pack','Q7 Lifestyle Edition',
            'A4 1.4 TFSI Multitronic','A4 30 TFSI Premium Plus','A4 30 TFSI Quick Lift Premium Plus','A4 30 TFSI Quick Lift Technology Pack', 'A4 30 TFSI Technology Pack','A4 35 TDI Premium Plus','A4 35 TDI Technology','A4 35 TFSI','A4 Lifestyle Edition',
            'A3 35 TFSI','A3 40 TFSI','Q5 2.0 TDI quattro Technology Pack','Q5 2.0 TFSI quattro Premium','Q5 2.0 TFSI quattro Premium Plus','Q5 2.0 TFSI quattro Technology Pack', 'Q5 3.0 TDI quattro Premium Plus','Q5 3.0 TDI quattro Technology Pack','Q5 30 TDI Design Edition','Q5 30 TDI Premium Edition','Q5 30 TDI Sports Edition','Q5 45 TDI Technology S Line',
            'R8 Coupe 4.2 FSI quattro','R8 Coupe 4.2 FSI quattro R tronic','A5 S5','A5 Sportback 35 TDI','A5 Sportback 40 TDI','A5 Cabriolet 2.0 TDI','TT 45 TFSI','RS5 Coupe','RS Avant','A8 3.0 TDI quattro','A8 4.0 FSI quattro','A8 4.2 FSI quattro','A8 4.2 TDI quattro','A8 50 TDI','A8 50 TDI Plus','A8 60 TDI','A8 60 TFSI','A8 W12',
            'A6 2.5 TDI'],
    'Jaguar': ['F-Pace First Edition','F-Pace Prestige','F-Pace Prestige [2016-2017]','F-Pace Prestige Petrol','F-Pace Prestige Petrol [2018-2020]','F-Pace Pure','F-Pace R-Sport','F-Pace R-Sport Petrol','F-type 2.0 Convertible','F-type 2.0 Convertible R Dynamic','F-type 2.0 Coupe','F-type 2.0 Coupe R Dynamic','F-type Coupe','F-type R Convertible','F-type R Coupe','F-type S Convertible','F-type S Coupe','F-type SVR Convertible','F-type SVR Coupe','XE Portfolio','XE Portfolio Diesel','XE Prestige','XE Prestige Diesel','XE Pure','XE Pure Diesel','XF 2.7 V6','XJ 2.0 Portfolio','XJ 3.0 Portfolio','XJ 3.0 Premium Luxury','XJ 3.0 Premium Luxury [2016-2018]','XJ 5.0 V8 Autobiography','XJ 50'],
    'Fiat': ['1100 Petrol','Millicento Petrol','500 Competizione','500 Competizione [2015-2016]','Linea 1.3L Multijet Classic','Linea 1.3L Multijet Classic+','Linea 1.4L P Classic','Punto 1.2 Petrol','Punto 1.3 Diesel','Avventura Active 1.4','Avventura Active Multijet 1.3','Avventura Dynamic 1.4','Avventura Dynamic Multijet 1.3','Avventura Emotion Multijet 1.3','Avventura T-Jet 1.4','Punto EVO T-Jet 1.4 Abarth','Siena EL 1.6','Siena EL 1.7 PS','Siena ELX 1.6','Siena ELX 1.7 PS','Siena EX 1.2','Siena Maestro 1.6','Siena Maestro 1.6 SP','Urban Cross Active Multijet 1.3','Urban Cross Dynamic Multijet 1.3','Urban Cross Emotion Multijet 1.3','Urban Cross Emotion T-Jet 1.4'],
    'Datsun': ['Go A [2014-2017]','Go A EPS','Go Anniversary Edition','Go D','Go D1','Go NXT','Go Remix Edition','Go Style Edition','Go T','Go T (O)',
              'Go Plus A [2014-2017]','Go Plus A EPS','Go Plus Anniversary Edition','Go Plus D','Go Plus D1','Go Plus Remix Edition','Go Plus Style Edition','Go Plus T','Go Plus T (O)','Redigo A','Redigo T (O) 1.0 AMT','Redigo T (O) [2017-2019]','Redigo T (O) 1.0','Redigo T (O) 0.8L Limited Edition','Redigo T [2019-2019]','Redigo T [2016-2019]','Redigo T (O) [2016-2019]','Redigo T (O)','Redigo T','Redigo Sport','Redigo S 1.0 AMT [2019-2019]','Redigo S 1.0 AMT [2018-2019]','Redigo S 1.0 [2017-2019]','Redigo S 1.0','Redigo S [2019-2019]','Redigo S [2016-2019]','Redigo S','Redigo Gold Limited Edition','Redigo D [2019-2019]','Redigo D [2016-2019]','Redigo D','Redigo A [2019-2019]','Redigo A [2016-2019]','Redigo T (O) 1.0 AMT [2018-2019]','Redigo T (O)L Limited Edition'],
    'Mini': ['Clubman Cooper S','Clubman Indian Summer Red Edition','Convertible 2.0','Cooper D3 Door','Cooper D5 Door','Cooper S',
            'Countryman Cooper S JCW Inspired','Countryman Cooper S','Countryman Cooper S [2018-2020]','Countryman Cooper S [2020-2021]','Countryman Cooper S JCW Inspired [2018-2020]','Countryman Cooper S JCW Inspired [2020-2021]','Countryman Cooper S JCW Inspired Black Edition','Countryman Cooper SD','Countryman Shadow Edition'],
    'Mitsubishi': ['Challenger 2.8 Diesel','FTO Sports Coupe','Galant 2.0 AT','Galant GL','GTO Coupe','Montero 3.2 Di-D AT','Outlander 4x4','Pajero 2.5 AT','Pajero 2.5 MT','Pajero Limited Edition','Pajero Select Plus AT','Pajero Select Plus MT'],
    'Volvo': ['S60 Cross Country Inscription','S60 Cross Country Inscription [2016-2020]','S60 D4 R','S60 D5 2.4L','S60 Inscription','S60 Kinetic','S60 Momentum','S60 Momentum [2015-2020]','S60 Polestar','S60 Polestar [2017-2020]','S60 T6',
             'S90 D4 Inscription','S90 Cross Inscription D4 [2016-2020]','S90 Momentum D4','S90 Momentum D4 [2018-2020]',
             'V40 D3 Kinetic','V40 D3 R-Design','V40 Cross Country D3 Inscription','V40 Cross Country T4 Momentum','V70 2.0 Petrol I5','V90 Cross Country D5 Inscription','V90 Cross Country D5 Inscription [2017-2020]','XC40 D4 R-Design','XC40 Inscription','XC40 Momentum','XC40 T4 R-Design','XC40 T4 R-Design [2019-2020]',
             'XC60 D5 Inscription','XC60 Inscription [2017-2020]','XC60 Momentum','XC60 Momentum [2019-2020]','XC90 D5 Inscription','XC90 D5 Momentum','XC90 Excellence [2016-2020]','XC90 Excellence Lounge','XC90 Excellence Lounge [2019-2020]','XC90 Inscription Luxury [2015-2020]','XC90 Momentum Luxury [2015-2020]','XC90 R-Design','XC90 Recharge','XC90 T8 Excellence (4 STR)','XC90 T8 Inscription','XC90 T8 Inscription (7 STR)'],
    'Force': ['Gurkha'],
    'Isuzu': ['D-Max V-Cross High','D-Max V-Cross Standard','MU-X 4x2','MU-X 4x4','MU7 Base BS-III','MU7 Base BS-IV','MU7 High BS-III','MU7 High BS-IV','MU7 Premium AT','Trooper 4x4']
}

    Brand = st.selectbox('Select the Car Brand',['Select'] + list(car_brands_and_models.keys()))
    if Brand and Brand != 'Select':
        Model = car_brands_and_models[Brand]
        selected_model = st.selectbox('Select the Car Model',['Select'] + Model)
    
    year_options = list(range(2000, 2025))
    p1 = st.selectbox('In which year car was manufactured ?',['Select'] + year_options)
    p2 = st.number_input('What is distance covered by the car in Kilometers ?', 0, step=100)
    s1 = st.selectbox('What is the fuel type of the car ?', ('Select','CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'))
    p3 = {'CNG': 0, 'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'Electric': 4}.get(s1, -1)

    s2 = st.selectbox('What is the Transmission Type ?', ('Select','Manual', 'Automatic'))
    p4 = {'Manual': 0, 'Automatic': 1}.get(s2, -1)

    s3 = st.selectbox('Number of Owners the car previously had', ('Select','First', 'Second', 'Third', 'Fourth & Above'))
    p5 = {'First': 0, 'Second': 1, 'Third': 2, 'Fourth & Above': 3}.get(s3, -1)

    p6 = st.number_input('What is the mileage of the car in Km/l or equivalent ?', 0.0,step=0.10)
    p7 = st.number_input('What is the power of the car in bhp ?', 0, step=1)
    p8 = st.number_input('What is the engine capacity of the car in CC ?', 0, step=1)
    p9 = st.number_input('What is the number of passenger seats in the car ?', 0, 20, step=1)
    p10 = st.number_input('What is the current/ex-showroom price of the car ? (In Lakhs)', 0.0, step=0.1)

    data_new = pd.DataFrame({
        'Year': p1,
        'Kilometers_Driven': p2,
        'Fuel_Type': p3,
        'Transmission': p4,
        'Owner_Type': p5,
        'Mileage': p6,
        'Power': p7,
        'Engine': p8,
        'Seats': p9,
        'New_Price': p10
    }, index=[0])

    if st.button('Predict'):
        if (p1 == 0 or p2 == 0 or p6 == 0 or p7 == 0 or p8 == 0 or p9 == 0 or p10 == 0):
            st.warning("All fields are mandatory and cannot be zero.")
        else:
            try:
                xgb_predictions = xgb_model.predict(data_new)
                gb_predictions = gb_model.predict(data_new)
                final_prediction = (xgb_predictions * weight3) + (gb_predictions * weight4)
                final_prediction = np.clip(final_prediction, 0, p10)
                
                if final_prediction > 0:
                    st.balloons()
                    st.success(f'You can sell the car for {final_prediction[0]:.2f} lakhs')
                else:
                    st.warning("You will not be able to sell this car!!")
            except Exception as e:
                st.error(f"Oops!! Something went wrong. Try again. Error: {e}")

def show_help_and_support():
    remove_bg()
    st.title("Help & Support")
    st.markdown('<h3 style="color:black; font-size:15px;">Here you can find help and support for using the app</h3>', unsafe_allow_html=True)

    # Add FAQs
    st.subheader("Frequently Asked Questions")
    faq_data = {
        "What does this app do?": "This app helps you predict the resale price of your car based on various features and inputs.",
        "How do I use the app?": "Navigate to the Home section, fill in the required details about your car, and click 'Predict' to get the estimated resale value.",
        "How accurate are the predictions?": "The predictions are based on hybrid machine learning models trained on historical data. While they provide a good estimate, actual market conditions may affect the final resale value.",
        "Can I trust the results?": "The results are generated using advanced models and algorithms, but it's always a good idea to cross-check with market trends and professional appraisers.",
        "How is my data used?": "Your data is used only for the purpose of generating predictions and is not stored or shared with third parties.",
        "Are there any limitations to the predictions?": "Predictions may not account for all variables such as local market conditions, the condition of the car, or unique features.",
        "What should I do if the app is not working correctly?": "Ensure all required fields are filled out correctly and check for any error messages. If the issue persists, contact support.",
        "I’m seeing an error message, what should I do?": "Check the error message for details. If you need further assistance, please contact our support team with the error information.",
        "Why are some options not available in the dropdown menus?": "Options may be limited to those relevant to the models and data available. If you believe an option is missing, please let us know.",
        "Who can I contact for support?": "You can reach us at rayyanpp502@gmail.com."
    }

    for question, answer in faq_data.items():
        st.write(f"**{question}**")
        st.write(answer)

    # Add a contact form
    st.subheader("Contact Us")
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if save_to_file(name, email, message):
                st.success("Thank you for your message! We have saved your contact details.")
            else:
                st.error("There was an error saving your message. Please try again later.")

    # Add links to external documentation or resources
    #st.subheader("Additional Resources")
    #st.write("For more detailed documentation, please visit our [Documentation Page](https://example.com/docs).")

def add_sidebar_styles():
    st.markdown(
        """
        <style>
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background-color: #4A6C6F; /* Deep Teal */
            padding-top: 20px; /* Padding at the top */
        }
        /* Sidebar title and widget headers */
        .css-1d391kg { /* This class selector may vary; check actual class in your browser */
            color: #E0FFFF; /* Light Cyan for titles */
            font-weight: bold; /* Bold text */
            font-size: 1.5rem; /* Increase font size */
        }
        /* Sidebar selectbox styling */
        .css-1cpxqw2 {
            background-color: #0A0A0A; /* Light pastel cyan */
            color: #FFFFFF; /* Matching Deep Teal text */
            border-radius: 10px; /* Rounded corners */
            padding: 8px; /* Padding for selectbox */
            border: 2px solid #20B2AA; /* Light Sea Green border */
        }
        /* Sidebar items on hover */
        .css-1cpxqw2:hover {
            background-color: #20B2AA; /* Light Sea Green on hover */
            color: #014D4E; /* Deep Teal text on hover */
        }
        /* Sidebar section titles */
        .css-1d391kg h1 {
            font-family: 'Roboto', sans-serif; /* Font family */
            color: #FFFFFF; /* Light Cyan for text */
            margin-bottom: 10px; /* Space below */
        }
        /* Sidebar selectbox active item */
        .css-1cpxqw2:focus {
            background-color: #2E8B57; /* Sea Green for focused items */
            color: #FFFFFF; /* White text on focus */
            border: 2px solid #20B2AA; /* Light Sea Green border on focus */
        }
        /* NAVIGATION text color */
        [data-testid="stSidebar"] .element-container:first-child .css-1d391kg {
            color: yellow !important; /* Yellow color for NAVIGATION text */
        }
        /* Select the car brand text color */
        [data-testid="stSidebar"] .element-container:nth-child(2) .css-1d391kg {
            color: red !important; /* Red color for "Select the car brand" text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    add_sidebar_styles()
    # Create navigation options
    st.sidebar.title("Navigation")
    menu = ["Home", "Help & Support"]
    choice = st.sidebar.selectbox("Select a page", menu)

    # Show Home page content
    if choice == "Home":
        show_home_page()

    # Show Help & Support page content
    elif choice == "Help & Support":
        show_help_and_support()
    footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
            font-size: small;
            z-index: 9999;
        }
        </style>
        <div class="footer">
            <p>Copyright © &copy; 2024 | All Rights Reserved</p>
        </div>
        """
    st.markdown(footer, unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()
