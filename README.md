This project is a **Streamlit-based dashboard** for analyzing Amazon India sales data.  
It connects to a MySQL database (`amazon_sales_analytics`) and provides **30 business questions** answered through interactive charts and metrics.

Setup Instructions

1. Clone Repository
git clone https://github.com/MathanRajh/amazon_project.git
cd amazon_projec

2. Install the dependencies:

pip install -r requirements.txt
download datasets https://drive.google.com/drive/folders/1ZHB4x8nZHuXmyDlwujWtbOaxiMHWf-3-


3. Set up the MySQL database:

Create a database named amazon_sales_analytics

Import your transactions, customers, and products tables

Update database credentials inside streamlit_dashboard_30q.py:

DB_USER = "root"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_NAME = "amazon_sales_analytics"


4. Run the Streamlit dashboard:

streamlit run streamlit app.py

Open the provided local URL in your browser to explore the dashboard.
