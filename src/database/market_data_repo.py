import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://root:root@localhost:3306/stock", pool_pre_ping=True)

def retrieve_by_ticker(ticker):
    query = text("""
        SELECT * FROM tbl_stock_market_data WHERE ticker = :ticker ORDER BY `Date` ASC
    """)
    df = pd.read_sql(query, con=engine, params={'ticker': ticker})
    return df