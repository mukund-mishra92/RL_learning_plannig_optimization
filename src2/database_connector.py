import mysql.connector
import pandas as pd

def connect_to_warehouse_database(host, user, password, database):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return conn

def load_location_data(conn, table_name="location_master"):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

def create_mock_location_data(num_locations=100):
    import numpy as np
    data = {
        'location_id': range(num_locations),
        'x': np.random.randint(0, 100, num_locations),
        'y': np.random.randint(0, 100, num_locations),
        'z': np.random.randint(0, 10, num_locations)
    }
    return pd.DataFrame(data)
