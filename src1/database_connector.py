# 🔌 Database Connection and Data Loading Module
"""
Database connectivity and data loading functions for RL Path Planning System
"""

import numpy as np
import pandas as pd
import mysql.connector
import getpass
import random

def connect_to_warehouse_database():
    """
    Connect to MySQL database and load warehouse location data
    
    Returns:
        pandas.DataFrame: Location master table with warehouse layout
    """
    try:
        print("🔑 Connecting to MySQL database...")
        
        # Get password securely
        password = getpass.getpass("Enter MySQL password for root: ")
        
        # Connect to MySQL server with specific database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=password,
            database='neo-sim-noon-minutes'
        )
        
        if connection.is_connected():
            print("✅ Successfully connected to MySQL database")
            
            # Load location_master table
            table_name = 'location_master'
            query = f"SELECT * FROM {table_name}"
            
            df = pd.read_sql(query, connection)
            
            print(f"📊 Loaded table '{table_name}' successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display sample data
            print(f"\n👀 Sample Data:")
            print(df.head())
            
            # Basic statistics
            print(f"\n📈 Basic Statistics:")
            print(df.describe())
            
            return df
            
    except mysql.connector.Error as error:
        print(f"❌ Failed to connect to MySQL: {error}")
        print("🔄 Creating mock dataset for demonstration...")
        return create_mock_location_data()
        
    finally:
        # Close connection
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("🔐 MySQL connection closed")

def create_mock_location_data():
    """
    Create mock warehouse data for testing when database is not available
    
    Returns:
        pandas.DataFrame: Mock location master data
    """
    print("🏭 Creating mock warehouse layout...")
    
    # Create a 10x10 warehouse grid
    locations = []
    location_id = 1
    
    for y in range(10):
        for x in range(10):
            # Define movement possibilities based on grid position
            xp = 1 if x < 9 else 0  # Can move right unless at right edge
            xn = 1 if x > 0 else 0  # Can move left unless at left edge
            yp = 1 if y < 9 else 0  # Can move up unless at top
            yn = 1 if y > 0 else 0  # Can move down unless at bottom
            
            # Add some obstacles (random blocked cells)
            if random.random() < 0.1:  # 10% obstacles
                location_type = 'OBSTACLE'
                xp = xn = yp = yn = 0
            else:
                location_type = 'PATH'
            
            locations.append({
                'location_id': location_id,
                'x': x,
                'y': y,
                'z': 0,
                'type': location_type,
                'XP': xp,
                'XN': xn,
                'YP': yp,
                'YN': yn,
                'ZP': 0,  # No vertical movement in this example
                'ZN': 0
            })
            location_id += 1
    
    df = pd.DataFrame(locations)
    print(f"✅ Mock warehouse created: {len(df)} locations")
    print(f"📊 Layout: 10x10 grid with {df[df['type'] == 'OBSTACLE'].shape[0]} obstacles")
    
    return df

def load_location_data():
    """
    Main function to load warehouse data (attempts database, falls back to mock)
    
    Returns:
        pandas.DataFrame: Location data for path planning
    """
    print("🏗️ Loading warehouse data...")
    try:
        df = connect_to_warehouse_database()
    except Exception as e:
        print(f"⚠️ Database loading failed: {e}")
        print("🔄 Using mock data instead...")
        df = create_mock_location_data()
    
    print(f"\n🎯 Warehouse data loaded successfully!")
    print(f"Ready for environment creation with {len(df)} locations")
    return df