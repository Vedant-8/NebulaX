import json
import sqlite3

def save_to_json(filepath, data):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_from_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_to_sqlite(db_path, table_name, data):
    """
    Save data to an SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to save data.
        data (dict): Data to save.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Create table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            experiment_name TEXT,
            description TEXT,
            timestamp TEXT,
            parameters TEXT,
            metrics TEXT
        )
    """)

    # Insert data
    cursor.execute(f"""
        INSERT INTO {table_name} (experiment_name, description, timestamp, parameters, metrics)
        VALUES (?, ?, ?, ?, ?)
    """, (
        data["name"],
        data["description"],
        data["timestamp"],
        json.dumps(data["parameters"]),
        json.dumps(data["metrics"]),
    ))

    connection.commit()
    connection.close()

def load_from_sqlite(db_path, table_name, experiment_name):
    """
    Load data from an SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to query data from.
        experiment_name (str): Name of the experiment to load.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Query the table
    cursor.execute(f"""
        SELECT description, timestamp, parameters, metrics
        FROM {table_name}
        WHERE experiment_name = ?
    """, (experiment_name,))

    row = cursor.fetchone()
    connection.close()

    if row:
        return {
            "name": experiment_name,
            "description": row[0],
            "timestamp": row[1],
            "parameters": json.loads(row[2]),
            "metrics": json.loads(row[3]),
        }
    else:
        raise ValueError(f"No experiment found with name: {experiment_name}")
