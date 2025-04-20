import os
import unittest
from nebulaPy.storage import save_to_json, load_from_json, save_to_sqlite, load_from_sqlite

class TestStorage(unittest.TestCase):
    def setUp(self):
        """Set up test data and temporary files."""
        self.test_data = {
            "name": "Test Experiment",
            "description": "Testing storage backends",
            "timestamp": "2025-04-20T12:00:00",
            "parameters": {"learning_rate": 0.001},
            "metrics": {"accuracy": 0.95},
        }
        self.json_file = "test_experiment.json"
        self.db_file = "test_experiment.db"
        self.table_name = "experiments"

    def tearDown(self):
        """Clean up temporary files after each test."""
        if os.path.exists(self.json_file):
            os.remove(self.json_file)
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

    def test_save_and_load_json(self):
        """Test saving and loading data to/from JSON."""
        save_to_json(self.json_file, self.test_data)
        loaded_data = load_from_json(self.json_file)
        
        self.assertEqual(self.test_data, loaded_data)

    def test_save_and_load_sqlite(self):
        """Test saving and loading data to/from SQLite."""
        save_to_sqlite(self.db_file, self.table_name, self.test_data)
        loaded_data = load_from_sqlite(self.db_file, self.table_name, self.test_data["name"])

        self.assertEqual(self.test_data["name"], loaded_data["name"])
        self.assertEqual(self.test_data["description"], loaded_data["description"])
        self.assertEqual(self.test_data["timestamp"], loaded_data["timestamp"])
        self.assertEqual(self.test_data["parameters"], loaded_data["parameters"])
        self.assertEqual(self.test_data["metrics"], loaded_data["metrics"])

if __name__ == "__main__":
    unittest.main()
