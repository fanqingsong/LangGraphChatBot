


import os
import shutil
import sqlite3
import pandas as pd
import requests


class SQLiteTool:
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    local_file = "travel2.sqlite"
    backup_file = "travel2.backup.sqlite"

    def __init__(self):
        self.conn = sqlite3.connect(self.local_file)
        self.cursor = self.conn.cursor()

    def execute_query(self, query):
        try:
            result = self.cursor.execute(query).fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(result, columns=columns)
        except sqlite3.Error as e:
            return f"Error: {e}"

    def close(self):
        self.conn.close()

    def load_data(self):
        overwrite = False
        if overwrite or not os.path.exists(self.local_file):
            response = requests.get(self.db_url)
            response.raise_for_status()  # Ensure the request was successful
            with open(self.local_file, "wb") as f:
                f.write(response.content)
            # Backup - we will use this to "reset" our DB in each section
            shutil.copy(self.local_file, self.backup_file)

    # Convert the flights to present time for our tutorial
    def update_dates(self):
        if not os.path.exists(self.local_file):
            shutil.copy(self.backup_file, self.local_file)
            print("Backup file copied to local file.")

        conn = sqlite3.connect(self.local_file)
        cursor = conn.cursor()

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        ).name.tolist()
        tdf = {}
        for t in tables:
            tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

        example_time = pd.to_datetime(
            tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
        ).max()
        current_time = pd.to_datetime("now").tz_localize(example_time.tz)
        time_diff = current_time - example_time

        tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
            + time_diff
        )

        datetime_columns = [
            "scheduled_departure",
            "scheduled_arrival",
            "actual_departure",
            "actual_arrival",
        ]
        for column in datetime_columns:
            tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
            )

        for table_name, df in tdf.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        del df
        del tdf
        conn.commit()
        conn.close()

