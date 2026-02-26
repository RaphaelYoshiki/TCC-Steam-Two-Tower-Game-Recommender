import pandas as pd

DF_PATH = "./csv_files/treated_dataframe.csv"

class GameLookup:
    def __init__(self):
        df = pd.read_csv(DF_PATH)
        self.id_to_name = {
            int(row["game_id"]): row["name"]
            for _, row in df.iterrows()
        }

    def resolve(self, appids):
        return [
            {
                "steamid": int(appid),
                "name": self.id_to_name.get(int(appid), "Unknown Game")
            }
            for appid in appids
        ]
