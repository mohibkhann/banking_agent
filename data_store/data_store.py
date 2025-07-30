import pandas as pd

# GLOBAL DATA STORE - Deals with the retrieval of Banking Data

class DataStore:
    """Global data store for tools to access banking data"""
    _instance = None
    _banking_data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_data(self, banking_data_path: str):
        """Load banking data"""
        #"C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv"
        self._banking_data = pd.read_csv(banking_data_path)
        self._banking_data['date'] = pd.to_datetime(self._banking_data['date'])
    
    def get_client_data(self, client_id: int) -> pd.DataFrame:
        """Get filtered data for specific client"""
        if self._banking_data is None:
            raise ValueError("Banking data not loaded")
        return self._banking_data[self._banking_data['client_id'] == client_id].copy()
    

if __name__=="__main__":
    testing_class = DataStore()
    print("Loading")
    testing_class.load_data("C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv")
    print("this is client 440 data")
    print(testing_class.get_client_data(430).head())
    print(testing_class._banking_data.head())

    print("The data has been loaded successfully") 