import requests
import pandas as pd
from datetime import datetime
import dotenv
import os

dotenv.load_dotenv('.env.local')

class FREDClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/"

    def get_series_data(self, series_id, start_date=None, end_date=None):
        """
        Descarga datos de una serie específica de FRED
        
        Args:
            series_id (str): ID de la serie de FRED (ej: 'GDP', 'UNRATE')
            start_date (str, opcional): Fecha inicial en formato 'YYYY-MM-DD'
            end_date (str, opcional): Fecha final en formato 'YYYY-MM-DD'
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date if start_date else '1776-07-04',
            'observation_end': end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        }
        
        response = requests.get(
            f"{self.base_url}observations",
            params=params
        )
        
        if response.status_code == 200:
            datas = response.json()
            df = pd.DataFrame(datas['observations'])[['date', 'value']]
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        else:
            raise Exception(f"Error al obtener datos: {response.status_code}")

    def download_multiple_series(self, series_ids, start_date=None, end_date=None):
        """
        Descarga múltiples series y las combina en un único DataFrame
        
        Args:
            series_ids (list): Lista de IDs de series
            start_date (str, opcional): Fecha inicial
            end_date (str, opcional): Fecha final
        """
        dfs = {}
        for series_id in series_ids:
            try:
                df = self.get_series_data(series_id, start_date, end_date)
                dfs[series_id] = df
            except Exception as e:
                print(f"Error al descargar {series_id}: {str(e)}")
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(dfs, axis=1)
        combined_df.columns = combined_df.columns.droplevel(1)  # Eliminar el nivel 'value'
        return combined_df
    
# Ejemplo de uso
if __name__ == "__main__":
    fred = FREDClient(os.getenv('FRED_API_KEY'))
    
    # Lista de series a descargar
    series_ids = [
            'JHDUSRGDPBR', 'T10Y3M', 'BAMLC0A0CM', 'BAMLH0A0HYM2',
            'BAMLHE00EHYIOAS', 'UMCSENT', 'UNRATE', 'GDPC1', 'DCOILWTICO',
            'CORESTICKM159SFRBATL', 'USSLIND', 'VIXCLS', 'ICSA',
            'MARTSMPCSM44000USS', 'RSXFS', 'TREAST', 'DGS1', 
            ##########################
            "TOTBKCR","BUSLOANS","WPU08", "PCOTTINDUSDM", "PWHEAMTUSDM", "PMAIZMTUSDM", "PCOFFOTMUSDM", 
            "PNRGINDEXM", "PCOPPUSDM", "PNGASEUUSDM", "PALLFNFINDEXQ", "BUSINV", "CP", "PCU33443344", 
            "CPIAUCSL", "M2SL", "REAINTRATREARAT10Y", "HOUST", "CUSR0000SERA02", "IP7108"
        ]
    
    # Descargar datos desde 2010
    datas = fred.download_multiple_series(
        series_ids,
        start_date='1999-01-01'
    )
    
    # Guardar en CSV
    datas.to_csv('fred_data.csv')
    print("Datos descargados y guardados en fred_data.csv")