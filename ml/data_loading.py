from pathlib import Path
import pandas as pd

def load_employees(csv_path: Path) -> pd.DataFrame:
    """Carrega dataset de funcionários.
    Levanta erro claro se não existir.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    return df
