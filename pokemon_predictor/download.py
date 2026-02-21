import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pokemon_predictor import config


def fetch_pokemon_metadata() -> pd.DataFrame:
    """Fetches Pokemon metadata from local CSV."""
    csv_path = config.EXTERNAL_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter/Clean data if necessary
    # The CSV has 'pokedex_number', 'name', 'type_1', 'type_2'
    # We need to construct image_url.
    # Using PokeAPI official artwork convention:
    # https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{id}.png

    # Note: Some entries might be variants (Mega, Alolan).
    # For simplicity, we'll strip 'Mega' or analyze 'pokedex_number'.
    # If pokedex_number is duplicated, it means variants.
    # official-artwork usually exists for the base form ID.
    # Let's keep unique pokedex_numbers to avoid duplicate images/overwrites.

    df_unique = df.drop_duplicates(subset=['pokedex_number']).copy()

    data = []
    for _, row in df_unique.iterrows():
        pid = row['pokedex_number']
        img_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pid}.png"

        data.append({
            'id': pid,
            'name': row['name'],
            'type1': row['type_1'],
            'type2': row['type_2'] if pd.notna(row['type_2']) else None,
            'hp': row['hp'],
            'attack': row['attack'],
            'defense': row['defense'],
            'sp_attack': row['sp_attack'],
            'sp_defense': row['sp_defense'],
            'speed': row['speed'],
            'image_url': img_url
        })

    return pd.DataFrame(data)


def download_image(row: pd.Series, output_dir: str) -> None:
    """Downloads a single image."""
    if not row['image_url']:
        return

    file_path = os.path.join(output_dir, f"{row['name']}.png")
    if os.path.exists(file_path):
        return  # Skip if already exists

    try:
        img_data = requests.get(row['image_url']).content
        with open(file_path, 'wb') as f:
            f.write(img_data)
    except Exception as e:
        print(f"Error downloading {row['name']}: {e}")


def make_dataset():
    """Runs the full data acquisition pipeline."""
    # Ensure directories exist
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Fetch Metadata
    try:
        df_pokemon = fetch_pokemon_metadata()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Save Metadata
    csv_path = config.PROCESSED_DATA_DIR / "pokemon_metadata.csv"
    df_pokemon.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")

    # 3. Download Images
    print("Downloading images...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(lambda row: download_image(row, config.RAW_DATA_DIR),
                               [row for _, row in df_pokemon.iterrows()]),
                  total=len(df_pokemon), desc="Downloading Images"))
    print("Download complete.")

if __name__ == "__main__":
    make_dataset()
