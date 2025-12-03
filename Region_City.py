from pathlib import Path
import shutil
import pandas as pd

FILE = Path(__file__).resolve().parent / "Vietnam_Climate_enhanced_features.xlsx"

def map_region(city: str) -> str:
  if city in ["Hanoi", "Hai Phong", "Quang Ninh", "Thanh Hoa", "Nghe An (Vinh)"]:
      return "North"
  elif city in [
      "Hue (Thua Thien Hue)",
      "Da Nang",
      "Binh Dinh (Quy Nhon)",
      "Nha Trang (Khanh Hoa)",
      "Pleiku",
      "Buon Ma Thuot (Dak Lak)",
      "Da Lat (Lam Dong)",
  ]:
      return "Central"
  else:
      return "South"

def main():
    print(f"[INFO] Loading file: {FILE.name}")
    df = pd.read_excel(FILE)

    backup = FILE.with_name(FILE.stem + "_backup_orig.xlsx")
    print(f"[INFO] Creating backup: {backup.name}")
    shutil.copy(FILE, backup)

    print("[INFO] Adding 'Region' and one-hot columns...")
    df["Region"] = df["City"].apply(map_region)

    df["Region_North"] = df["Region"] == "North"
    df["Region_Central"] = df["Region"] == "Central"
    df["Region_South"] = df["Region"] == "South"

    df.to_excel(FILE, index=False)
    print("Updated file saved.")

if __name__ == "__main__":
    main()