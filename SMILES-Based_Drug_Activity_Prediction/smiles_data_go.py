import pandas as pd
import numpy as np
from rdkit import Chem # 🔥 RDKit 임포트 추가

# --- 🔥 1. SMILES 정규화 함수 정의 ---
def canonicalize_smiles(smi):
    """SMILES 문자열을 표준 형식(Canonical SMILES)으로 변환합니다."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None # 유효하지 않은 SMILES는 None으로 처리
    except Exception:
        return None
    
# --- 1. CAS Data Processing (pX Value) ---
# This function was correct and remains the same.
def process_cas(file_path):
    print("--- 1. Processing CAS data ---")
    try:
        df = pd.read_excel(file_path, sheet_name='MAP3K5 Ligand IC50s', header=1)
        df.columns = [col.strip() for col in df.columns]
        df_clean = df[['SMILES', 'pX Value']].copy()
        df_clean.dropna(inplace=True)
        df_clean.rename(columns={'pX Value': 'pIC50'}, inplace=True)

        df_clean['source'] = 'CAS' # ✅ 데이터 출처(source) 컬럼 추가

        print(f"✅ CAS: Found {len(df_clean)} entries.")
        return df_clean
    except Exception as e:
        print(f"🚨 CAS Error: {e}")
        return pd.DataFrame()

# --- 2. ChEMBL Data Processing (Corrected: Using pChEMBL Value) ---
def process_chembl_corrected(file_path):
    print("\n--- 2. Processing ChEMBL data (Corrected) ---")
    try:
        df = pd.read_csv(file_path, delimiter=';')
        # Prioritize using the pre-calculated 'pChEMBL Value'
        df_clean = df[['Smiles', 'pChEMBL Value']].copy()
        df_clean.rename(columns={'Smiles': 'SMILES', 'pChEMBL Value': 'pIC50'}, inplace=True)
        df_clean.dropna(inplace=True)

        df_clean['source'] = 'ChEMBL' # ✅ 데이터 출처(source) 컬럼 추가
        # Ensure pIC50 is a numeric type
        df_clean['pIC50'] = pd.to_numeric(df_clean['pIC50'], errors='coerce')
        df_clean.dropna(inplace=True)
        print(f"✅ ChEMBL: Found {len(df_clean)} entries using 'pChEMBL Value'.")
        return df_clean
    except Exception as e:
        print(f"🚨 ChEMBL Error: {e}")
        return pd.DataFrame()

# --- 3. PubChem Data Processing (Corrected: Using Activity_Value) ---
def process_pubchem_corrected(file_path):
    print("\n--- 3. Processing PubChem data (Corrected) ---")
    try:
        df = pd.read_csv(file_path)

        df = df[df['Activity_Qualifier'] == '='].copy()

        # Use the correct 'Activity_Value' column
        df_clean = df[['SMILES', 'Activity_Value']].copy()
        df_clean.dropna(inplace=True)
        df_clean['Activity_Value'] = pd.to_numeric(df_clean['Activity_Value'], errors='coerce')
        df_clean.dropna(inplace=True)

        # Assume Activity_Value is in µM. Convert to pIC50.
        # Formula: pIC50 = -log10(IC50_µM * 10^-6) = 6 - log10(IC50_µM)
        df_positive = df_clean[df_clean['Activity_Value'] > 0].copy()
        df_positive['pIC50'] = 6 - np.log10(df_positive['Activity_Value'])

        df_positive['source'] = 'PubChem' # ✅ 데이터 출처(source) 컬럼 추가
        
        print(f"✅ PubChem: Found {len(df_positive)} entries using 'Activity_Value'.")
        return df_positive[['SMILES', 'pIC50', 'source']]
    except Exception as e:
        print(f"🚨 PubChem Error: {e}")
        return pd.DataFrame()

# --- 4. Execute All Processing and Combine ---
cas_file = './data/smiles/CAS_KPBMA_MAP3K5_IC50s.xlsx'
chembl_file = './data/smiles/ChEMBL_ASK1(IC50).csv'
pubchem_file = './data/smiles/Pubchem_ASK1.csv'

cas_df = process_cas(cas_file)
chembl_df = process_chembl_corrected(chembl_file)
pubchem_df = process_pubchem_corrected(pubchem_file)

print("\n--- Canonicalizing SMILES for all sources ---")

for name, df in [('CAS', cas_df), ('ChEMBL', chembl_df), ('PubChem', pubchem_df)]:
    if not df.empty:
        initial_count = len(df)
        df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
        df.dropna(subset=['SMILES'], inplace=True)
        final_count = len(df)
        print(f"  - {name}: {initial_count} -> {final_count} valid SMILES ({initial_count - final_count} removed)")


print("\n--- 4. Combining all data sources ---")
# Combine all three dataframes
final_df = pd.concat([cas_df, chembl_df, pubchem_df], ignore_index=True)
final_df.dropna(inplace=True) # Ensure no NaN values remain
print(f"- Total entries before deduplication: {len(final_df)}")


# --- 5. Save Final Dataset ---
output_filename = 'train_dataset_with_3source.csv' # Canonicalized 의미로 파일명 변경
try:
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Success: Final canonicalized training dataset saved as '{output_filename}'")
    print("\n--- Final Dataset Sample (Top 5) ---")
    print(final_df.head())
    print("\n--- Source Distribution ---")
    print(final_df['source'].value_counts())
except Exception as e:
    print(f"🚨 An error occurred while saving the final file: {e}")
    
