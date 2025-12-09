try:
    from Bio.PDB.Polypeptide import one_to_three
    print(f"Import successful. M -> {one_to_three('M')}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
