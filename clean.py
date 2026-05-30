def clean_dataset(df):
    # Strip any leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Check if required columns exist
    required_columns = {'Date', 'Close', 'Open', 'High', 'Low', 'Volume'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns}")
        return None  # Skip processing if crucial columns are missing

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    # Drop rows where 'Date' couldn't be converted
    df.dropna(subset=['Date'], inplace=True)

    # Remove invalid price rows
    df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]

    # Remove rows where Volume is zero or missing
    df = df[df['Volume'] > 0]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df
