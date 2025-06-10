from datetime import timedelta

def generate_sliding_windows(transactions_df, window_size_days=7, step_days=1):
    """
    Generate sliding window snapshots of transaction data.

    Args:
        transactions_df (pd.DataFrame): Contains 'timestamp' column.
        window_size_days (int): Size of each window.
        step_days (int): Step size to move the window.

    Yields:
        pd.DataFrame: Transactions within the current window.
    """
    start = transactions_df['timestamp'].min()
    end = transactions_df['timestamp'].max()

    while start + timedelta(days=window_size_days) <= end:
        mask = (transactions_df['timestamp'] >= start) & \
               (transactions_df['timestamp'] < start + timedelta(days=window_size_days))
        yield transactions_df.loc[mask]
        start += timedelta(days=step_days)
