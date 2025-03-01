from sklearn.model_selection import train_test_split


def split_data_by_track(processed_df, test_size=0.2, val_size=0.25, random_state=42):
    """Split data into train/val/test sets while keeping tracks intact"""
    # Get unique tracks
    tracks = processed_df['source_file'].unique()
    
    # First split: separate test tracks
    train_val_tracks, test_tracks = train_test_split(
        tracks, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Second split: separate validation tracks from remaining
    train_tracks, val_tracks = train_test_split(
        train_val_tracks, 
        test_size=val_size/(1-test_size),  # Adjust for previous split
        random_state=random_state
    )
    
    # Create filtered dataframes
    train_df = processed_df[processed_df.source_file.isin(train_tracks)]
    val_df = processed_df[processed_df.source_file.isin(val_tracks)]
    test_df = processed_df[processed_df.source_file.isin(test_tracks)]
    
    return train_df, val_df, test_df