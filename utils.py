import numpy as np  # ✅ Correct

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two geographic points"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * 1000 * np.arcsin(np.sqrt(a))  # Meters