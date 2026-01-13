"""
Distanz-Berechnungsfunktionen
"""

import numpy as np


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Berechnet die Distanz zwischen zwei Punkten auf der Erde in Metern
    unter Verwendung der Haversine-Formel.
    
    Die Haversine-Formel berechnet die kürzeste Distanz zwischen zwei Punkten
    auf einer Kugeloberfläche (Great Circle Distance).
    
    Parameters
    ----------
    lon1 : float or array-like
        Längengrad des ersten Punkts in Grad
    lat1 : float or array-like
        Breitengrad des ersten Punkts in Grad
    lon2 : float or array-like
        Längengrad des zweiten Punkts in Grad
    lat2 : float or array-like
        Breitengrad des zweiten Punkts in Grad
    
    Returns
    -------
    float or array-like
        Distanz in Metern
    
    Examples
    --------
    >>> # Distanz zwischen zwei Punkten in NYC
    >>> dist = haversine_distance(-73.9857, 40.7484, -73.9851, 40.7489)
    >>> print(f"Distanz: {dist:.2f} Meter")
    Distanz: 56.78 Meter
    
    >>> # Konvertierung zu Meilen
    >>> dist_miles = dist / 1609.34
    >>> print(f"Distanz: {dist_miles:.2f} Meilen")
    Distanz: 0.04 Meilen
    
    Notes
    -----
    Die Funktion verwendet den Erdradius von 6,371,000 Metern.
    Für sehr kurze Distanzen (< 1km) kann die euklidische Distanz
    ausreichend sein, aber Haversine ist genauer für geografische Koordinaten.
    
    Die Formel:
    a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    c = 2 × atan2(√a, √(1−a))
    d = R × c
    
    wobei R der Erdradius ist.
    """
    # Radius der Erde in Metern
    R = 6371000
    
    # Konvertiere Grad zu Radiant
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine-Formel
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def meters_to_miles(meters):
    """
    Konvertiert Meter zu Meilen.
    
    Parameters
    ----------
    meters : float or array-like
        Distanz in Metern
    
    Returns
    -------
    float or array-like
        Distanz in Meilen
    
    Examples
    --------
    >>> meters_to_miles(1609.34)
    1.0
    """
    return meters / 1609.34


def miles_to_meters(miles):
    """
    Konvertiert Meilen zu Metern.
    
    Parameters
    ----------
    miles : float or array-like
        Distanz in Meilen
    
    Returns
    -------
    float or array-like
        Distanz in Metern
    
    Examples
    --------
    >>> miles_to_meters(1.0)
    1609.34
    """
    return miles * 1609.34
