"""
Hospital Module for Medical Chatbot

This module handles:
1. Finding hospitals using LocationIQ API
2. Geocoding user location
3. Finding nearby hospitals for referrals

Requires: LOCATIONIQ_API_KEY in .env file
Get free API key from: https://locationiq.com/
"""

import os
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# LocationIQ API Configuration
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY", "")
LOCATIONIQ_BASE_URL = "https://us1.locationiq.com/v1"


@dataclass
class Hospital:
    """Hospital data class."""
    name: str
    address: str
    latitude: float
    longitude: float
    distance: Optional[float] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    speciality: str = "General"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'address': self.address,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'distance': self.distance,
            'phone': self.phone or "Not available",
            'website': self.website or "",
            'speciality': self.speciality
        }


class LocationIQHospitalFinder:
    """
    Hospital finder using LocationIQ API.
    """
    
    def __init__(self):
        """Initialize the hospital finder."""
        self.api_key = LOCATIONIQ_API_KEY
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make a request to LocationIQ API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response JSON or None if error
        """
        if not self.api_key:
            print("Warning: LocationIQ API key not configured")
            return None
        
        url = f"{LOCATIONIQ_BASE_URL}/{endpoint}"
        params["key"] = self.api_key
        params["format"] = "json"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"LocationIQ API error: {e}")
            return None
    
    def geocode_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Geocode a location string to coordinates.
        
        Args:
            location: Location string (e.g., "Chennai, India")
            
        Returns:
            Tuple of (latitude, longitude) or None
        """
        params = {
            "q": location,
            "limit": 1
        }
        
        result = self._make_request("search", params)
        
        if result and len(result) > 0:
            lat = float(result[0].get("lat", 0))
            lon = float(result[0].get("lon", 0))
            return (lat, lon)
        
        return None
    
    def find_nearby_hospitals(self, 
                              location: str,
                              radius: int = 10000,
                              max_results: int = 5) -> List[Dict]:
        """
        Find hospitals near a location.
        
        Args:
            location: Location string (city, address, etc.)
            radius: Search radius in meters (default 10km)
            max_results: Maximum number of results
            
        Returns:
            List of hospital dictionaries
        """
        # First, geocode the location
        coords = self.geocode_location(location)
        
        if not coords:
            print(f"Could not geocode location: {location}")
            return []
        
        lat, lon = coords
        
        # Search for hospitals near the coordinates
        params = {
            "lat": lat,
            "lon": lon,
            "tag": "hospital",
            "radius": radius,
            "limit": max_results
        }
        
        results = self._make_request("nearby", params)
        
        if not results:
            return []
        
        hospitals = []
        for place in results:
            hospital = Hospital(
                name=place.get("name", "Unknown Hospital"),
                address=place.get("display_name", "Address not available"),
                latitude=float(place.get("lat", 0)),
                longitude=float(place.get("lon", 0)),
                distance=place.get("distance"),
                phone=place.get("phone"),
                website=place.get("website"),
                speciality="General"
            )
            hospitals.append(hospital.to_dict())
        
        return hospitals
    
    def find_hospitals_by_specialty(self,
                                     location: str,
                                     specialty: str,
                                     radius: int = 15000,
                                     max_results: int = 5) -> List[Dict]:
        """
        Find hospitals by specialty near a location.
        
        Args:
            location: Location string
            specialty: Medical specialty (e.g., "cardiology", "pediatrics")
            radius: Search radius in meters
            max_results: Maximum number of results
            
        Returns:
            List of hospital dictionaries
        """
        # First, geocode the location
        coords = self.geocode_location(location)
        
        if not coords:
            print(f"Could not geocode location: {location}")
            return []
        
        lat, lon = coords
        
        # Map specialist to search terms
        specialty_keywords = {
            "Cardiologist": ["cardiology", "heart", "cardiac"],
            "Dermatologist": ["dermatology", "skin"],
            "Ophthalmologist": ["eye", "ophthalmology", "vision"],
            "Neurologist": ["neurology", "brain", "neuro"],
            "Orthopedist": ["orthopedic", "orthopaedics", "bone", "joint"],
            "ENT Specialist": ["ent", "ear nose throat", "otolaryngology"],
            "Urologist": ["urology", "urological"],
            "Gynecologist": ["gynecology", "obstetrics", "women's health"],
            "Psychiatrist": ["psychiatry", "mental health", "psychiatric"],
            "Endocrinologist": ["endocrinology", "diabetes", "thyroid"],
            "Oncologist": ["oncology", "cancer"],
            "Pulmonologist": ["pulmonology", "lung", "respiratory"],
            "Gastroenterologist": ["gastroenterology", "digestive", "gi"],
            "Pediatrician": ["pediatric", "children's hospital", "child"],
            "General Physician": ["hospital", "medical center", "clinic"]
        }
        
        # Get keywords for the specialty
        keywords = specialty_keywords.get(specialty, ["hospital"])
        
        all_hospitals = []
        
        # Search for each keyword
        for keyword in keywords:
            params = {
                "lat": lat,
                "lon": lon,
                "tag": "hospital",
                "name": keyword,
                "radius": radius,
                "limit": max_results
            }
            
            results = self._make_request("nearby", params)
            
            if results:
                for place in results:
                    hospital = Hospital(
                        name=place.get("name", "Unknown Hospital"),
                        address=place.get("display_name", "Address not available"),
                        latitude=float(place.get("lat", 0)),
                        longitude=float(place.get("lon", 0)),
                        distance=place.get("distance"),
                        phone=place.get("phone"),
                        website=place.get("website"),
                        speciality=specialty
                    )
                    all_hospitals.append(hospital.to_dict())
        
        # Remove duplicates based on name
        seen_names = set()
        unique_hospitals = []
        for h in all_hospitals:
            if h['name'] not in seen_names:
                seen_names.add(h['name'])
                unique_hospitals.append(h)
        
        # Sort by distance if available
        unique_hospitals.sort(key=lambda x: x.get('distance') or float('inf'))
        
        return unique_hospitals[:max_results]


# Singleton instance for reuse
_hospital_finder = None


def get_hospital_finder() -> LocationIQHospitalFinder:
    """
    Get or create the hospital finder singleton.
    
    Returns:
        LocationIQHospitalFinder instance
    """
    global _hospital_finder
    if _hospital_finder is None:
        _hospital_finder = LocationIQHospitalFinder()
    return _hospital_finder


def get_nearby_hospitals(location: str,
                         specialist: str,
                         max_results: int = 5) -> List[Dict]:
    """
    Convenience function to get nearby hospitals.
    
    Args:
        location: Location string (city, address)
        specialist: Specialist type
        max_results: Maximum number of results
        
    Returns:
        List of hospital dictionaries
    """
    finder = get_hospital_finder()
    
    # First try to find hospitals matching the specialty
    hospitals = finder.find_hospitals_by_specialty(location, specialist, max_results=max_results)
    
    # If no specialty-specific hospitals found, search for general hospitals
    if not hospitals:
        hospitals = finder.find_nearby_hospitals(location, max_results=max_results)
    
    return hospitals


def format_hospital_list(hospitals: List[Dict]) -> str:
    """
    Format hospital list for display.
    
    Args:
        hospitals: List of hospital dictionaries
        
    Returns:
        Formatted string
    """
    if not hospitals:
        return "No hospitals found in your area. Please visit the nearest government hospital or consult an online telemedicine service."
    
    lines = []
    for i, h in enumerate(hospitals, 1):
        distance_str = ""
        if h.get('distance'):
            distance_km = h['distance'] / 1000
            distance_str = f" ({distance_km:.1f} km)"
        
        line = f"{i}. **{h['name']}**{distance_str}\n   📍 {h['address'][:100]}..."
        if h.get('phone') and h['phone'] != "Not available":
            line += f"\n   📞 {h['phone']}"
        lines.append(line)
    
    return "\n\n".join(lines)


# Test function
if __name__ == "__main__":
    print("Testing LocationIQ Hospital Module...")
    print("=" * 70)
    
    if not LOCATIONIQ_API_KEY:
        print("Error: LOCATIONIQ_API_KEY not set in .env file")
        print("Get your free API key from: https://locationiq.com/")
        exit(1)
    
    finder = get_hospital_finder()
    
    # Test 1: Geocode location
    print("\nTest 1: Geocoding 'Chennai, India'")
    coords = finder.geocode_location("Chennai, India")
    if coords:
        print(f"  Coordinates: {coords}")
    else:
        print("  Failed to geocode")
    
    # Test 2: Find nearby hospitals
    print("\nTest 2: Finding hospitals near Chennai")
    hospitals = finder.find_nearby_hospitals("Chennai, India", max_results=3)
    print(f"  Found {len(hospitals)} hospitals")
    for h in hospitals:
        print(f"  - {h['name']}")
    
    # Test 3: Find hospitals by specialty
    print("\nTest 3: Finding cardiology hospitals near Chennai")
    cardiology_hospitals = finder.find_hospitals_by_specialty(
        "Chennai, India", "Cardiologist", max_results=3
    )
    print(f"  Found {len(cardiology_hospitals)} hospitals")
    for h in cardiology_hospitals:
        print(f"  - {h['name']} ({h['speciality']})")
    
    print("\n" + "=" * 70)
    print("Tests completed!")
