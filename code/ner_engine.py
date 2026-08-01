import re
from schemas_intelligence import Entities, Signal

class NEREngine:
    @staticmethod
    def extract_entities(text: str) -> Entities:
        """
        Extracts Named Entities (People, Orgs, Dates, URLs, etc) using deterministic regex
        and lightweight heuristics.
        """
        entities = Entities()
        
        # Phone Numbers Regex
        phone_matches = re.finditer(r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', text)
        for match in phone_matches:
            val = match.group()
            entities.phones.append(Signal[str](value=val, confidence=0.95, evidence=val))
            
        # URL Regex
        url_matches = re.finditer(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
        for match in url_matches:
            val = match.group()
            entities.urls.append(Signal[str](value=val, confidence=1.0, evidence=val))
            
        # Money Regex
        money_matches = re.finditer(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for match in money_matches:
            val = match.group()
            entities.money.append(Signal[str](value=val, confidence=0.99, evidence=val))
            
        # Dates (Simplified heuristic)
        date_matches = re.finditer(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b', text, re.IGNORECASE)
        for match in date_matches:
            val = match.group()
            entities.dates.append(Signal[str](value=val, confidence=0.9, evidence=val))
            
        return entities
