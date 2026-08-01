from schemas_intelligence import StructuralElements, Signal

class StructuralExtractor:
    @staticmethod
    def extract(text: str) -> StructuralElements:
        """
        Identifies imperative clauses, temporal deadlines, and calendar events.
        """
        structures = StructuralElements()
        
        # Deadlines
        deadline_keywords = ['by tomorrow', 'due', 'deadline is', 'needs to be done by', 'expires']
        for kw in deadline_keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                snippet = text[max(0, idx-10):min(len(text), idx+40)]
                structures.deadlines.append(Signal[str](value="Relative Deadline", confidence=0.85, evidence=snippet.strip()))
                break
                
        # Action Items
        action_keywords = ['please sign', 'pay by', 'call me back', 'reply to', 'action required', 'click here to']
        for kw in action_keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                snippet = text[max(0, idx-10):min(len(text), idx+40)]
                structures.action_items.append(Signal[str](value="Imperative Action", confidence=0.9, evidence=snippet.strip()))
                break
                
        # Events
        event_keywords = ['invite you to', 'scheduled for', 'calendar event', 'rsvp']
        for kw in event_keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                snippet = text[max(0, idx-10):min(len(text), idx+40)]
                structures.events.append(Signal[str](value="Calendar Event", confidence=0.85, evidence=snippet.strip()))
                break
                
        return structures
