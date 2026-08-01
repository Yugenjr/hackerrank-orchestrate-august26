"""
Feature Engineering Module for Message Notification Router
Computes deterministic features and aggregates contextual JSON payloads.
"""
import logging
from typing import Dict, Any, Union
from datetime import datetime
import pandas as pd

from data_loader import DataLoader

logger = logging.getLogger(__name__)

class ContextBuilder:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        # Cache DND parsed windows: { "22:00-07:00": (start_time, end_time) }
        self._dnd_cache = {}

    def build_context(self, message_row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a comprehensive JSON payload of deterministic features for a single message.
        """
        # Support both pandas series (from main loop) and dicts (from tests/optimized loops)
        def get_val(key, default=None):
            return message_row.get(key, default) if hasattr(message_row, "get") else getattr(message_row, key, default)

        user_id = str(get_val("user_id", ""))
        sender_user_id = str(get_val("sender_user_id", ""))
        group_id = str(get_val("group_id", ""))
        business_id = str(get_val("business_id", ""))
        created_at_str = str(get_val("created_at", ""))
        
        forwarded_count = get_val("forwarded_count", 0)
        if forwarded_count is None or pd.isna(forwarded_count):
            forwarded_count = 0
            
        conversation_type = str(get_val("conversation_type", ""))

        context: Dict[str, Any] = {
            "conversation_type": conversation_type,
            "forwarded_count": int(forwarded_count),
        }

        # 1. User Context & DND
        user_info = self.data_loader.get_user(user_id)
        if user_info is not None:
            context["user_dnd_active"] = self._is_dnd_active(
                created_at_str, str(user_info.get("do_not_disturb_window") or "")
            )
            context["user_global_dismissal_rate"] = self._safe_divide(
                user_info.get("notifications_dismissed_30d"),
                (user_info.get("messages_opened_30d") or 1) + (user_info.get("notifications_dismissed_30d") or 0)
            )

        # 2. Group Context
        if conversation_type == "group" and group_id and group_id.lower() != "nan":
            group_member_info = self.data_loader.get_group_member(group_id, user_id)
            if group_member_info is not None:
                context["group_muted_by_user"] = bool(group_member_info.get("group_muted_by_user", False))
                context["group_priority_score"] = self._safe_divide(
                    group_member_info.get("replies_sent_30d"),
                    group_member_info.get("messages_read_30d")
                )

        # 3. Business Context
        if conversation_type == "business" and business_id and business_id.lower() != "nan":
            biz_info = self.data_loader.get_business(business_id)
            biz_history = self.data_loader.get_user_business_history(user_id, business_id)

            if biz_info is not None:
                context["business_trust_score"] = self._compute_business_trust(biz_info)
                context["business_verified"] = bool(biz_info.get("verified", False))
                
                reports = biz_info.get("user_reports_30d")
                context["business_high_reports"] = int(reports) > 50 if reports is not None else False

            if biz_history is not None:
                context["business_promotions_opted_out"] = biz_history.get("promotions_opted_out_at") is not None
                context["business_interaction_strength"] = self._safe_divide(
                    biz_history.get("messages_replied_30d"),
                    biz_history.get("messages_opened_30d")
                )

        # 4. Scam Probability Baseline
        context["baseline_scam_risk"] = (
            int(forwarded_count) > 5 and context.get("business_trust_score", 1.0) < 0.5
        )

        return context

    def _is_dnd_active(self, created_at: str, dnd_window: str) -> bool:
        """
        Determines if the message was sent during the user's DND window.
        Optimized by caching the parsed time strings.
        """
        if not dnd_window or dnd_window.lower() == "nan" or dnd_window == "None":
            return False
            
        try:
            # Only slice the time portion of ISO8601 (T14:30:00Z -> 14:30)
            time_part = created_at.split('T')[1][:5]
            msg_hr, msg_min = map(int, time_part.split(':'))
            msg_mins = msg_hr * 60 + msg_min
            
            if dnd_window not in self._dnd_cache:
                start_str, end_str = dnd_window.split("-")
                start_hr, start_min = map(int, start_str.split(':'))
                end_hr, end_min = map(int, end_str.split(':'))
                self._dnd_cache[dnd_window] = (start_hr * 60 + start_min, end_hr * 60 + end_min)
                
            start_mins, end_mins = self._dnd_cache[dnd_window]
            
            if start_mins < end_mins:
                return start_mins <= msg_mins <= end_mins
            else:
                return msg_mins >= start_mins or msg_mins <= end_mins
        except Exception as e:
            logger.debug(f"Failed to parse DND window '{dnd_window}' or time '{created_at}': {e}")
            return False

    def _compute_business_trust(self, biz_info: Dict[str, Any]) -> float:
        """Computes a heuristic trust score between -1.0 and 1.0."""
        score = 0.0
        if biz_info.get("verified"):
            score += 0.5
        
        official = str(biz_info.get("official_domain", ""))
        used = str(biz_info.get("domain_used_by_sender", ""))
        if official and used and official != "None" and official == used:
            score += 0.5
        elif used and used != "None" and official != used:
            score -= 0.5
            
        reports = biz_info.get("user_reports_30d")
        if reports is not None and int(reports) > 10:
            score -= min(1.0, int(reports) / 100.0)
            
        return max(-1.0, min(1.0, score))

    def _safe_divide(self, num: Any, denom: Any) -> float:
        try:
            n = float(num) if num is not None else 0.0
            d = float(denom) if denom is not None else 1.0
            if d == 0:
                return 0.0
            return n / d
        except (ValueError, TypeError):
            return 0.0
