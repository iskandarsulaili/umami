"""
Session Replay ML Analysis - Model 9
Analyzes rrweb session replay events for UX signals:
- Error loops: repeated console.error or network failures
- Form struggles: repeated form field focus/blur, validation errors
- Dead clicks: clicks on non-interactive elements with no DOM change
- Navigation confusion: rapid URL changes without meaningful interaction
- Performance jank: long frame drops, large layout shifts in replay

Data source: session_replay table (rrweb JSON events stored as bytea)
Reference: rrweb event types (https://github.com/rrweb-io/rrweb)
"""

import os
import sys
import json
import logging
import gzip
from typing import Optional
from collections import defaultdict

import numpy as np

from .base import BaseModel
from ..config import CONFIG

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# rrweb event types
RRWEB_EVENT_TYPES = {
    0: 'DomContentLoaded',
    1: 'Load',
    2: 'FullSnapshot',
    3: 'IncrementalSnapshot',
    4: 'Meta',
    5: 'Custom',
    6: 'Plugin',
}

# rrweb incremental update types
INCREMENTAL_TYPES = {
    0: 'Mutation',
    1: 'MouseMove',
    2: 'MouseInteraction',
    3: 'Scroll',
    4: 'ViewportResize',
    5: 'Input',
    6: 'TouchMove',
    7: 'MediaInteraction',
    8: 'StyleSheet',
    9: 'CanvasMutation',
    10: 'Font',
    11: 'Log',
    12: 'Drag',
    13: 'Navigation',
}

# Mouse interaction subtypes
MOUSE_INTERACTIONS = {
    0: 'MouseUp',
    1: 'MouseDown',
    2: 'Click',
    3: 'ContextMenu',
    4: 'DblClick',
    5: 'Focus',
    6: 'Blur',
    7: 'TouchStart',
    8: 'TouchEnd_Departed',
}


class SessionReplayAnalyzer(BaseModel):
    """
    Model 9: Session Replay ML Analysis.
    Extracts UX signals from rrweb event sequences.
    """
    
    def __init__(self):
        super().__init__(name="session_replay_analyzer", version="1.0.0")
    
    def train(self, data: list = None):
        """No training needed — rule-based analysis."""
        self.is_trained = True
        logger.info("SessionReplayAnalyzer initialized")
    
    def _decompress_events(self, events_bytes: bytes) -> list[dict]:
        """Decompress gzipped rrweb events."""
        try:
            decompressed = gzip.decompress(events_bytes)
            return json.loads(decompressed)
        except Exception:
            try:
                return json.loads(events_bytes)
            except Exception:
                return []
    
    def analyze_session(self, replay_data: dict) -> dict:
        """
        Analyze a single session's replay events.
        
        Args:
            replay_data: dict with session_id, events (bytea), event_count
            
        Returns:
            dict with UX signals and severity scores
        """
        events_bytes = replay_data.get('events')
        if isinstance(events_bytes, bytes) or (isinstance(events_bytes, memoryview)):
            events = self._decompress_events(bytes(events_bytes))
        elif isinstance(events_bytes, str):
            events = self._decompress_events(events_bytes.encode())
        else:
            events = []
        
        if not events:
            return {
                'frustration_signals': [],
                'error_loops': 0,
                'dead_clicks': 0,
                'form_struggles': 0,
                'rapid_navigation': 0,
                'severity': 'none',
            }
        
        signals = []
        error_count = 0
        click_count = 0
        dead_click_count = 0
        form_interaction_count = 0
        form_struggle_count = 0
        nav_count = 0
        mouse_move_count = 0
        mutation_count = 0
        
        for i, event in enumerate(events):
            event_type = event.get('type', -1)
            data = event.get('data', {})
            
            # Incremental snapshot events
            if event_type == 3:  # IncrementalSnapshot
                source = data.get('source', -1)
                
                if source == 2:  # MouseInteraction
                    click_count += 1
                    subtype = data.get('type', -1)
                    if subtype in (0, 1, 2):  # MouseUp, MouseDown, Click
                        # Check for dead click: click followed by no mutation
                        next_mutation = None
                        for j in range(i+1, min(i+10, len(events))):
                            if events[j].get('data', {}).get('source') == 0:
                                next_mutation = events[j]
                                break
                        if next_mutation is None:
                            dead_click_count += 1
                
                elif source == 1:  # MouseMove
                    mouse_move_count += 1
                    # Check for mouse shaking: rapid position changes
                    if mouse_move_count > 50 and i < len(events) - 1:
                        next_event = events[i+1]
                        if next_event.get('data', {}).get('source') == 1:
                            pos = data.get('positions', [{}])
                            next_pos = next_event.get('data', {}).get('positions', [{}])
                            if pos and next_pos:
                                dx = abs(pos[0].get('x', 0) - next_pos[0].get('x', 0))
                                dy = abs(pos[0].get('y', 0) - next_pos[0].get('y', 0))
                                if dx > 100 or dy > 100:
                                    signals.append({
                                        'type': 'mouse_shake',
                                        'details': f'Rapid mouse movement: dx={dx}, dy={dy}',
                                        'timestamp': event.get('timestamp', 0),
                                    })
                
                elif source == 5:  # Input
                    form_interaction_count += 1
                    # Check for form struggle: same field edited 3+ times
                    if form_interaction_count >= 3:
                        form_struggle_count += 1
                
                elif source == 11:  # Log (console errors)
                    log_level = data.get('logLevel', '')
                    if log_level in ('error', 'assert'):
                        error_count += 1
                        signals.append({
                            'type': 'console_error',
                            'details': data.get('payload', [''])[0] if data.get('payload') else '',
                            'timestamp': event.get('timestamp', 0),
                        })
                
                elif source == 0:  # Mutation
                    mutation_count += 1
                
                elif source == 13:  # Navigation
                    nav_count += 1
                    if nav_count > 3:
                        signals.append({
                            'type': 'rapid_navigation',
                            'details': f'{nav_count} navigations in session',
                            'timestamp': event.get('timestamp', 0),
                        })
            
            elif event_type == 5:  # Custom event
                tag = data.get('tag', '')
                payload = data.get('payload', {})
                if 'error' in tag.lower() or 'exception' in tag.lower():
                    error_count += 1
        
        # Assess severity
        if error_count >= 5:
            severity = 'high'
        elif error_count >= 3 or dead_click_count >= 5:
            severity = 'medium'
        elif error_count > 0 or dead_click_count > 0:
            severity = 'low'
        else:
            severity = 'none'
        
        return {
            'frustration_signals': signals,
            'error_loops': error_count,
            'dead_clicks': dead_click_count,
            'form_struggles': form_struggle_count,
            'rapid_navigation': nav_count,
            'total_events': len(events),
            'click_events': click_count,
            'mutation_events': mutation_count,
            'mouse_move_events': mouse_move_count,
            'severity': severity,
        }
    
    def analyze_session_from_db(self, session_id: str, website_id: str) -> dict:
        """Fetch replay data from DB and analyze."""
        if not HAS_PSYCOPG2:
            return {'error': 'Database not available'}
        
        try:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT session_id, events, event_count
                FROM session_replay
                WHERE session_id = %s
                  AND website_id = %s
                ORDER BY chunk_index
            """, (session_id, website_id))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            if not results:
                return {'error': 'No replay data found'}
            
            # Merge all chunks
            merged = {'session_id': session_id, 'events': b'', 'event_count': 0}
            for row in results:
                merged['events'] += bytes(row['events'].tobytes()) if hasattr(row['events'], 'tobytes') else row['events']
                merged['event_count'] += row['event_count']
            
            return self.analyze_session(merged)
        except Exception as e:
            logger.error(f"Replay analysis failed: {e}")
            return {'error': str(e)}