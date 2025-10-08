#!/usr/bin/env python3
"""
Weak labeling for pain detection using heuristics
"""
from dataclasses import dataclass, asdict
import json
import re
import sys
import math

# Pain detection patterns
PATS = [
    re.compile(r"won't work", re.I), 
    re.compile(r"stuck", re.I), 
    re.compile(r"error", re.I),
    re.compile(r"how do i", re.I), 
    re.compile(r"integration.*fail", re.I), 
    re.compile(r"manual", re.I)
]

# Problem categorization
PROBLEM = [
    (re.compile(r'onboarding|SOP|checklist', re.I), 'Manual onboarding → SOP + bot'),
    (re.compile(r'webhook|retry|queue', re.I), 'Unreliable webhooks → queue + replay'),
    (re.compile(r'csv|copy.*paste|export', re.I), 'Data shuffling → scheduled ETL'),
]

def pain_score(text: str, eng: int) -> float:
    hits = sum(1 for p in PATS if p.search(text))
    return math.tanh(hits * (1 + math.log1p(max(0, eng))))

def guess_problem(txt: str) -> str:
    for rx, lab in PROBLEM:
        if rx.search(txt): 
            return lab
    return 'Ambiguous automation interest → discovery call'

@dataclass
class Row:
    id: str
    time: str
    source: str
    text: str
    eng: int
    from typing import Optional
    url: Optional[str] = None
    org: Optional[str] = None

def label(row: Row) -> dict:
    p = pain_score(row.text, row.eng)
    sev = 3 if p >= 0.8 else 2 if p >= 0.5 else 1 if p >= 0.25 else 0
    opp = bool(re.search(r'launch|released|pricing|tier|starter|pro|studio|discount', row.text, re.I))
    
    return {
        **asdict(row),
        "labels": {
            "pain": p >= 0.25, 
            "opportunity": opp, 
            "problem_guess": guess_problem(row.text), 
            "severity": sev
        },
        "features": {"pain_score": p}
    }

if __name__ == "__main__":
    for line in sys.stdin:
        r = json.loads(line)
        row_data = {k: r.get(k) for k in Row.__annotations__.keys()}
        print(json.dumps(label(Row(**row_data))))