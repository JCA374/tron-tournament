"""Add a note to the site's log:  python experiments/journal.py "title" "text" """
import json
import os
import sys
import time

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'journal.jsonl')
with open(path, 'a') as f:
    f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M'), 'title': sys.argv[1], 'text': sys.argv[2] if len(sys.argv) > 2 else ''}) + '\n')
