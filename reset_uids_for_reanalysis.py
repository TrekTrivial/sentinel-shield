#!/usr/bin/env python3
"""Reset UIDs for re-analysis with new SHAP explanations."""

import json
from datetime import datetime

# Read current state
with open('.sentinel_state.json', 'r') as f:
    state = json.load(f)

original_count = len(state['processed_uids'])
print(f'Original processed UIDs: {original_count}')

# Remove first 10 UIDs for re-analysis
uids_to_remove = state['processed_uids'][:10]
state['processed_uids'] = state['processed_uids'][10:]
state['last_updated'] = datetime.now().isoformat()

print(f'\nRemoved UIDs for re-analysis:')
for uid in uids_to_remove:
    print(f'  - {uid}')

print(f'\nRemaining processed UIDs: {len(state["processed_uids"])}')

# Save modified state
with open('.sentinel_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print('\n✅ State file updated - monitor will re-analyze removed UIDs on next run')
print('📊 This will help verify if new SHAP explanations produce better analysis')
