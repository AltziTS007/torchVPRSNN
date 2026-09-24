import pathlib

p = pathlib.Path('core/neuronal_assignments.py')
content = p.read_text(encoding='utf-8')

old_lines = [
    '    eps = 1e-8',
    '    if rule == "product":',
    '        log_probs = torch.log(prob_scores + eps)',
    '    ',
    '    for valid_idx, i in enumerate(valid_indices):',
    '        valid_targets.append(targets[i])',
    '        ',
    '        # Window bounds: [i - k + 1, i]',
    '        start_idx = i - k + 1',
    '        end_idx = i + 1',
    '        ',
    '        if rule == "product":',
    '            window_log_probs = log_probs[start_idx:end_idx]',
]

new_lines = [
    '    eps = 1e-8',
    '    # NOTE: Compute log on-the-fly per window to avoid OOM on large datasets.',
    '    # Pre-computing the full log_probs tensor would duplicate a [n_query, n_classes]',
    '    # matrix in memory, which exceeds GPU memory for datasets > ~10k samples.',
    '    ',
    '    for valid_idx, i in enumerate(valid_indices):',
    '        valid_targets.append(targets[i])',
    '        ',
    '        # Window bounds: [i - k + 1, i]',
    '        start_idx = i - k + 1',
    '        end_idx = i + 1',
    '        ',
    '        if rule == "product":',
    '            window_log_probs = torch.log(prob_scores[start_idx:end_idx] + eps)',
]

# Detect line ending
if '\r\n' in content:
    le = '\r\n'
elif '\r' in content:
    le = '\r'
else:
    le = '\n'

old_block = le.join(old_lines)
new_block = le.join(new_lines)

if old_block in content:
    content = content.replace(old_block, new_block)
    p.write_text(content, encoding='utf-8', newline='')
    print('Patch applied successfully')
else:
    print('ERROR: old string not found')
    idx = content.find('eps = 1e-8')
    print('eps at index:', idx)
    if idx >= 0:
        print(repr(content[idx:idx+400]))
