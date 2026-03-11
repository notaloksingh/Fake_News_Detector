from collections import Counter
p = r'C:\Users\alok\OneDrive\Pictures\Desktop\fake-news-detection\data\news.csv'

cnt = Counter()
lasts = Counter()
first = None

with open(p, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if i == 0:
            first = line
        if i >= 10000:
            break
        parts = line.rstrip('\n').rsplit(',', maxsplit=1)
        if len(parts) == 2:
            lasts[parts[1].strip()] += 1
            cnt['comma_split_2'] += 1
        else:
            # fallback: whitespace token at end
            toks = line.rstrip('\n').split()
            if toks:
                lasts[toks[-1].strip()] += 1
            else:
                lasts[''] += 1
            cnt['fallback_split'] += 1

print('First line repr (truncated):')
print(repr(first)[:500])
print('\nCounts of split method used (first 10k lines):')
print(cnt)
print('\nTop 30 last-token samples and counts:')
for t, c in lasts.most_common(30):
    print(repr(t), c)
