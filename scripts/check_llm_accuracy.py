import csv

def collapse(label):
    if label in ('incidental', 'comparative'):
        return 'not_about_sga'
    return label

with open('docs/validation_sample.csv') as f:
    manual_rows = {r['comment_id']: r for r in csv.DictReader(f) if r['manual_subject_label'].strip()}

with open('docs/validation_sample_with_llm.csv') as f:
    llm_rows = {r['comment_id']: r for r in csv.DictReader(f)}

correct = 0
total = 0
mismatches = []

for cid, mrow in manual_rows.items():
    if cid not in llm_rows:
        continue
    manual_label = collapse(mrow['manual_subject_label'].strip())
    llm_label = llm_rows[cid]['llm_subject'].strip()
    total += 1
    if manual_label == llm_label:
        correct += 1
    else:
        mismatches.append((cid, manual_label, llm_label, mrow['body'][:80]))

print(f'Accuracy on {total} labeled rows: {correct}/{total} = {correct/total:.1%}')
print()
print('Mismatches:')
for cid, manual, llm, body in mismatches:
    print(f'  [{cid}] manual={manual!r} llm={llm!r}: {body}')
