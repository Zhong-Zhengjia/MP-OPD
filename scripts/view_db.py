import sqlite3
from rich import print


def get_question_acc_details(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT question_id, rollout_id, acc
        FROM generations
        ORDER BY question_id, rollout_id
    """)

    question_to_accs = {}

    for question_id, rollout_id, acc in cursor.fetchall():
        if question_id not in question_to_accs:
            question_to_accs[question_id] = []
        question_to_accs[question_id].append(acc)

    results = []
    for question_id, accs in question_to_accs.items():
        all_same = len(set(accs)) == 1
        results.append({
            "question_id": question_id,
            "accs": accs,
            "all_same": all_same,
        })

    conn.close()
    return results

if __name__ == '__main__':
    db_path = 'eval_outputs/Qwen3-1.7B_DeepMath-103K_pass@4.sqlite'
    res = get_question_acc_details(db_path)
    for item in res:
        print(item["question_id"], item["accs"], item["all_same"])