import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "data"))
from make_dev_set import pick_dev_pages  # noqa: E402

SCRIPT = REPO / "src" / "data" / "make_dev_set.py"


def fake_train():
    """Shaped like Kaggle's train.csv: many German letters, a few account books,
    and a survey category that is one single bound notebook."""
    rows = []
    for d in range(40):  # letters of 1-6 pages
        for p in range(1 + d % 6):
            rows.append(dict(page_id=f"kade_{d:03}_p{p:03}", doc_id=f"kade_{d:03}",
                             text=f"Lieber Bruder {d} {p}", category="kade_letters",
                             label_source="human"))
    for d in range(8):  # account books of 2-4 pages
        for p in range(2 + d % 3):
            rows.append(dict(page_id=f"dominy_{d:03}_p{p:03}", doc_id=f"dominy_{d:03}",
                             text=f"To 3 chairs {d} {p}", category="dominy_accounts",
                             label_source="silver_claude"))
    for p in range(30):  # one notebook
        rows.append(dict(page_id=f"survey_p{p:03}", doc_id="survey_001",
                         text=f"N 40 E {p}", category="survey_notes",
                         label_source="silver_claude"))
    return rows


def by_page(rows):
    return {r["page_id"]: r for r in rows}


def test_no_document_is_split_between_dev_and_train():
    rows = fake_train()
    dev = pick_dev_pages(rows)
    lookup = by_page(rows)
    for doc in {lookup[p]["doc_id"] for p in dev}:
        if doc == "survey_001":
            continue  # the single-document category is split by page on purpose
        pages = {r["page_id"] for r in rows if r["doc_id"] == doc}
        assert pages <= dev, f"{doc} is split between dev and train"


def test_every_category_is_in_dev_and_in_train():
    rows = fake_train()
    dev = pick_dev_pages(rows)
    for cat in {r["category"] for r in rows}:
        in_cat = {r["page_id"] for r in rows if r["category"] == cat}
        assert in_cat & dev, f"{cat} has no dev pages"
        assert in_cat - dev, f"{cat} has no training pages left"


def test_single_document_category_is_split_by_page():
    rows = fake_train()
    dev = pick_dev_pages(rows)
    survey_dev = {p for p in dev if p.startswith("survey")}
    assert 0 < len(survey_dev) < 30


def test_caps_are_respected():
    rows = fake_train()
    lookup = by_page(rows)
    dev = pick_dev_pages(rows, per_category=15, max_fraction=0.25)
    for cat in {r["category"] for r in rows}:
        n = sum(r["category"] == cat for r in rows)
        d = sum(lookup[p]["category"] == cat for p in dev)
        assert d <= max(1, int(0.25 * n)), f"{cat}: {d} dev pages of {n}"


def test_same_seed_same_answer_regardless_of_row_order():
    rows = fake_train()
    assert pick_dev_pages(rows, seed=0) == pick_dev_pages(list(reversed(rows)), seed=0)
    assert pick_dev_pages(rows, seed=0) != pick_dev_pages(rows, seed=1)


def write_train(path, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "doc_id", "text", "category", "label_source"])
        w.writeheader()
        w.writerows(fake_train())


def test_cli_writes_list_without_labels_and_never_repicks(tmp_path):
    train = tmp_path / "train.csv"
    write_train(train)
    out, sol = tmp_path / "dev_ids.csv", tmp_path / "dev_solution.csv"
    base = [sys.executable, str(SCRIPT), "--train", str(train), "--out", str(out), "--solution-out", str(sol)]

    subprocess.run(base, check=True, capture_output=True)
    first = out.read_text(encoding="utf-8")
    header = first.splitlines()[0]
    assert header == "page_id,doc_id,category,label_source"  # labels stay out of git

    # A second run with a different seed keeps the committed list and rebuilds the answer key.
    sol.unlink()
    run = subprocess.run(base + ["--seed", "7"], check=True, capture_output=True, text=True)
    assert "kept the existing dev set" in run.stdout
    assert out.read_text(encoding="utf-8") == first
    with open(sol, encoding="utf-8") as f:
        assert {r["page_id"] for r in csv.DictReader(f)} == {
            line.split(",")[0] for line in first.splitlines()[1:]}

    # --force really does re-pick.
    subprocess.run(base + ["--seed", "7", "--force"], check=True, capture_output=True)
    assert out.read_text(encoding="utf-8") != first


def test_cli_reads_train_csv_with_byte_order_mark(tmp_path):
    # Kaggle's real train.csv starts with a UTF-8 BOM.
    train = tmp_path / "train.csv"
    write_train(train, encoding="utf-8-sig")
    run = subprocess.run([sys.executable, str(SCRIPT), "--train", str(train), "--out", str(tmp_path / "dev_ids.csv"),
                          "--solution-out", str(tmp_path / "sol.csv")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_cli_rejects_a_list_that_does_not_match_train(tmp_path):
    train = tmp_path / "train.csv"
    write_train(train)
    out = tmp_path / "dev_ids.csv"
    out.write_text("page_id,doc_id,category,label_source\nnot_a_page,x,y,human\n", encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), "--train", str(train), "--out", str(out),
                          "--solution-out", str(tmp_path / "sol.csv")], capture_output=True, text=True)
    assert run.returncode != 0 and "not in" in run.stderr


def test_dev_solution_scores_with_kaggle_metric(tmp_path):
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO))
    import metric

    train = tmp_path / "train.csv"
    write_train(train)
    sol = tmp_path / "dev_solution.csv"
    subprocess.run([sys.executable, str(SCRIPT), "--train", str(train),
                    "--out", str(tmp_path / "dev_ids.csv"), "--solution-out", str(sol)],
                   check=True, capture_output=True)

    solution = pd.read_csv(sol, keep_default_na=False)
    perfect = solution[["page_id", "text"]]
    empty = perfect.assign(text="")
    assert metric.score(solution, perfect, "page_id") == 0.0
    assert metric.score(solution, empty, "page_id") == 1.0
