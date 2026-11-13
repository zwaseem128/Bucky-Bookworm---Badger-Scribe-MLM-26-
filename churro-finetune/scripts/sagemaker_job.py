"""Run one of this folder's scripts as a SageMaker training job instead of on the notebook's own GPU.

A training job rents its GPU only while the script runs, plus a few minutes to start. That means
the notebook instance can be a small CPU one (ml.t3.medium) that costs a few cents an hour, and
the GPU is paid for only while it trains or reads pages.

How it works, so that nothing else in the notebooks has to change:

- Every folder under churro-work that the script reads (the data, and any run whose adapter it
  uses, including adapters named in that run's summary.json) is copied to the same path in S3,
  s3://<bucket>/churro-work/..., and handed to the job as an input channel.
- The folder the script writes to becomes the job's checkpoint folder. SageMaker copies it to
  s3://<bucket>/churro-work/<the same path> while the job runs, and copies it back in if the job
  restarts, so a run that stopped carries on where it was, just as it does on the notebook.
- Inside the job, job_entry.py links each of those folders to the path it has on the notebook,
  so the scripts' arguments, and the paths written into summary.json, are the same in both places.
- When the job ends, its results are copied back into the same folder on the notebook (without
  the large checkpoint files), so the notebooks' later steps read them as if they ran locally.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.dirname(HERE)        # churro-finetune/
REPO = os.path.dirname(NOTEBOOKS)        # the repository, with metric.py at the top


def _s3(path, work, bucket):
    """s3://<bucket>/churro-work/<path inside churro-work>/  (the trailing / keeps prefixes apart)."""
    rel = os.path.relpath(path, work).replace(os.sep, "/")
    return f"s3://{bucket}/{os.path.basename(work)}/{rel}/"


def _sync(src, dst, skip_checkpoints=True):
    cmd = ["aws", "s3", "sync", src, dst, "--only-show-errors"]
    if skip_checkpoints:
        cmd += ["--exclude", "checkpoints/*", "--exclude", "*/checkpoints/*"]
    subprocess.run(cmd, check=True)


def _inside(path, folder):
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    return path == folder or path.startswith(folder + os.sep)


def _root(path, work):
    """The folder shipped for a path: churro-work/data, or one run's folder churro-work/runs/<run>."""
    rel = os.path.relpath(os.path.abspath(path), work).split(os.sep)
    if rel[0] == "..":
        return None
    return os.path.join(work, *rel[:2]) if rel[0] in ("runs", "predictions") and len(rel) > 1 \
        else os.path.join(work, rel[0])


def _paths_in(value, work):
    """Every churro-work path inside a summary.json (adapter_dir, compare_adapter, ...)."""
    if isinstance(value, str) and value.startswith(work + "/"):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _paths_in(v, work)
    elif isinstance(value, list):
        for v in value:
            yield from _paths_in(v, work)


def _folders_to_ship(args, out_folder, work):
    roots, todo = [], [v for v in args.values() if isinstance(v, str)]
    while todo:
        path = todo.pop()
        if not path.startswith(work + "/") or not os.path.exists(path) or _inside(path, out_folder):
            continue
        root = _root(path, work)
        if root is None or root in roots or _inside(out_folder, root):
            continue
        roots.append(root)
        summary = os.path.join(root, "summary.json")
        if os.path.exists(summary):
            todo += list(_paths_in(json.load(open(summary)), work))
    return roots


def _stage_code(work):
    """The job's code: this folder's scripts, the competition's metric.py and requirements.txt."""
    stage = os.path.join(work, "job_code")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for f in glob.glob(os.path.join(HERE, "*.py")):
        shutil.copy(f, stage)
    shutil.copy(os.path.join(REPO, "metric.py"), stage)
    shutil.copy(os.path.join(NOTEBOOKS, "requirements.txt"), stage)
    return stage


def _sdk():
    """The SageMaker Python SDK, version 2 (version 3 dropped the estimator classes used here)."""
    try:
        import sagemaker
        if int(sagemaker.__version__.split(".")[0]) >= 3:
            raise ImportError
    except ImportError:
        print('Installing the SageMaker SDK, version 2 ("sagemaker<3") ...')
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sagemaker>=2.240,<3"], check=True)
        raise SystemExit("Installed. Restart the kernel (Kernel -> Restart Kernel), then run the cells again from the top.")
    return sagemaker


def run_job(script, args, out_folder, *, work, bucket, run_name, instance_type="ml.g5.xlarge",
            max_hours=6, spot=False, owner="", log_name="job.log", wait=True):
    """Run scripts/<script> with --<key> <value> for each of args, in a training job on instance_type.

    out_folder is the churro-work folder the script writes to. Returns the job's name."""
    if not bucket:
        raise SystemExit("No bucket saved. Run Notebook 1, Step 2 first (it saves your bucket name).")
    sagemaker = _sdk()
    from sagemaker.huggingface import HuggingFace

    out_folder = os.path.abspath(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    roots = _folders_to_ship(args, out_folder, work)
    channels = {("data" if os.path.basename(r) == "data" else f"run{i}"): r for i, r in enumerate(roots)}
    for channel, root in channels.items():
        print(f"Copying {os.path.relpath(root, work)} to S3 (only what's new) ...", flush=True)
        _sync(root, _s3(root, work, bucket))
    if os.listdir(out_folder):   # a run that already started: the job carries on from its checkpoints
        print(f"Copying what {os.path.relpath(out_folder, work)} already has to S3 ...", flush=True)
        _sync(out_folder, _s3(out_folder, work, bucket), skip_checkpoints=False)

    session = sagemaker.Session()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", run_name).strip("-")[:40].strip("-")
    job_name = f"churro-{slug}-{time.strftime('%m%d-%H%M%S')}"
    environment = {"CHURRO_SCRIPT": script, "CHURRO_OUT": out_folder, "CHURRO_LOG": log_name,
                   **{f"CHURRO_LINK_{c}": r for c, r in channels.items()}}
    tags = [{"Key": "Project", "Value": "BadgerScribe"}] + ([{"Key": "Name", "Value": owner}] if owner else [])
    estimator = HuggingFace(
        entry_point="job_entry.py",
        source_dir=_stage_code(work),
        role=sagemaker.get_execution_role(),
        instance_type=instance_type,
        instance_count=1,
        volume_size=60,                      # GB: the 7.5 GB model, the page images and checkpoints
        max_run=int(max_hours * 3600),       # SageMaker stops the job after this, whatever happens
        use_spot_instances=spot,
        **({"max_wait": int(max_hours * 3600 * 2)} if spot else {}),
        checkpoint_s3_uri=_s3(out_folder, work, bucket),
        checkpoint_local_path="/opt/ml/checkpoints",
        transformers_version="4.56",         # the container; requirements.txt then sets the exact versions
        pytorch_version="2.8",
        py_version="py312",
        output_path=f"s3://{bucket}/{os.path.basename(work)}/job-output/",
        hyperparameters={k: v for k, v in args.items()},
        environment=environment,
        tags=tags,
        sagemaker_session=session,
        disable_profiler=True,
    )
    inputs = {c: _s3(r, work, bucket) for c, r in channels.items()}
    print(f"Starting training job {job_name} on {instance_type}{' (spot)' if spot else ''}. "
          f"The GPU is billed from now until it ends; this notebook can be closed meanwhile.", flush=True)
    estimator.fit(inputs, job_name=job_name, wait=False)
    if not wait:
        return job_name
    try:
        estimator.latest_training_job.wait(logs="All")
    except KeyboardInterrupt:
        print(f"\nStopped watching, but the job is still running (and billed). To stop it:\n"
              f"    sagemaker_job.stop('{job_name}')\nIts results reach S3 either way; fetch them with Step 6.")
        return job_name
    except Exception as e:   # the job failed: still bring back its log
        print(f"\nThe job ended with an error: {e}")
    fetch(out_folder, work=work, bucket=bucket)
    report(job_name)
    return job_name


def fetch(out_folder, *, work, bucket):
    """Copy a job's results from S3 into out_folder on the notebook (without the large checkpoints)."""
    os.makedirs(out_folder, exist_ok=True)
    _sync(_s3(os.path.abspath(out_folder), work, bucket), out_folder)


def report(job_name):
    import boto3
    d = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)
    billed = d.get("BillableTimeInSeconds") or 0
    print(f"Job {job_name}: {d['TrainingJobStatus']}"
          f"{' (' + d['FailureReason'] + ')' if d.get('FailureReason') else ''}; "
          f"billed for {billed / 60:.0f} min of {d['ResourceConfig']['InstanceType']}.")


def stop(job_name):
    import boto3
    boto3.client("sagemaker").stop_training_job(TrainingJobName=job_name)
    print(f"Stopping {job_name}. What it saved so far is in S3; running the same run again carries on from it.")
