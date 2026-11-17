"""Starts one of this folder's scripts inside a SageMaker training job (launched by sagemaker_job.py).

Before running the script, it links each folder the notebook would use to where the job keeps it:

    /home/ec2-user/SageMaker/churro-work/data          ->  /opt/ml/input/data/data   (an input channel)
    /home/ec2-user/SageMaker/churro-work/runs/<a run>  ->  /opt/ml/input/data/run0   (an adapter to start from)
    /home/ec2-user/SageMaker/churro-work/runs/<this>   ->  /opt/ml/checkpoints       (copied to S3 as it's written)

So the script's arguments, and the paths saved in summary.json, are exactly the same in the job
as on the notebook instance. Everything the script prints is also appended to a log file in the
output folder (like `tee -a` in the notebook), so it reaches S3 while the job runs.
"""

import json
import os
import shutil
import subprocess
import sys


def main():
    out = os.environ["CHURRO_OUT"]
    links = {value: "/opt/ml/input/data/" + key[len("CHURRO_LINK_"):]
             for key, value in os.environ.items() if key.startswith("CHURRO_LINK_")}
    links[out] = "/opt/ml/checkpoints"
    for local, target in links.items():
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if not os.path.lexists(local):
            os.symlink(target, local)
        print(f"{local} -> {target}", flush=True)

    # The 7.5 GB model download goes wherever the job has most room (its storage volume holds
    # /tmp and /opt/ml; the container's own disk can be small). Not under checkpoints, model or
    # output, which SageMaker copies to S3.
    if "HF_HOME" not in os.environ and os.path.isdir("/opt/ml"):
        roomiest = max(("/tmp", "/opt/ml", os.path.expanduser("~")),
                       key=lambda d: shutil.disk_usage(d).free if os.path.isdir(d) else -1)
        os.environ["HF_HOME"] = os.path.join(roomiest, "hf_cache")
        print(f"model downloads -> {os.environ['HF_HOME']}", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, os.environ["CHURRO_SCRIPT"])
    # The script's arguments come from job_args.json, written by sagemaker_job.py. The ones the
    # container puts on this command line are incomplete: it drops empty values (--enhance "").
    args_file = os.path.join(here, "job_args.json")
    if os.path.exists(args_file):
        args = [part for key, value in json.load(open(args_file, encoding="utf-8")).items()
                for part in (f"--{key}", str(value))]
    else:
        args = sys.argv[1:]
    with open(os.path.join(out, os.environ.get("CHURRO_LOG", "job.log")), "a", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", script, *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
