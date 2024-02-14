import os
from glob import glob
import submitit
import dt.main
from joblib import load
from termcolor import cprint
from hydra.core.utils import JobStatus, JobReturn


def load_result(file_path):
    try:
        _, job_result = load(file_path)
        if not isinstance(job_result, JobReturn):
            return None
        else:
            return job_result
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_status(job_result):
    if job_result is None:
        return JobStatus.UNKNOWN
    else:
        return job_result.status


def rerun_job(file_path):
    # Adjust this function according to your setup
    executor = submitit.AutoExecutor(folder=os.path.dirname(file_path))
    hydra_return = load_result(file_path)

    executor.update_parameters(
        timeout_min=720, slurm_partition="compute",
        tasks_per_node=1, slurm_gpus_per_task=1, cpus_per_task=8
    )
    job = executor.submit(dt.main.main, hydra_return.cfg)
    print("Submitted", job.job_id)

    return job


def main(base_dir):
    result_files = glob(os.path.join(base_dir, "**/.submitit/**/*_result.pkl"), recursive=True)

    completed_count = 0
    failed_jobs = []
    unknown_jobs = []

    print(result_files)
    for file_path in result_files:
        job_result = load_result(file_path)
        status = get_status(job_result)
        job_id, job_array_id, job_seq_id = os.path.basename(file_path).removesuffix("_result.pkl").split("_")
        print(f"{job_id}_{job_array_id}: {status}")

        if status == JobStatus.COMPLETED:
            completed_count += 1
        elif status == JobStatus.FAILED:
            failed_jobs.append(file_path)
        else:
            unknown_jobs.append(file_path)

    print("\nSummary:")
    print(f"Total jobs: {len(result_files)}")
    print(f"Successful jobs: {completed_count}")
    print(f"Failed jobs: {len(failed_jobs)}")
    print(f"Unknown jobs: {len(unknown_jobs)}")

    for file_path in failed_jobs + unknown_jobs:
        check_outputs = input("\nDo you want to check the output from the failed jobs? (yes/no): ").strip().lower()
        if check_outputs == "yes":
            with open(file_path.replace("_result.pkl", "_log.out")) as f:
                std_out = list(filter(lambda x: "sqlitedict" not in x, f.readlines()))
                cprint(text="".join(std_out), color="blue")
            with open(file_path.replace("_result.pkl", "_log.err")) as f:
                std_error = list(filter(lambda x: x.strip() and "cost=" not in x, f.readlines()))
                cprint(text="".join(std_error), color="red")
            print_next = input("\nContinue to check next failed job? (enter/no): ").strip().lower()
            if print_next == "no":
                break

    rerun = input("\nDo you want to rerun the failed jobs? (yes/no): ").strip().lower()
    if rerun == "yes":
        jobs = []
        for file_path in failed_jobs:
            jobs.append(rerun_job(file_path))
        print("Waiting all jobs to finish")
        [job.wait() for job in jobs]
        cprint("The jobs below failed again!", color="red")
        cprint(str([job.job_id for job in jobs if job.state not in ('DONE', 'COMPLETED')]), color="red")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--base-dir", type=str)

    main(parser.parse_args().base_dir)
