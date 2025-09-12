import re
import json
import docker
import platform
import traceback

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    TESTENHANCER_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.test_enhancer.path_approx import get_mut_paths, pairwise
from swebench.harness.run_evaluation import GIT_APPLY_CMDS

MAX_SELECTED_CONST = 10

def select_uncovered_paths(cov_report, src_file, src, path_history, logger):
    # Get index of relevant file
    cov_files = list(cov_report['files'].keys())
    for file in cov_files:
        if file == src_file:
            key_file = file
            break
    missed_lines = cov_report['files'][key_file]['missing_lines']
    missed_branches = cov_report['files'][key_file]['missing_branches']

    # print(missed_lines)
    # print(missed_branches)

    logger.info(f"Approximating paths for src file: {src_file}")
    methodDict = get_mut_paths(src, src_file, logger)
    # print(methodDict)

    selected_paths = {}
    for method, paths in methodDict.items():
        # if method != 'is_fits': continue
        # print(method)
        # print(paths)
        # if method == 'read_table_fits':
        #     logger.info(paths)
        logger.info(f"Selecting paths for method: {method}")
        candidate_paths = []
        path_history.setdefault(method, {})
        for path in paths:
            cnt_missed_lines = 0
            cnt_missed_branches = 0
            for line in missed_lines:
                for edge in path:
                    if line >= edge[0] and line <= edge[1]:
                        cnt_missed_lines+=1
            for a,b in pairwise(path):
                for branch in missed_branches:
                    if a[0]<=branch[0] and a[1]>=branch[0] and b[0]<=branch[1] and b[1]>=branch[1]:
                        cnt_missed_branches += 1
            missed_score = cnt_missed_lines + cnt_missed_branches
            if missed_score <= 0: continue
            selected_count = path_history[method].get(tuple(path), 0)
            if selected_count >= MAX_SELECTED_CONST: continue
            candidate_paths.append((path,missed_score,selected_count))
        # print(f"candidate_paths: {candidate_paths}")

        if len(candidate_paths) == 0:
            logger.info(f"Zero uncovered paths for method: {method} - skipping")
            continue

        ## Exploitation: Pick highest missed path
        highest_path = max(candidate_paths, key=lambda o: o[1])[0]
        highest_path = tuple(highest_path)
        # print(f"highest_path: {highest_path}")
        ## FIXME: path_history[method][highest_path] += 1
        path_history[method][highest_path] = path_history[method].get(highest_path,0) + 1
        selected_paths.setdefault(method, []).append(highest_path)

        ## Exploration: Pick least selected path
        least_path = min(candidate_paths, key=lambda o: -o[2])[0]
        least_path = tuple(least_path)
        # print(f"least_path: {least_path}")
        if least_path != highest_path:
            ## FIXME: path_history[method][least_path] += 1
            path_history[method][least_path] = path_history[method].get(least_path,0) + 1
            selected_paths.setdefault(method, []).append(least_path)
        logger.info(f"Selected paths for method: {method}\n{selected_paths}")

    return selected_paths


def main(
    instance_id,
    dataset_name,
    split,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None,
    namespace: str | None,
    rewrite_reports: bool,
    instance_image_tag: str = "latest",
    report_dir: str = ".",
):
    log_dir = TESTENHANCER_LOG_DIR / run_id / instance_id

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    dataset = load_swebench_dataset(dataset_name, split)
    dataset = [ i for i in dataset if i[KEY_INSTANCE_ID] == instance_id ]
    assert len(dataset) == 1
    instance = dataset[0]
    src_files = re.findall(r'^diff --git a/(.*?) b/', instance['patch'], flags=re.MULTILINE)

    test_spec = make_test_spec(
        instance, namespace=namespace, instance_image_tag=instance_image_tag
    )

    container = None
    try:
        container = build_container(
            test_spec, client, run_id, logger, rm_image, force_rebuild
        )
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # NOTE: apply gold and test patches
        patch_content = instance['patch'] + instance['test_patch']
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(patch_content)
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container (TODO: FIX THIS)
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                logger,
            )

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        cov_output, timed_out, total_runtime = exec_run_with_timeout(
            container, "cat coverage.json", timeout
        )
        cov_output_path = log_dir / "coverage.json"
        with open(cov_output_path, "w") as f:
            f.write(cov_output)
            logger.info(f"Coverage output for {instance_id} written to {cov_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Cat coverage timed out after {timeout} seconds.",
                    logger,
                )

        # changes to dictionary in a function is changed in the main dict
        # i.e., dicts are passed by ref
        selected_paths = dict()
        for src_file in src_files:
            path_history = dict()
            file_output, timed_out, total_runtime = exec_run_with_timeout(
                container, f"cat {src_file}", timeout
            )
            file_output_path = log_dir / f"{src_file.replace('/','__')}"
            with open(file_output_path, "w") as f:
                f.write(file_output)
                logger.info(f"File output for {instance_id} written to {file_output_path}")
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Cat coverage timed out after {timeout} seconds.",
                        logger,
                    )
            cov_report = json.loads(cov_output)
            _selected_paths = select_uncovered_paths(cov_report, src_file, file_output,
                                                     path_history, logger)
            selected_paths[src_file] = _selected_paths
        print('-'*60)
        print(selected_paths)
        print('-'*60)

    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Path approximation with static analysis",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_name",
        default="SWE-bench/SWE-bench",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )

    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        action='store_true',
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", action='store_true', help="Clean images above cache level"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default="swebench",
        help='Namespace for images. (use "none" to use no namespace)',
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--rewrite_reports",
        action='store_true',
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default=".", help="Directory to write reports to"
    )
    parser.add_argument(
        "--instance_id", type=str, required=True, help="Instance ID",
    )
    args = parser.parse_args()

    # run instances locally
    if platform.system() == "Linux":
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.open_file_limit, args.open_file_limit))
    client = docker.from_env()

    main(
        args.instance_id,
        args.dataset_name,
        args.split,
        False,
        args.force_rebuild,
        client,
        args.run_id,
        args.timeout,
        args.namespace,
        args.rewrite_reports,
        args.instance_image_tag,
        args.report_dir,
    )
