import re
import math
import json
import docker
import jinja2
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
    TEST_FILE_PATTERN,
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
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.test_enhancer.path_approx import get_mut_paths, pairwise
from swebench.test_enhancer.path_selection import select_uncovered_paths
from swebench.test_enhancer.llm_invocation import LLMInvocation

maxCYC = 10
maxNoIncreaseLimit = 3

def run_tests_and_get_coverage(container, instance, log_dir, timeout, logger):
    instance_id = instance['instance_id']
    log_dir.mkdir(parents=True, exist_ok=True)
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
    cov_report = json.loads(cov_output)
    return cov_report


def build_prompt(src_numbered, test_file, selected_paths):
    prompt_template = """
## Overview
You are an expert software engineer code assistant tasked with generating additional unit tests for a Java source file and its corresponding test file.
Your objective is to enhance both line coverage and branch coverage by adding new unit tests to the existing test suite.

### Guidelines:
1. Analyze the Code: Examine the provided source code to understand its functionality, inputs, outputs, and core logic.
2. Identify Test Cases: Develop a detailed list of test cases that will fully validate the source code and achieve 100% line coverage and branch coverage.
3. Add and Review Tests: Integrate individual tests ensuring they collectively cover all possible scenarios, including edge cases and exception handling.
4. Maintain Consistency: Ensure new tests are consistent with the existing test suite in terms of style, naming conventions, and structure. Assume new tests are part of the same suite if a test suite exists.

## Source File
Here is the source file that you will be writing tests against, called `CSVParser.java`.
We have manually added line numbers to assist in understanding the code coverage.
These line numbers are not part of the original code.

## Source File
Here is the source file that you will be writing tests against.
{{source_file_numbered}}

## Test File
Here is the file that contains the existing tests.
{{test_file}}
    """
    test_template = """
Please generate test for `{{method_name}} to cover the path
{{selected_path_for_method}}
    """
    jenv = jinja2.Environment()
    test_prompt = []
    for method, paths in selected_paths.items():
        for path in paths:
            lines_in_path = []
            for node in path:
                if node[0] == node[1]:
                    lines_in_path.append(node[0])
                else:
                    lines_in_path.extend(list(range(node[0], node[1]+1)))
            path_src = [
                src_numbered[line]
                for line in range(len(src_numbered))
                if line+1 in lines_in_path
            ]
            path_src = "\n".join(path_src)
            _test_prompt = jenv.from_string(test_template).render(method_name=method, selected_path_for_method=path_src)
            test_prompt.append(_test_prompt)
    test_prompt = """

## Methods Under Test
""" + "\n".join(test_prompt)
    user_prompt = jenv.from_string(prompt_template).render(source_file_numbered=src_numbered, test_file=test_file)
    user_prompt = user_prompt + test_prompt
    print("="*60)
    print(test_prompt)
    print("="*60)
    print(selected_paths)
    system_prompt = "You are an expert Python test-driven developer"
    return {"system": system_prompt, "user": user_prompt}


def generate_test_by_prompt_llm(prompt):
    llm_invoker =  LLMInvocation("gpt-4o-2024-08-06")
    response, prompt_token_count, response_token_count = llm_invoker.call_model(prompt)
    token_count = prompt_token_count + response_token_count
    response_list = response.split('\n')
    started = False
    codeblock = []
    for line in response_list:
        if not started and line.startswith('```'):
            started = True
            continue
        if started:
            if line.startswith('```'):
                started = False
                break
            codeblock.append(line)
    codeblock = "\n".join(codeblock)
    return response, codeblock

def add_tests_to_test_file(container, log_dir, codeblock, src_file, test_file, tst, logger):
    your_module = src_file.split('.py')[0].replace('/','.')
    codeblock = codeblock.replace('your_module', your_module)
    new_test_content = tst + '\n' + codeblock
    new_test_file = Path(log_dir / f"{test_file.replace('/','__')}" )
    new_test_file.write_text(new_test_content)
    logger.info(
        f"Generated tests written to {new_test_file}, now applying to container..."
    )
    copy_to_container(container, new_test_file, PurePosixPath(test_file))


def generate_tests(container, instance, log_dir, src_file, src, test_file, tst, timeout, logger):
    instance_id = instance['instance_id']
    ## TODO: testDeps = extractDependenciesForTestScope()
    iter, iter_no_increase = 0, 0
    path_history = dict()
    # failedTestFeedback = []
    # methodDict = Algorithm 1 (srcFile)
    # TODO: maxCYC = getMaxComplexity(methodDict)
    cur_cov_report = run_tests_and_get_coverage(container, instance, log_dir, timeout, logger)
    cur_coverage = cur_cov_report['files'][src_file]['summary']['percent_covered']
    # TODO: add conditional: iter < maxCYC
    print(f"cur_coverage: {cur_coverage}")
    while iter_no_increase < maxNoIncreaseLimit and iter < 2 and math.ceil(cur_coverage) < 100:
        selected_paths = select_uncovered_paths(cur_cov_report, src_file, src, path_history, logger)
        src_numbered = get_lined_source(src)
        prompt = build_prompt(src_numbered, tst, selected_paths)
        response, codeblock = generate_test_by_prompt_llm(prompt)
        _log_dir = log_dir / str(iter)
        _log_dir.mkdir(parents=True, exist_ok=True)
        file_output_path = _log_dir / f"new_{test_file.replace('/','__')}"
        with open(file_output_path, "w") as f:
            f.write(codeblock)
            logger.info(f"Generated tests for {src_file} written to {file_output_path}")
        add_tests_to_test_file(container, _log_dir, codeblock, src_file, test_file, tst, logger )
        new_cov_report = run_tests_and_get_coverage(container, instance, _log_dir, timeout, logger)
        new_coverage = new_cov_report['files'][src_file]['summary']['percent_covered']
        print(f"{iter} new_coverage: {new_coverage}")
        iter_no_increase = 0 if new_coverage > cur_coverage else iter_no_increase + 1
        cur_coverage = new_coverage
        iter += 1

def get_lined_source(src, range=None):
    src = src.split('\n')
    if src[-1] == '':
        src = src[:-1]
    lines = []
    i = 1
    for line in src:
        line = str(i) + ' ' + line
        if range is None or (i<=range[1] and i >= range[0]):
            lines.append(line)
        i+=1
    return lines

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
    # instance_id = test_spec.instance_id
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
    # test_files = re.findall(r'^diff --git a/(.*?) b/', instance['test_patch'], flags=re.MULTILINE)

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

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(instance['patch'] or "")
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

        def get_file_output(file_path, file_output_path):
            file_output, timed_out, total_runtime = exec_run_with_timeout(
                container, f"cat {file_path}", timeout
            )
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
            return file_output

        for src_file in src_files:
            test_file = TEST_FILE_PATTERN[instance['repo']](src_file)
            logger.info(f"Generating tests for {src_file} -> {test_file}")
            path_history = dict()
            # src = get_lined_source(file_output, (400,410))
            # import IPython; IPython.embed()
            # print(src)
            file_output_path = log_dir / f"{src_file.replace('/','__')}"
            src = get_file_output(src_file, file_output_path)
            file_output_path = log_dir / f"{test_file.replace('/','__')}"
            tst = get_file_output(test_file, file_output_path)
            generate_tests(container, instance, log_dir, src_file, src, test_file, tst, timeout, logger)

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
