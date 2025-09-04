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
    UTF8,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
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
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.test_spec import extract_test_headers, get_node
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.test_enhancer.path_approx import get_mut_paths, pairwise
from swebench.test_enhancer.path_selection import select_uncovered_paths
from swebench.test_enhancer.llm_invocation import LLMInvocation

def run_tests(container, instance, log_dir, timeout, logger, patch_coverage=False):
    instance_id = instance['instance_id']
    log_dir.mkdir(parents=True, exist_ok=True)
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']]

    HEREDOC_DELIMITER = "EOF_114329324913"
    # reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]][
                "test_cmd"
            ],
            *get_test_directives(instance),
        ]
    )
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if patch_coverage and instance['repo'] == "django/django" and float(instance['version']) < 4.0 :
        coverage_command = "coverage json -o coverage.json"
        coverage_install = "python --version && python -m pip install -U pip\npip install -U 'coverage==6.2'"
        coverage_patch = '''
diff --git a/coverage/jsonreport.py b/coverage/jsonreport.py
index 43edc4520..7ca468e32 100644
--- a/coverage/jsonreport.py
+++ b/coverage/jsonreport.py
@@ -102,4 +102,17 @@ def report_one_file(self, coverage_data, analysis):
                 'covered_branches': nums.n_executed_branches,
                 'missing_branches': nums.n_missing_branches,
             })
+            reported_file['executed_branches'] = list(
+                [-1, -2] # _convert_branch_arcs(analysis.executed_branch_arcs())
+            )
+            reported_file['missing_branches'] = list(
+                _convert_branch_arcs(analysis.missing_branch_arcs())
+            )
         return reported_file
+
+
+def _convert_branch_arcs(branch_arcs):
+    """Convert branch arcs to a list of two-element tuples."""
+    for source, targets in branch_arcs.items():
+        for target in targets:
+            yield source, target if target != -1 else 0
        '''
        coverage_apply_patch_command = "\n".join(
            "pushd $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')",
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{coverage_patch}\n{HEREDOC_DELIMITER}"
            "popd",
        )
    elif patch_coverage and instance['repo'] == "django/django":
        coverage_command = "coverage json -o coverage.json"
        coverage_install = "python --version\npip install -U coverage"
        coverage_apply_patch_command = ""
    elif instance['repo'] == 'sympy/sympy':
        coverage_command = "coverage json -o coverage.json"
        coverage_install = "python --version\npip install -U coverage\ncoverage --version"
        coverage_apply_patch_command = ""
    elif instance['repo'] == 'sphinx-doc/sphinx':
        coverage_command = ""
        coverage_install = ""
        coverage_apply_patch_command = "sed -i -e 's/ pytest / pytest --cov --cov-branch --cov-report json /g' tox.ini"
    else:
        coverage_command = ""
        coverage_install = ""
        coverage_apply_patch_command = ""
    eval_commands += [
        # reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
        coverage_install,
        coverage_apply_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        coverage_command,
        f": '{END_TEST_OUTPUT}'",
        # reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]

    test_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_commands) + "\n"

    eval_file = Path(log_dir / "run_tests.sh")
    eval_file.write_text(test_script)
    logger.info(
        f"Testrun script for {instance_id} written to {eval_file}; copying to container..."
    )
    copy_to_container(container, eval_file, PurePosixPath("/run_tests.sh"))

    # Run eval script, write output to logs
    test_output, timed_out, total_runtime = exec_run_with_timeout(
        container, "/bin/bash /run_tests.sh", timeout
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
    return test_output_path

def get_coverage(container, instance, log_dir, timeout, logger):
    instance_id = instance['instance_id']
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

import ast
from pathlib import Path
from typing import Dict, Set, Tuple, Union, Iterable

def parse_targets(targets: Iterable[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    funcs: Set[str] = set()
    methods: Dict[str, Set[str]] = {}
    for t in targets:
        if "." in t:
            cls, meth = t.split(".", 1)
            methods.setdefault(cls, set()).add(meth)
        else:
            funcs.add(t)
    return funcs, methods

def remove_functions_from_ast(tree: ast.Module,
                              top_funcs: Set[str],
                              class_methods: Dict[str, Set[str]]) -> ast.Module:
    def is_func(node): return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    def prune_class(cnode: ast.ClassDef) -> ast.ClassDef:
        new_body = []
        for n in cnode.body:
            # remove direct methods whose names match
            if is_func(n) and n.name in class_methods.get(cnode.name, set()):
                continue
            # recurse into nested classes
            if isinstance(n, ast.ClassDef):
                new_body.append(prune_class(n))
            else:
                new_body.append(n)
        cnode.body = new_body
        return cnode

    new_body = []
    for node in tree.body:
        # remove top-level functions
        if is_func(node) and node.name in top_funcs:
            continue
        # prune class methods
        if isinstance(node, ast.ClassDef):
            new_body.append(prune_class(node))
        else:
            new_body.append(node)
    tree.body = new_body
    return tree

def remove_functions_from_file(src, targets):
    tree = ast.parse(src)
    top_funcs, class_methods = parse_targets(targets)
    tree = remove_functions_from_ast(tree, top_funcs, class_methods)

    try:
        new_src = ast.unparse(tree)  # Python 3.9+
    except AttributeError:
        import astor  # pip install astor
        new_src = astor.to_source(tree)
    new_src_split = new_src.split('\n')

    new_src = []
    for i in range(len(new_src_split)):
        if new_src_split[i].startswith('class'):
            if (i != len(new_src_split) - 1) and (len(new_src_split[i+1])>0) and (new_src_split[i+1][0]==' '):
                new_src.append(new_src_split[i])
        else:
            new_src.append(new_src_split[i])

    new_src = "\n".join(new_src)
    return new_src

def get_successful_tests(container, dataset_name, split, instance,
               test_spec, test_file, test_content, test_output_path,
               log_dir, timeout, logger):
    instance_id = instance['instance_id']
    repo = instance['repo']
    test_headers = extract_test_headers(repo, test_file, test_content)
    test_spec.FAIL_TO_PASS.extend(test_headers)
    # print(f"fail -> pass: {test_spec.FAIL_TO_PASS}")
    predictions = get_predictions_from_file('gold', dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
    pred = predictions[instance_id]
    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=test_output_path,
        include_tests_status=True,
    )
    tests_success = report[instance_id]['tests_status']['FAIL_TO_PASS']['success']
    tests_failure = report[instance_id]['tests_status']['FAIL_TO_PASS']['failure']
    new_tests_success = [ test for test in test_headers if test in tests_success ]
    new_tests_failure = [ test for test in test_headers if test in tests_failure ]
    to_remove = [ get_node(repo, test_file, test) for test in new_tests_failure ]
    to_remove = [ entry for entry in to_remove if entry is not None ]
    logger.info(f"Remove targets: {to_remove}")
    if to_remove is not None and len(to_remove) > 0:
        correct_test_content = remove_functions_from_file(test_content, to_remove)
    else:
        correct_test_content = test_content
    return correct_test_content

def build_prompt(src_numbered, test_file, selected_paths, log_dir):
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
Here is the source file that you will be writing tests against.
We have manually added line numbers to assist in understanding the code coverage.
These line numbers are not part of the original code.

-----------------------------------------------------------
{{source_file_numbered}}
-----------------------------------------------------------

## Test File
Here is the file that contains the existing tests.
-----------------------------------------------------------
{{test_file}}
-----------------------------------------------------------
    """
    test_template = """
Please generate test for `{{method_name}} to cover the path
{{selected_path_for_method}}
-----------------------------------------------------------
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
    user_prompt = jenv.from_string(prompt_template).render(source_file_numbered="\n".join(src_numbered), test_file=test_file)
    user_prompt = user_prompt + test_prompt
    # print("="*60)
    # print(test_prompt)
    # print("="*60)
    # print(selected_paths)
    system_prompt = "You are an expert Python test-driven developer"
    file_output_path = log_dir / f"prompt.txt"
    with open(file_output_path, "w") as f:
        f.write(user_prompt)
    return {"system": system_prompt, "user": user_prompt}


def generate_test_by_prompt_llm(prompt, log_dir, iter):
    llm_invoker =  LLMInvocation("gpt-4o-2024-08-06")
    response, prompt_token_count, response_token_count = llm_invoker.call_model(prompt)
    token_count = prompt_token_count + response_token_count
    response_path = log_dir / "response.txt"
    with open(response_path, "w") as f:
        f.write(response)

    if False:
        response = """
This is the dummy response
```python
def test_pass():
    assert 2 == 1+1
def test_fail():
    assert 2 == 1-1
```
        """

        if iter == 1:
            response = '''
To enhance the test coverage for the `read_table_fits` function, we need to focus on the specific paths and conditions within the function. The paths we want to cover are:

1. When the input is an `HDUList` and contains multiple tables.
2. When the input is an `HDUList` and contains a single table.
3. When the input is an `HDUList` but contains no tables, which should raise a `ValueError`.

Here are the test cases that cover these scenarios:

```python
import pytest
import numpy as np
from astropy.io.fits import HDUList, BinTableHDU, PrimaryHDU
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

def test_read_table_fits_multiple_tables(tmp_path):
    # Create an HDUList with multiple tables
    data1 = np.array([(1, 'a'), (2, 'b')], dtype=[('col1', int), ('col2', 'U1')])
    data2 = np.array([(3, 'c'), (4, 'd')], dtype=[('col3', int), ('col4', 'U1')])
    hdu1 = BinTableHDU(data1, name='FIRST')
    hdu2 = BinTableHDU(data2, name='SECOND')
    hdulist = HDUList([PrimaryHDU(), hdu1, hdu2])

    # Write to a temporary file
    filename = tmp_path / "test_multiple_tables.fits"
    hdulist.writeto(filename, overwrite=True)

    # Read the table without specifying HDU
    with pytest.warns(AstropyUserWarning, match="hdu= was not specified but multiple tables are present"):
        table = Table.read(filename)
    assert np.all(table['col1'] == data1['col1'])
    assert np.all(table['col2'] == data1['col2'])

def test_read_table_fits_single_table(tmp_path):
    # Create an HDUList with a single table
    data = np.array([(1, 'a'), (2, 'b')], dtype=[('col1', int), ('col2', 'U1')])
    hdu = BinTableHDU(data, name='ONLY')
    hdulist = HDUList([PrimaryHDU(), hdu])

    # Write to a temporary file
    filename = tmp_path / "test_single_table.fits"
    hdulist.writeto(filename, overwrite=True)

    # Read the table without specifying HDU
    table = Table.read(filename)
    assert np.all(table['col1'] == data['col1'])
    assert np.all(table['col2'] == data['col2'])

def test_read_table_fits_no_table(tmp_path):
    # Create an HDUList with no tables
    hdulist = HDUList([PrimaryHDU()])

    # Write to a temporary file
    filename = tmp_path / "test_no_table.fits"
    hdulist.writeto(filename, overwrite=True)

    # Attempt to read the table should raise ValueError
    with pytest.raises(ValueError, match="No table found"):
        Table.read(filename)
```

### Explanation:

1. **`test_read_table_fits_multiple_tables`**: This test creates an `HDUList` with multiple tables and checks if the first table is read by default, emitting a warning about multiple tables being present.

2. **`test_read_table_fits_single_table`**: This test creates an `HDUList` with a single table and verifies that the table is read correctly without any warnings.

3. **`test_read_table_fits_no_table`**: This test creates an `HDUList` with no tables and ensures that attempting to read it raises a `ValueError` with the message "No table found".

These tests should cover the specified paths in the `read_table_fits` function, ensuring that the function behaves as expected in these scenarios.
            '''

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

def write_to_test_file(container, log_dir, test_file, test_content, logger):
    new_test_file = Path(log_dir / f"{test_file.replace('/','__')}" )
    new_test_file.write_text(test_content)
    logger.info(
        f"Writing to test file {new_test_file}, now applying to container..."
    )
    copy_to_container(container, new_test_file, PurePosixPath(test_file))

def add_tests_to_test_file(container, log_dir, codeblock, src_file, test_file, test_content, logger):
    file_output_path = log_dir / f"new_{test_file.replace('/','__')}"
    with open(file_output_path, "w") as f:
        f.write(codeblock)
        logger.info(f"Generated tests for {src_file} written to {file_output_path}")
    your_module = src_file.split('.py')[0].replace('/','.')
    codeblock = codeblock.replace('your_module', your_module)
    new_test_content = test_content + '\n' + codeblock
    # new_test_content = codeblock
    write_to_test_file(container, log_dir, test_file, new_test_content, logger)
    return new_test_content

def reset_test_file(container, log_dir, test_file, test_content, logger):
    write_to_test_file(container, log_dir, test_file, test_content, logger)

def generate_tests(container, dataset_name, split, instance, test_spec, log_dir, src_file, src, test_file, tests, timeout, logger):
    instance_id = instance['instance_id']
    ## TODO: testDeps = extractDependenciesForTestScope()
    iter, iter_no_increase = 0, 0
    path_history = dict()
    # failedTestFeedback = []
    # methodDict = Algorithm 1 (srcFile)
    # TODO: maxCYC = getMaxComplexity(methodDict)
    maxCYC = 10
    maxNoIncreaseLimit = 3
    test_output_path = run_tests(container, instance, log_dir, timeout, logger, patch_coverage=True)
    cur_cov_report = get_coverage(container, instance, log_dir, timeout, logger)
    try:
        cur_coverage = cur_cov_report['files'][src_file]['summary']['percent_covered']
    except KeyError:
        logger.error(f"Coverage of src file {src_file} not found")
        return
    logger.info(f"cur_coverage: {cur_coverage}")
    while iter_no_increase < maxNoIncreaseLimit and iter < maxCYC and math.ceil(cur_coverage) < 100:
        # logger.info(src)
        _log_dir = log_dir / str(iter)
        _log_dir.mkdir(parents=True, exist_ok=True)
        selected_paths = select_uncovered_paths(cur_cov_report, src_file, src, path_history, logger)
        src_numbered = get_lined_source(src)
        prompt = build_prompt(src_numbered, tests, selected_paths, _log_dir)
        response, codeblock = generate_test_by_prompt_llm(prompt, _log_dir, iter)
        new_tests = add_tests_to_test_file(container, _log_dir, codeblock, src_file, test_file, tests, logger )
        test_output_path = run_tests(container, instance, _log_dir, timeout, logger)
        new_codeblock = get_successful_tests(container, dataset_name, split, instance, test_spec, test_file, codeblock, test_output_path, log_dir, timeout, logger)

        new_tests = add_tests_to_test_file(container, _log_dir, new_codeblock, src_file, test_file, tests, logger )
        test_output_path = run_tests(container, instance, _log_dir, timeout, logger)
        new_cov_report = get_coverage(container, instance, _log_dir, timeout, logger)
        new_coverage = new_cov_report['files'][src_file]['summary']['percent_covered']
        logger.info(f"{iter} new_coverage: {new_coverage}")
        # print(new_tests)

        if new_coverage <= cur_coverage:
            reset_test_file(container, _log_dir, test_file, tests, logger)
        else:
            tests = new_tests
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
    test_files = re.findall(r'^diff --git a/(.*?) b/', instance['test_patch'], flags=re.MULTILINE)

    def get_modified_files(diff_text: str):
        modified_files = []
        for header in re.finditer(r"^diff --git a/(.+?) b/\1", diff_text, re.MULTILINE):
            file_path = header.group(1)
            # Ensure file is not marked as new or deleted
            context_start = diff_text.find(header.group(0))
            context = diff_text[context_start: context_start + 200]  # look ahead
            if "new file mode" not in context and "deleted file mode" not in context:
                modified_files.append(file_path)
        return modified_files
    test_files = get_modified_files(instance['test_patch'])

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

        # Copy golden patch and test patch as patch file to container
        patch_content = instance['patch'] + '\n' + instance['test_patch']
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

        def match_test_file(src_file, test_files):
            src_tail = src_file.split('/')[-1].split('.py')[0]
            if len(test_files) == 1: return test_files[0]
            for test_file in test_files:
                test_tail = test_file.split('/')[-1].split('.py')[0]
                if test_tail == f'test_{src_tail}':
                    return test_file
        for src_file in src_files:
            test_file = match_test_file(src_file, test_files)
            if test_file is None: continue
            print(f"Generating tests for {src_file} -> {test_file}")
            path_history = dict()
            file_output_path = log_dir / f"{src_file.replace('/','__')}"
            src = get_file_output(src_file, file_output_path)
            file_output_path = log_dir / f"{test_file.replace('/','__')}"
            tests = get_file_output(test_file, file_output_path)
            generate_tests(container, dataset_name, split, instance, test_spec, log_dir, src_file, src, test_file, tests, timeout, logger)

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
