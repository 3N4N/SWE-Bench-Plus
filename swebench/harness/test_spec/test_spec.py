import hashlib
import os
import re
import json
import platform
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from swebench.harness.constants import (
    DEFAULT_DOCKER_SPECS,
    KEY_INSTANCE_ID,
    LATEST,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
    SWEbenchInstance,
    UTF8,
)
from swebench.harness.dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from swebench.harness.test_spec.create_scripts import (
    make_repo_script_list,
    make_env_script_list,
    make_eval_script_list,
)


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """

    instance_id: str
    repo: str
    version: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    env_script_list: list[str]
    arch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str
    docker_specs: dict
    namespace: Optional[str]
    base_image_tag: str = LATEST
    env_image_tag: str = LATEST
    instance_image_tag: str = LATEST

    @property
    def setup_env_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list)
            + "\n"
        )

    @property
    def eval_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list)
            + "\n"
        )
        # Don't exit early because we need to revert tests at the end

    @property
    def install_repo_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list)
            + "\n"
        )

    @property
    def base_image_key(self):
        """
        If docker_specs are present, the base image key includes a hash of the specs.
        """
        if self.docker_specs != {}:
            hash_key = str(self.docker_specs)
            hash_object = hashlib.sha256()
            hash_object.update(hash_key.encode("utf-8"))
            hash_value = hash_object.hexdigest()
            val = hash_value[
                :10
            ]  # 10 characters is still likely to be unique given only a few base images will be created
            return f"sweb.base.{MAP_REPO_TO_EXT[self.repo]}.{self.arch}.{val}:{self.base_image_tag}"
        return (
            f"sweb.base.{MAP_REPO_TO_EXT[self.repo]}.{self.arch}:{self.base_image_tag}"
        )

    @property
    def env_image_key(self):
        """
        The key for the environment image is based on the hash of the environment script list.
        If the environment script list changes, the image will be rebuilt automatically.

        Note that old images are not automatically deleted, so consider cleaning up old images periodically.
        """
        hash_key = str(self.env_script_list)
        if self.docker_specs != {}:
            hash_key += str(self.docker_specs)
        hash_object = hashlib.sha256()
        hash_object.update(hash_key.encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"sweb.env.{MAP_REPO_TO_EXT[self.repo]}.{self.arch}.{val}:{self.env_image_tag}"

    @property
    def instance_image_key(self):
        key = f"sweb.eval.{self.arch}.{self.instance_id.lower()}:{self.instance_image_tag}"
        if self.is_remote_image:
            key = f"{self.namespace}/{key}".replace("__", "_1776_")
        return key

    @property
    def is_remote_image(self):
        return self.namespace is not None

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"sweb.eval.{self.instance_id}"
        return f"sweb.eval.{self.instance_id.lower()}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(
            self.platform,
            self.arch,
            self.language,
            **{**DEFAULT_DOCKER_SPECS, **self.docker_specs},
        )

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(
            self.platform,
            self.arch,
            self.language,
            self.base_image_key,
            **{**DEFAULT_DOCKER_SPECS, **self.docker_specs},
        )

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.language, self.env_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")


def get_test_specs_from_dataset(
    dataset: Union[list[SWEbenchInstance], list[TestSpec]],
    namespace: Optional[str] = None,
    instance_image_tag: str = LATEST,
) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEbenchInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(
        map(
            lambda x: make_test_spec(x, namespace, instance_image_tag),
            cast(list[SWEbenchInstance], dataset),
        )
    )

import ast

def extract_nodes(file_content):
    tree = ast.parse(file_content)

    def _is_func(node):  # regular or async function
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    def _class_info(cls_node: ast.ClassDef):
        methods = []
        nested_classes = []
        for n in cls_node.body:
            if _is_func(n):
                methods.append(n.name)
            elif isinstance(n, ast.ClassDef):
                nested_classes.append(_class_info(n))
        methods = [ f'{cls_node.name}.{method}' for method in methods ]
        nested_classes = [  f'{cls_node.name}.{method}' for method in nested_classes ]
        methods.extend(nested_classes)
        return methods

    classes= []
    functions= []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.extend(_class_info(node))
        elif _is_func(node):
            functions.append(node.name)

    return classes + functions

def extract_test_headers(repo, test_file, test_content):
    if repo == 'django/django':
        test_module = '.'.join(test_file.replace('.py','').split('/')[1:])
        nodes = extract_nodes(test_content)
        tests = []
        for node in nodes:
            if '.' in node:
                nodesplit = node.split('.')
                head, tail = '.'.join(nodesplit[:-1]), nodesplit[-1]
                tests.append(f"{tail} ({test_module}.{head})")
            else:
                tests.append(f"{node} ({test_module})")
        return tests
    elif repo == 'sympy/sympy':
        tests = []
        nodes = extract_nodes(test_content)
        for node in nodes:
            tests.append(f"{node}")
        return tests
    elif repo in ['astropy/astropy', 'matplotlib/matplotlib', 'mwaskom/seaborn',
                  'pydata/xarray', 'pytest-dev/pytest', 'scikit-learn/scikit-learn',
                  'sphinx-doc/sphinx', 'psf/requests' ]:
        nodes = extract_nodes(test_content)
        tests = []
        for node in nodes:
            if '.' in node:
                nodesplit = node.split('.')
                head, tail = '.'.join(nodesplit[:-1]), nodesplit[-1]
                tests.append(f"{test_file}::{head}::{tail}")
            else:
                tests.append(f"{test_file}::{node}")
        return tests
    else:
        print("ERROR: extract_test_headers: repo not recognized")
        return []

def get_node(repo, test_file, entry):
    if repo == 'django/django':
        test_module = '.'.join(test_file.replace('.py','').split('/')[1:])
        pattern = f"([\\d\\w_\\.]+) \\({test_module}.([\\d\\w_\\.]+)\\)"
        match = re.search(pattern, entry)
        if not match:
            pattern = f"([\\d\\w_\\.]+) \\({test_module}\\)"
            match = re.search(pattern, entry)
            if not match:
                print(f"get_node: No match: {entry} -> {test_module}")
                return None
            node = match.groups()
            return node
        tail, head = match.groups()
        node = f"{head}.{tail}"
        return node
    elif repo == 'sympy/sympy':
        return entry
    elif repo in ['astropy/astropy', 'matplotlib/matplotlib', 'mwaskom/seaborn',
                  'pydata/xarray', 'pytest-dev/pytest', 'scikit-learn/scikit-learn',
                  'sphinx-doc/sphinx', 'psf/requests' ]:
        node = ".".join(entry.split("::")[1:])
        return node
    else: return None

def make_test_spec(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance.get("version")
    base_commit = instance["base_commit"]
    problem_statement = instance.get("problem_statement")
    hints_text = instance.get("hints_text")  # Unused
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    ## TODO: 
        ## read the json file containing tests
        ## add those tests to pass_to_pass & fail_to_pass test lists
    # if isinstance(fail_to_pass, list):
    #     fail_to_pass.append("astropy/io/fits/tests/test_connect.py::test_testenhancer_failing")

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    docker_specs = specs.get("docker_specs", {})

    repo_script_list = make_repo_script_list(
        specs, repo, repo_directory, base_commit, env_name
    )
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    test_files = re.findall(r'^diff --git a/(.*?) b/', instance['test_patch'], flags=re.MULTILINE)

    # def get_modified_files_from_patch(diff_text: str):
    #     modified_files = []
    #     for header in re.finditer(r"^diff --git a/(.+?) b/\1", diff_text, re.MULTILINE):
    #         file_path = header.group(1)
    #         # Ensure file is not marked as new or deleted
    #         context_start = diff_text.find(header.group(0))
    #         context = diff_text[context_start: context_start + 200]  # look ahead
    #         if "new file mode" not in context and "deleted file mode" not in context:
    #             modified_files.append(file_path)
    #     return modified_files
    # test_files = get_modified_files_from_patch(instance['test_patch'])

    # print("-"*10, repo)
    # for test_file in test_files:
    #     testgen_patch_dir = Path(f"logs/test_enhancer/TE_2/{instance['instance_id']}")
    #     test_file_path = testgen_patch_dir / f"{test_file.replace('/','__')}"
    #     if test_file_path.is_file():
    #         test_content = test_file_path.read_text(encoding=UTF8)
    #         tests = extract_test_headers(repo, test_file, test_content)
    #         for test in tests:
    #             node = get_node(repo, test_file, test)
    #             print(test, '->', node)
    #     else:
    #         print(f"File not found: {repo} -> {test_file}")

    add_test_enhancer_patches = False

    if add_test_enhancer_patches:
        HEREDOC_DELIMITER = "EOF_114329324912"
        update_test_file_command = list()
        new_fail_to_pass = list()
        TE_Id = os.environ['TE_ID']
        testgen_patch_dir = Path(f"logs/test_enhancer/{TE_Id}/{instance['instance_id']}")
        run_instance_log_path = testgen_patch_dir / "run_instance.log"
        run_instance_log = run_instance_log_path.read_text(encoding=UTF8)
        coverage_reports = [ line for line in run_instance_log.split('\n') if "_coverage:" in line ]
        cur_cov = 0
        max_cov = 0
        max_iter = -1
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - (\d+ )?(?:cur|new)_coverage: (\d+\.\d+)"
        for report in coverage_reports:
            match = re.search(pattern, report)
            if match:
                iter, cov = match.groups()
                if iter is None:
                    cur_cov = float(cov)
                else:
                    iter, cov = int(iter), float(cov)
                    if cov > max_cov:
                        max_cov = cov
                        max_iter = iter
        # print(cur_cov)
        # print(max_cov)
        # print(max_iter)
        if max_cov > cur_cov:
            add_test_enhancer_patches = True
        else:
            add_test_enhancer_patches = False
            print(f"{instance_id}: Not adding TestEnhancer patches")
        test_files = []

    if add_test_enhancer_patches:
        for test_file in test_files:
            max_cov = 0
            max_iter = -1
            for i in range(100):
                file_output_dir = testgen_patch_dir / str(i)
                file_output_path = file_output_dir / f"{test_file.replace('/','__')}"
                if file_output_dir.is_dir():
                    cov_output_path = file_output_dir / 'coverage.json'
                    with cov_output_path.open("r", encoding=UTF8) as f:
                        cov_report = json.load(f)
                    this_cov = cov_report['totals']['percent_covered']
                    if this_cov > max_cov:
                        max_cov = this_cov
                        max_iter = i
                        # print(max_cov, max_iter, this_cov)
            if max_iter != -1:
                file_output_path = testgen_patch_dir / str(max_iter) / f"{test_file.replace('/','__')}"
                if file_output_path.is_file():
                    print(file_output_path)
                    test_content = file_output_path.read_text(encoding=UTF8)
                    update_test_file_command.append(
                        f"cat > {test_file} <<'{HEREDOC_DELIMITER}'\n{test_content}\n{HEREDOC_DELIMITER}"
                    )
                    new_test_file = testgen_patch_dir / str(max_iter) / f"new_{test_file.replace('/','__')}"
                    new_test_content = new_test_file.read_text(encoding=UTF8)
                    new_fail_to_pass.extend( extract_test_headers(repo, test_file, new_test_content))

        eval_script_list = eval_script_list[:-4] + update_test_file_command + eval_script_list[-4:]
        fail_to_pass.extend(new_fail_to_pass)

    # fail_to_pass.append('astropy/io/fits/tests/test_connect.py::test_testenhancer_failing')

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )
