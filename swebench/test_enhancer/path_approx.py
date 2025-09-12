import re
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


import copy
from queue import Queue
from py2cfg import CFGBuilder

def pairwise(iterable):
    import itertools
    a,b = itertools.tee(iterable)
    next(b, None)
    return zip(a,b)

def get_nedges(cfg):
    start = cfg.entryblock
    q = Queue()
    n_edges = 0
    res = []
    v = list()
    q.put(start)
    while not q.empty():
        curr = q.get()
        n_edges += len(curr.exits)
        res.append((curr.at(), curr.end()))
        successors = [ block.target for block in curr.exits ]
        for x in successors:
            if x not in v:
                v.append(x)
                q.put(x)
    return n_edges


def _get_lineno_path(path):
    line_path = []
    for block in path:
        start, end = block.at(), block.end()
        line_path.append((start,end))
    return line_path

def _get_lineno_paths(paths):
    line_paths = []
    for path in paths:
        line_path = _get_lineno_path(path)
        line_paths.append(line_path)
    return line_paths

def explore_paths(cfg, logger, flag=False):
    start = cfg.entryblock
    n_edges = get_nedges(cfg)
    paths = []
    visited = set()
    q = Queue()
    q.put([start])
    while not q.empty():
        # for i in range(q.qsize()):
            current_path = q.get()
            last_block = current_path[-1]
            successors = [ block.target for block in last_block.exits ]
            # successors = list({(item.at(),item.end()): item for item in successors}.values())
            for successor in successors:
                if successor in current_path: continue
                next_path = copy.copy(current_path)
                next_path.append(successor)
                if len(successor.exits) > 0:    # if successor is NOT a terminal node
                    line_path = _get_lineno_path(next_path)
                    # print(line_path)
                    q.put(next_path)
                else:
                    edges = [(x,y) for x,y in pairwise(next_path)]
                    if len( [ edge for edge in edges if edge not in visited ] ) > 0:
                        paths.append(next_path)
                        visited.update(set(edges))
                    if flag:
                        if len(visited) >= n_edges - 1:
                            line_paths = _get_lineno_paths(paths)
                            logger.info(f"Visited edges: {len(visited)}")
                            return paths, line_paths
                        else:
                            v = [ (x.at(), y.at()) for x,y in visited ]
                            logger.info(v)
                            logger.info(f"Visited edges: {len(visited)} / {n_edges}")
    logger.info(f"Visited edges: {len(visited)}")
    line_paths = _get_lineno_paths(paths)
    return paths, line_paths

def get_mut_paths(src, name, logger):
    methodDict = {}
    CFG_f = CFGBuilder(True).build_from_src(name, src)
    for name, CFG_m in CFG_f.functioncfgs.items():
        # if name in [ '_decode_mixins', 'read_table_fits', '_encode_mixins']: continue
        mut_start, mut_end = CFG_m.lineno, CFG_m.end_lineno
        logger.info(f"{name}: {mut_start}, {mut_end}, {get_nedges(CFG_m)}")
        paths, line_paths = explore_paths(CFG_m, logger)
        methodDict[name] = line_paths
    return methodDict


def _main():
    log_dir = Path("/tmp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    instance_id = "XXX"
    logger = setup_logger(instance_id, log_file)
    src = Path('tmp.py').read_text(encoding=UTF8)
    methodDict = get_mut_paths(src, "tmp", logger)
    print(methodDict)

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

        # eval_file = Path(log_dir / "eval.sh")
        # eval_file.write_text(test_spec.eval_script)
        # logger.info(
        #     f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        # )
        # copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        srcs = {}
        methodDicts = {}
        for src_file in src_files:
            file_output, timed_out, total_runtime = exec_run_with_timeout(
                container, f"cat {src_file}", timeout
            )
            srcs[src_file] = file_output
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

        for src_file, src in srcs.items():
            logger.info(f"Path approximation for {src_file}")
            methodDict = get_mut_paths(src, name=src_file, logger=logger)
            methodDicts[src_file] = methodDict
            logger.info(f"Approximate paths: {methodDict}")

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
    # _main()
