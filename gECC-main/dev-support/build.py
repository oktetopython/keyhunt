#!/usr/bin/env python
import argparse
import subprocess as sp
from pathlib import Path
import shutil
import sys
import os


def run(cmd, env=None):
    result = sp.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--clean', action='store_true')
    parser.add_argument('-A', '--arch', required=True)
    parser.add_argument('-T', '--test')
    parser.add_argument('-F', '--filter')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-D', '--debug', action='store_true')
    group.add_argument('-R', '--release', action='store_true')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print(project_root)
    if args.debug:
        build_root = project_root / 'build' / 'Debug'
    else:
        build_root = project_root / 'build' / \
            'Release'

    if args.clean:
        shutil.rmtree(build_root, ignore_errors=True)
    if not build_root.exists():
        if args.debug:
            # clang currently does not support CUDA 11
            sub_env = os.environ.copy()
            sub_env['PATH'] = ':'.join(
                ['/usr/local/cuda-10.1', *sub_env['PATH'].split(':')])
            run([
                'cmake',
                '-S{}'.format(project_root),
                '-B{}'.format(build_root),
                '-DCMAKE_BUILD_TYPE=Debug',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=YES',
                '-DCMAKE_CXX_COMPILER=clang++',
                '-DCMAKE_CUDA_COMPILER=clang++',
                '-DCMAKE_CUDA_ARCHITECTURES={}'.format(args.arch)
            ], env=sub_env)
        else:
            sub_env = os.environ.copy()
            sub_env['PATH'] = ':'.join(
                ['/usr/local/cuda-10.1', *sub_env['PATH'].split(':')])
            run([
                'cmake',
                '-S{}'.format(project_root),
                '-B{}'.format(build_root),
                '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=YES',
                '-DCMAKE_CUDA_ARCHITECTURES={}'.format(args.arch)
            ], env=sub_env)
    build_command = ['cmake', '--build', str(build_root)]
    if args.test:
        build_command += ['--target', args.test]
    run(build_command)
    if args.test:
        test_command = [build_root / 'test' / args.test]
        if args.filter:
            test_command += ['--gtest_filter={}'.format(args.filter)]
        run(test_command)

# if [[ "$#" -gt 1 ]]; then
#   host=$1
#   executable=$2
#   rsync -aPz ${build_dir}/${executable} $host:/tmp \
#     && ssh -tt $host /tmp/$(basename ${executable}) $@
# fi
