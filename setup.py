import setuptools
import os
import subprocess
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources, include_dirs):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        include_dirs=[os.path.join(*module.split('.'), src) for src in include_dirs],
    )
    return cuda_ext


def get_git_commit_number():
    if not os.path.exists('.git'):
        return ''

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return "+" + git_commit_number

def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == "__main__":
    version = '0.1.0%s' % get_git_commit_number()
    write_version_to_file(version, 'compute/version.py'),
    setuptools.setup(
        version = version,
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules = [
            make_cuda_ext(
                name='vector_add',
                module='compute.op.vector_compute',
                sources=[
                    'src/vector_add.cpp',
                    'src/vector_add_kernal.cu',
                ],
                include_dirs=[
                    'src/vector_add.h',
                ]
            ),
            ],)
    
    
