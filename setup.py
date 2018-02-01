# System imports
import os
import sys
import subprocess
import string
import platform

from setuptools import setup, find_packages, Command

# Version number
major = 1
minor = 0

on_rtd = os.environ.get('READTHEDOCS') == 'True'


if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)


if on_rtd:
    REQUIREMENTS = []
else:
    REQUIREMENTS = [
        "numpy",
        "scipy",
        "matplotlib",
        "mesh_generation==0.1",
        "pyyaml",
        "h5py"        
    ]

dependency_links = ["git+ssh.//git@bitbucket.org:finsberg/mesh_generation.git#egg=mesh_generation-0.1"]

setup(name = "pulse_adjoint",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjointable cardiac mechanics solver and data assimilator.
      """,
      author = "Henrik Finsberg",
      author_email = "henriknf@simula.no",
      license="LGPL version 3 or later",
      install_requires=REQUIREMENTS,
      dependency_links=dependency_links,
      packages = ["pulse_adjoint",
                  "pulse_adjoint.models",
                  "pulse_adjoint.postprocess",
                  "pulse_adjoint.patient_data",
                  "pulse_adjoint.io",
                  'pulse_adjoint.example_meshes',
                  "pulse_adjoint.unloading"],
      package_data={'pulse_adjoint.example_meshes':  ["*.h5"]},
      package_dir = {"pulse_adjoint": "pulse_adjoint"},
      )
